from __future__ import annotations

import math
import os
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from .model import TextToLatentRFDiT, precompute_freqs_cis

_IR_CACHE_VERSION = 1
_MAX_DYNAMIC_SEQUENCE_LENGTH = 4096
_LATENT_SEQUENCE_BUCKETS = (
    8,
    12,
    16,
    24,
    32,
    48,
    64,
    96,
    128,
    192,
    256,
    384,
    512,
    768,
    1024,
    1536,
    2048,
    3072,
    4096,
)


def _bucket_latent_input(x_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if x_t.ndim != 3 or x_t.shape[1] == 0:
        raise ValueError(f"Latent input must have shape (B,S,D) with S > 0, got {x_t.shape}.")

    sequence_length = x_t.shape[1]
    bucket_length = next(
        (length for length in _LATENT_SEQUENCE_BUCKETS if length >= sequence_length),
        sequence_length,
    )
    latent_mask = torch.zeros(
        (x_t.shape[0], bucket_length),
        dtype=torch.bool,
        device=x_t.device,
    )
    latent_mask[:, :sequence_length] = True
    if bucket_length == sequence_length:
        return x_t, latent_mask

    padding = x_t.new_zeros(
        (x_t.shape[0], bucket_length - sequence_length, x_t.shape[2])
    )
    return torch.cat((x_t, padding), dim=1), latent_mask


def _safe_cache_component(value: object) -> str:
    return "".join(
        char if char.isalnum() or char in {"-", ".", "+", "_"} else "_"
        for char in str(value)
    )


def _replace_squared_sine_with_polynomial(ov: Any, model: Any) -> None:
    constants_by_type: dict[str, tuple[Any, ...]] = {}
    sin_nodes = [node for node in model.get_ops() if node.get_type_name() == "Sin"]

    for sin_node in sin_nodes:
        consumers = list(sin_node.output(0).get_target_inputs())
        if len(consumers) != 1 or consumers[0].get_node().get_type_name() != "Power":
            raise RuntimeError("DACVAE decoder Sin must be consumed by a single Power operation.")
        power_node = consumers[0].get_node()
        exponent = power_node.input_value(1).get_node()
        if exponent.get_type_name() != "Constant" or exponent.get_vector() != [2.0]:
            raise RuntimeError("DACVAE decoder Sin must be squared by a constant exponent of 2.")

        value = sin_node.input_value(0)
        element_type = value.get_element_type()
        type_name = element_type.get_type_name()
        constants = constants_by_type.get(type_name)
        if constants is None:
            # Reduce by the pi-period of sin^2, then evaluate its degree-12 Taylor series.
            coefficients = (
                1.0 / math.pi,
                0.5,
                math.pi,
                -2.0 / 467775.0,
                2.0 / 14175.0,
                -1.0 / 315.0,
                2.0 / 45.0,
                -1.0 / 3.0,
                1.0,
            )
            constants = tuple(
                ov.opset13.constant(coefficient, element_type) for coefficient in coefficients
            )
            constants_by_type[type_name] = constants
        inv_pi, half, pi, *polynomial_coefficients = constants

        periods = ov.opset13.floor(
            ov.opset13.add(ov.opset13.multiply(value, inv_pi), half)
        )
        reduced = ov.opset13.subtract(value, ov.opset13.multiply(periods, pi))
        squared = ov.opset13.multiply(reduced, reduced)
        polynomial = polynomial_coefficients[0]
        for coefficient in polynomial_coefficients[1:]:
            polynomial = ov.opset13.add(
                coefficient,
                ov.opset13.multiply(squared, polynomial),
            )
        approximation = ov.opset13.multiply(squared, polynomial)
        approximation.set_friendly_name(power_node.get_friendly_name())
        approximation.output(0).set_names(power_node.output(0).get_names())
        power_node.output(0).replace(approximation.output(0))

    model.validate_nodes_and_infer_types()


class _OpenVINOBackendBase(ABC):
    def __init__(self, *, device: str, checkpoint_path: str | Path) -> None:
        try:
            import openvino as ov
        except ImportError as exc:
            raise RuntimeError(
                "OpenVINO is required for NPU inference. Install the openvino package."
            ) from exc

        self.ov = ov
        self.device = str(device).strip().upper()
        self.core = ov.Core()
        if self.device not in self.core.available_devices:
            available = ", ".join(self.core.available_devices) or "none"
            raise RuntimeError(
                f"OpenVINO device {self.device!r} is unavailable. Available devices: {available}."
            )
        self._ir_dir: Path | None = None
        self._ir_model = None
        self._dynamic_ir_unavailable = False
        self._compiled_models: dict[
            tuple[tuple[tuple[int, ...], torch.dtype], ...], Any
        ] = {}
        self._initialize_cache(Path(checkpoint_path))

    @property
    @abstractmethod
    def _ir_filename(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def _supports_dynamic_ir(self, inputs: tuple[torch.Tensor, ...]) -> bool:
        raise NotImplementedError

    @abstractmethod
    def _export_dynamic_ir(self, inputs: tuple[torch.Tensor, ...]):
        raise NotImplementedError

    @abstractmethod
    def _convert_static(self, inputs: tuple[torch.Tensor, ...]):
        raise NotImplementedError

    @property
    def _ir_xml_path(self) -> Path:
        if self._ir_dir is None:
            raise RuntimeError("OpenVINO IR cache is not initialized.")
        return self._ir_dir / self._ir_filename

    def _initialize_cache(self, checkpoint_path: Path) -> None:
        try:
            checkpoint_path = checkpoint_path.expanduser()
            checkpoint_stat = checkpoint_path.stat()
            cache_root = checkpoint_path.with_name(f"{checkpoint_path.name}.openvino")
            torch_version = _safe_cache_component(torch.__version__)
            openvino_version = _safe_cache_component(self.ov.__version__)
            cache_key = (
                f"v{_IR_CACHE_VERSION}"
                f"-size{checkpoint_stat.st_size}"
                f"-mtime{checkpoint_stat.st_mtime_ns}"
                f"-torch{torch_version}"
                f"-ov{openvino_version}"
            )
            self._ir_dir = cache_root / cache_key
            compiled_dir = self._ir_dir / "compiled"
            compiled_dir.mkdir(parents=True, exist_ok=True)
            self.core.set_property(
                self.device,
                {self.ov.properties.cache_dir(): str(compiled_dir)},
            )
        except (OSError, RuntimeError) as exc:
            warnings.warn(
                f"Could not initialize OpenVINO cache next to {checkpoint_path}: {exc}",
                stacklevel=2,
            )
            self._ir_dir = None

    def _load_cached_ir_file(self, path: Path):
        try:
            if not path.is_file():
                return None
            return self.core.read_model(path)
        except (OSError, RuntimeError):
            return None

    def _save_ir_file(self, ov_model, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_xml = path.with_name(f"{path.stem}.{os.getpid()}.tmp.xml")
        tmp_bin = tmp_xml.with_suffix(".bin")
        try:
            self.ov.save_model(ov_model, tmp_xml, compress_to_fp16=False)
            tmp_bin.replace(path.with_suffix(".bin"))
            tmp_xml.replace(path)
        finally:
            for tmp_path in (tmp_xml, tmp_bin):
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass

    def _get_dynamic_ir(self, inputs: tuple[torch.Tensor, ...]):
        if self._ir_model is not None:
            return self._ir_model
        if self._ir_dir is None:
            raise RuntimeError("OpenVINO IR cache is not initialized.")

        path = self._ir_xml_path
        cached = self._load_cached_ir_file(path)
        if cached is not None:
            self._ir_model = cached
            return cached

        model = self._export_dynamic_ir(inputs)
        try:
            self._save_ir_file(model, path)
        except (OSError, RuntimeError) as exc:
            warnings.warn(f"Could not save OpenVINO IR cache: {exc}", stacklevel=2)
        else:
            cached = self._load_cached_ir_file(path)
            if cached is not None:
                model = cached
        self._ir_model = model
        return model

    def _compile(self, inputs: tuple[torch.Tensor, ...]):
        if (
            self._ir_dir is not None
            and not self._dynamic_ir_unavailable
            and self._supports_dynamic_ir(inputs)
        ):
            try:
                ov_model = self._get_dynamic_ir(inputs).clone()
            except Exception as exc:
                warnings.warn(
                    f"Dynamic OpenVINO IR export failed; using a static graph: {exc}",
                    stacklevel=2,
                )
                self._dynamic_ir_unavailable = True
                ov_model = self._convert_static(inputs)
        else:
            ov_model = self._convert_static(inputs)
        ov_model.reshape(
            {
                port.any_name: list(tensor.shape)
                for port, tensor in zip(ov_model.inputs, inputs, strict=True)
            }
        )
        return self.core.compile_model(ov_model, self.device)

    def _get_compiled_model(self, inputs: tuple[torch.Tensor, ...]):
        signature = tuple((tuple(tensor.shape), tensor.dtype) for tensor in inputs)
        compiled_model = self._compiled_models.get(signature)
        if compiled_model is None:
            compiled_model = self._compile(inputs)
            self._compiled_models[signature] = compiled_model
        return compiled_model


class _RFDiTExportWrapper(torch.nn.Module):
    def __init__(self, model: TextToLatentRFDiT) -> None:
        super().__init__()
        self.model = model
        self.use_speaker = model.cfg.use_speaker_condition_resolved
        self.use_caption = model.cfg.use_caption_condition

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        latent_mask: torch.Tensor,
        text_state: torch.Tensor,
        text_mask: torch.Tensor,
        speaker_state: torch.Tensor,
        speaker_mask: torch.Tensor,
        caption_state: torch.Tensor,
        caption_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.model.forward_with_encoded_conditions(
            x_t=x_t,
            t=t,
            text_state=text_state,
            text_mask=text_mask,
            speaker_state=speaker_state if self.use_speaker else None,
            speaker_mask=speaker_mask if self.use_speaker else None,
            caption_state=caption_state if self.use_caption else None,
            caption_mask=caption_mask if self.use_caption else None,
            latent_mask=latent_mask,
            context_kv_cache=None,
        )


class _MeanFlowDiTExportWrapper(torch.nn.Module):
    def __init__(self, model: TextToLatentRFDiT) -> None:
        super().__init__()
        self.model = model
        self.use_speaker = model.cfg.use_speaker_condition_resolved
        self.use_caption = model.cfg.use_caption_condition

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        delta_t: torch.Tensor,
        latent_mask: torch.Tensor,
        text_state: torch.Tensor,
        text_mask: torch.Tensor,
        speaker_state: torch.Tensor,
        speaker_mask: torch.Tensor,
        caption_state: torch.Tensor,
        caption_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.model.forward_with_encoded_conditions(
            x_t=x_t,
            t=t,
            delta_t=delta_t,
            text_state=text_state,
            text_mask=text_mask,
            speaker_state=speaker_state if self.use_speaker else None,
            speaker_mask=speaker_mask if self.use_speaker else None,
            caption_state=caption_state if self.use_caption else None,
            caption_mask=caption_mask if self.use_caption else None,
            latent_mask=latent_mask,
            context_kv_cache=None,
        )


class _DACVAEDecoderExportWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.out_proj = model.quantizer.out_proj
        self.decoder_layers = model.decoder.model
        # DACVAECodec uses forward_no_conv(), whose temporary module mutation is not exportable.
        self.output_layers = model.decoder.wm_model.encoder_block.pre[:-1]

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        x = self.out_proj(latent)
        for layer in self.decoder_layers:
            x = layer(x)
        return self.output_layers(x)


class OpenVINODiTBackend(_OpenVINOBackendBase):
    def __init__(
        self,
        model: TextToLatentRFDiT,
        *,
        device: str,
        checkpoint_path: str | Path,
    ) -> None:
        self.model = model.eval()
        self.is_meanflow = self.model.cfg.flow_parameterization == "meanflow"
        super().__init__(device=device, checkpoint_path=checkpoint_path)

    @property
    def _ir_filename(self) -> str:
        return "meanflow_dit.xml" if self.is_meanflow else "rf_dit.xml"

    def _supports_dynamic_ir(self, inputs: tuple[torch.Tensor, ...]) -> bool:
        sequence_indexes = (0, 4, 6, 8) if self.is_meanflow else (0, 3, 5, 7)
        sequence_inputs = tuple(inputs[index] for index in sequence_indexes)
        return all(
            tensor.shape[1] <= _MAX_DYNAMIC_SEQUENCE_LENGTH for tensor in sequence_inputs
        )

    def _dynamic_export_inputs(
        self, inputs: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, ...]:
        condition_sequences = (
            True,
            True,
            bool(self.model.cfg.use_speaker_condition_resolved),
            bool(self.model.cfg.use_speaker_condition_resolved),
            bool(self.model.cfg.use_caption_condition),
            bool(self.model.cfg.use_caption_condition),
        )
        use_sequence = (
            (True, False, False, True, *condition_sequences)
            if self.is_meanflow
            else (True, False, True, *condition_sequences)
        )
        example_inputs: list[torch.Tensor] = []
        for index, (tensor, dynamic_sequence) in enumerate(zip(inputs, use_sequence, strict=True)):
            shape = list(tensor.shape)
            shape[0] = 2
            if dynamic_sequence:
                shape[1] = max(2, shape[1])
            if tensor.dtype == torch.bool:
                example = torch.ones(shape, dtype=tensor.dtype, device=tensor.device)
            else:
                example = torch.zeros(shape, dtype=tensor.dtype, device=tensor.device)
            if index == 1 or (self.is_meanflow and index == 2):
                example.fill_(0.5)
            example_inputs.append(example)
        return tuple(example_inputs)

    def _export_dynamic_ir(self, inputs: tuple[torch.Tensor, ...]):
        wrapper = self._export_wrapper()
        export_inputs = self._dynamic_export_inputs(inputs)
        batch = torch.export.Dim("batch", min=1)
        latent = torch.export.Dim(
            "latent", min=1, max=_MAX_DYNAMIC_SEQUENCE_LENGTH
        )
        text = torch.export.Dim("text", min=1, max=_MAX_DYNAMIC_SEQUENCE_LENGTH)
        speaker = torch.export.Dim(
            "speaker", min=1, max=_MAX_DYNAMIC_SEQUENCE_LENGTH
        )
        caption = torch.export.Dim(
            "caption", min=1, max=_MAX_DYNAMIC_SEQUENCE_LENGTH
        )
        speaker_shapes = (
            {0: batch, 1: speaker}
            if self.model.cfg.use_speaker_condition_resolved
            else {0: batch}
        )
        caption_shapes = (
            {0: batch, 1: caption}
            if self.model.cfg.use_caption_condition
            else {0: batch}
        )
        condition_shapes = (
            {0: batch, 1: text},
            {0: batch, 1: text},
            speaker_shapes,
            speaker_shapes,
            caption_shapes,
            caption_shapes,
        )
        dynamic_shapes = (
            {0: batch, 1: latent},
            {0: batch},
            {0: batch},
            {0: batch, 1: latent},
            *condition_shapes,
        ) if self.is_meanflow else (
            {0: batch, 1: latent},
            {0: batch},
            {0: batch, 1: latent},
            *condition_shapes,
        )

        previous_rope_cache = self.model._freqs_cis_cache
        self.model._freqs_cis_cache = precompute_freqs_cis(
            self.model.head_dim,
            _MAX_DYNAMIC_SEQUENCE_LENGTH,
        ).to(device=previous_rope_cache.device)
        try:
            with torch.inference_mode():
                exported = torch.export.export(
                    wrapper,
                    export_inputs,
                    dynamic_shapes=dynamic_shapes,
                )
        finally:
            self.model._freqs_cis_cache = previous_rope_cache
        return self.ov.convert_model(exported)

    def _prepare_inputs(
        self,
        *,
        x_t: torch.Tensor,
        t: torch.Tensor,
        delta_t: torch.Tensor | None,
        text_state: torch.Tensor,
        text_mask: torch.Tensor,
        speaker_state: torch.Tensor | None,
        speaker_mask: torch.Tensor | None,
        caption_state: torch.Tensor | None,
        caption_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, ...]:
        x_t, latent_mask = _bucket_latent_input(x_t)
        batch = x_t.shape[0]
        if self.model.cfg.use_speaker_condition_resolved:
            if speaker_state is None or speaker_mask is None:
                raise ValueError("Speaker state and mask are required by this DiT model.")
        else:
            speaker_state = torch.empty((batch, 1, 1), dtype=x_t.dtype, device=x_t.device)
            speaker_mask = torch.zeros((batch, 1), dtype=torch.bool, device=x_t.device)
        if self.model.cfg.use_caption_condition:
            if caption_state is None or caption_mask is None:
                raise ValueError("Caption state and mask are required by this DiT model.")
        else:
            caption_state = torch.empty((batch, 1, 1), dtype=x_t.dtype, device=x_t.device)
            caption_mask = torch.zeros((batch, 1), dtype=torch.bool, device=x_t.device)
        condition_inputs = (
            text_state,
            text_mask,
            speaker_state,
            speaker_mask,
            caption_state,
            caption_mask,
        )
        if self.is_meanflow:
            if delta_t is None:
                raise ValueError("delta_t is required by this MeanFlow DiT model.")
            return (x_t, t, delta_t, latent_mask, *condition_inputs)
        if delta_t is not None:
            raise ValueError("delta_t is not supported by this RF DiT model.")
        return (
            x_t,
            t,
            latent_mask,
            *condition_inputs,
        )

    def _export_wrapper(self) -> torch.nn.Module:
        if self.is_meanflow:
            return _MeanFlowDiTExportWrapper(self.model).eval()
        return _RFDiTExportWrapper(self.model).eval()

    def _convert_static(self, inputs: tuple[torch.Tensor, ...]):
        wrapper = self._export_wrapper()
        with torch.inference_mode():
            exported = torch.export.export(wrapper, inputs)
        return self.ov.convert_model(exported)

    def __call__(
        self,
        *,
        x_t: torch.Tensor,
        t: torch.Tensor,
        text_state: torch.Tensor,
        text_mask: torch.Tensor,
        speaker_state: torch.Tensor | None,
        speaker_mask: torch.Tensor | None,
        caption_state: torch.Tensor | None = None,
        caption_mask: torch.Tensor | None = None,
        context_kv_cache: list[tuple[torch.Tensor, ...]] | None = None,
        delta_t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del context_kv_cache
        sequence_length = x_t.shape[1]
        inputs = self._prepare_inputs(
            x_t=x_t,
            t=t,
            delta_t=delta_t,
            text_state=text_state,
            text_mask=text_mask,
            speaker_state=speaker_state,
            speaker_mask=speaker_mask,
            caption_state=caption_state,
            caption_mask=caption_mask,
        )
        compiled_model = self._get_compiled_model(inputs)
        output = compiled_model(inputs)[0]
        return torch.from_numpy(output[:, :sequence_length].copy()).to(
            device=x_t.device,
            dtype=x_t.dtype,
        )


class OpenVINOPretrainedTextBackboneBackend(_OpenVINOBackendBase):
    def __init__(
        self,
        model: TextToLatentRFDiT,
        *,
        device: str,
        checkpoint_path: str | Path,
    ) -> None:
        backbone = model.pretrained_text_backbone
        if backbone is None:
            raise ValueError(
                "OpenVINO pretrained text backbone requires text_encoder_type='pretrained'."
            )
        self.backbone = backbone.eval()
        self.hidden_size = int(self.backbone.hidden_size)
        self.dtype = next(self.backbone.parameters()).dtype
        super().__init__(device=device, checkpoint_path=checkpoint_path)

    @property
    def _ir_filename(self) -> str:
        return "pretrained_text_backbone.xml"

    def _supports_dynamic_ir(self, inputs: tuple[torch.Tensor, ...]) -> bool:
        return inputs[0].shape[1] <= _MAX_DYNAMIC_SEQUENCE_LENGTH

    def _dynamic_export_inputs(
        self,
        inputs: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids, mask = inputs
        shape = (2, max(2, input_ids.shape[1]))
        return (
            torch.zeros(shape, dtype=input_ids.dtype, device=input_ids.device),
            torch.ones(shape, dtype=mask.dtype, device=mask.device),
        )

    def _export_dynamic_ir(self, inputs: tuple[torch.Tensor, torch.Tensor]):
        export_inputs = self._dynamic_export_inputs(inputs)
        batch = torch.export.Dim("batch", min=1)
        sequence = torch.export.Dim(
            "sequence",
            min=1,
            max=_MAX_DYNAMIC_SEQUENCE_LENGTH,
        )
        dynamic_shapes = (
            {0: batch, 1: sequence},
            {0: batch, 1: sequence},
        )
        with torch.inference_mode():
            exported = torch.export.export(
                self.backbone,
                export_inputs,
                dynamic_shapes=dynamic_shapes,
            )
        return self.ov.convert_model(exported)

    def _convert_static(self, inputs: tuple[torch.Tensor, torch.Tensor]):
        with torch.inference_mode():
            exported = torch.export.export(self.backbone, inputs)
        return self.ov.convert_model(exported)

    def __call__(
        self,
        input_ids: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if input_ids.ndim != 2 or mask.ndim != 2:
            raise ValueError(
                "Pretrained text backbone inputs must have shape (B,S): "
                f"input_ids={tuple(input_ids.shape)} mask={tuple(mask.shape)}"
            )
        if input_ids.shape != mask.shape:
            raise ValueError(
                "Pretrained text backbone input shape mismatch: "
                f"input_ids={tuple(input_ids.shape)} mask={tuple(mask.shape)}"
            )
        if not bool(mask.any()):
            return torch.zeros(
                (*input_ids.shape, self.hidden_size),
                dtype=self.dtype,
                device=input_ids.device,
            )
        inputs = (input_ids, mask)
        compiled_model = self._get_compiled_model(inputs)
        output = compiled_model(inputs)[0]
        return torch.from_numpy(output.copy()).to(device=input_ids.device, dtype=self.dtype)


class OpenVINODACVAEDecoderBackend(_OpenVINOBackendBase):
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        device: str,
        checkpoint_path: str | Path,
    ) -> None:
        self.model = model.eval()
        self.wrapper = _DACVAEDecoderExportWrapper(self.model).eval()
        self.dtype = next(self.model.parameters()).dtype
        super().__init__(device=device, checkpoint_path=checkpoint_path)

    @property
    def _ir_filename(self) -> str:
        return "dacvae_decoder_poly.xml"

    def _convert_exported(self, exported: torch.export.ExportedProgram):
        model = self.ov.convert_model(exported)
        _replace_squared_sine_with_polynomial(self.ov, model)
        return model

    def _supports_dynamic_ir(self, inputs: tuple[torch.Tensor, ...]) -> bool:
        return inputs[0].shape[2] <= _MAX_DYNAMIC_SEQUENCE_LENGTH

    def _export_dynamic_ir(self, inputs: tuple[torch.Tensor, ...]):
        latent = inputs[0]
        export_input = torch.zeros(
            (2, latent.shape[1], max(2, latent.shape[2])),
            dtype=latent.dtype,
            device=latent.device,
        )
        batch = torch.export.Dim("batch", min=1)
        sequence = torch.export.Dim(
            "latent",
            min=1,
            max=_MAX_DYNAMIC_SEQUENCE_LENGTH,
        )
        with torch.inference_mode():
            exported = torch.export.export(
                self.wrapper,
                (export_input,),
                dynamic_shapes=({0: batch, 2: sequence},),
            )
        return self._convert_exported(exported)

    def _convert_static(self, inputs: tuple[torch.Tensor, ...]):
        with torch.inference_mode():
            exported = torch.export.export(self.wrapper, inputs)
        return self._convert_exported(exported)

    def __call__(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 3:
            raise ValueError(f"Expected latent ndim=3, got shape={tuple(latent.shape)}")
        z = latent.transpose(1, 2).contiguous().to(device="cpu", dtype=self.dtype)
        compiled_model = self._get_compiled_model((z,))
        output = compiled_model((z,))[0]
        return torch.from_numpy(output.copy())


def create_dit_backend(
    model: TextToLatentRFDiT,
    *,
    device: str,
    checkpoint_path: str | Path,
) -> Callable[..., torch.Tensor]:
    return OpenVINODiTBackend(
        model,
        device=device,
        checkpoint_path=checkpoint_path,
    )


def create_pretrained_text_backbone_backend(
    model: TextToLatentRFDiT,
    *,
    device: str,
    checkpoint_path: str | Path,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    return OpenVINOPretrainedTextBackboneBackend(
        model,
        device=device,
        checkpoint_path=checkpoint_path,
    )


def create_dacvae_decoder_backend(
    model: torch.nn.Module,
    *,
    device: str,
    checkpoint_path: str | Path,
) -> Callable[[torch.Tensor], torch.Tensor]:
    return OpenVINODACVAEDecoderBackend(
        model,
        device=device,
        checkpoint_path=checkpoint_path,
    )
