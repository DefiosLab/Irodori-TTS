from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import torch

from .model import TextToLatentRFDiT, precompute_freqs_cis

_IR_CACHE_VERSION = 1
_MAX_DYNAMIC_SEQUENCE_LENGTH = 4096
_CONDITION_SEQUENCE_PROFILES = (
    # Text, speaker (after reference encoding), caption.
    (32, 64, 32),
    (128, 256, 128),
    (512, 1024, 512),
)
# Five profiles cover the default generation limit of 30 seconds at 25 latent frames/s.
_LATENT_SEQUENCE_BUCKETS = (48, 128, 256, 512, 768)
_CODEC_ENCODER_BUCKETS = (40, 128, 256)
_CODEC_DECODER_BUCKETS = (40, 80)
_WATERMARK_SEQUENCE_BUCKETS = (32, 64, 192, 384, 672)
# Per backend; retain both decoder windows and common text/caption profiles.
_MAX_RESIDENT_NPU_MODELS = 2
_MAX_CONTEXT_KV_CACHES = 4  # Conditional input plus text/speaker/caption CFG branches.


def _condition_profiles_up_to(lengths: tuple[int, int, int]) -> tuple[tuple[int, int, int], ...]:
    largest = _select_condition_profile(*lengths)
    profiles = []
    for profile in _CONDITION_SEQUENCE_PROFILES:
        profiles.append(profile)
        if profile == largest:
            return tuple(profiles)
    while profiles[-1] != largest:
        profiles.append(tuple(capacity * 2 for capacity in profiles[-1]))
    return tuple(profiles)


def _latent_buckets_up_to(length: int) -> tuple[int, ...]:
    if length <= 0:
        raise ValueError("Maximum latent sequence length must be positive.")
    largest = _select_latent_bucket(length)
    buckets = [capacity for capacity in _LATENT_SEQUENCE_BUCKETS if capacity <= largest]
    while buckets[-1] < largest:
        buckets.append(_select_latent_bucket(buckets[-1] + 1))
    return tuple(buckets)


def _iter_batch_inputs(inputs: tuple[torch.Tensor, ...]) -> Iterator[tuple[torch.Tensor, ...]]:
    batch_size = inputs[0].shape[0]
    if batch_size == 0 or any(tensor.shape[0] != batch_size for tensor in inputs):
        raise ValueError("OpenVINO inputs must have matching, non-empty batch dimensions.")
    # Keep compiled batch shapes at one, including independent CFG branches.
    for index in range(batch_size):
        yield tuple(tensor[index : index + 1].contiguous() for tensor in inputs)


def _select_condition_profile(
    text_length: int,
    speaker_length: int,
    caption_length: int,
) -> tuple[int, int, int]:
    lengths = (text_length, speaker_length, caption_length)
    for profile in _CONDITION_SEQUENCE_PROFILES:
        if all(length <= capacity for length, capacity in zip(lengths, profile, strict=True)):
            return profile

    # Keep oversized inputs intact without introducing independent condition buckets.
    largest = _CONDITION_SEQUENCE_PROFILES[-1]
    scale = max(
        (length + capacity - 1) // capacity
        for length, capacity in zip(lengths, largest, strict=True)
    )
    scale = 1 << (scale - 1).bit_length()
    return tuple(capacity * scale for capacity in largest)


def _pad_condition_sequence(
    state: torch.Tensor,
    mask: torch.Tensor,
    length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if state.ndim not in (2, 3) or mask.ndim != 2 or state.shape[:2] != mask.shape:
        raise ValueError(
            "Condition state and mask must have matching (B,S) dimensions: "
            f"state={tuple(state.shape)} mask={tuple(mask.shape)}"
        )
    padding_length = length - state.shape[1]
    if padding_length < 0:
        raise ValueError(f"Condition length {state.shape[1]} exceeds profile capacity {length}.")
    if padding_length == 0:
        return state, mask
    padding = state.new_zeros((state.shape[0], padding_length, *state.shape[2:]))
    mask_padding = mask.new_zeros((mask.shape[0], padding_length))
    return torch.cat((state, padding), dim=1), torch.cat((mask, mask_padding), dim=1)


def _bucket_latent_input(x_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if x_t.ndim != 3 or x_t.shape[1] == 0:
        raise ValueError(f"Latent input must have shape (B,S,D) with S > 0, got {x_t.shape}.")

    sequence_length = x_t.shape[1]
    bucket_length = _select_latent_bucket(sequence_length)
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


def _select_latent_bucket(length: int) -> int:
    for capacity in _LATENT_SEQUENCE_BUCKETS:
        if length <= capacity:
            return capacity
    return 1 << (length - 1).bit_length()


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
            raise RuntimeError("DACVAE Sin must be consumed by a single Power operation.")
        power_node = consumers[0].get_node()
        exponent = power_node.input_value(1).get_node()
        if exponent.get_type_name() != "Constant" or exponent.get_vector() != [2.0]:
            raise RuntimeError("DACVAE Sin must be squared by a constant exponent of 2.")

        value = sin_node.input_value(0)
        element_type = value.get_element_type()
        type_name = element_type.get_type_name()
        constants = constants_by_type.get(type_name)
        if constants is None:
            coefficients = (
                1.0 / math.pi,
                math.pi,
                -0.002423022697067143,
                0.04295343047848664,
                -0.33214428172806304,
                0.999703164102684,
            )
            constants = tuple(
                ov.opset13.constant(coefficient, element_type) for coefficient in coefficients
            )
            constants_by_type[type_name] = constants

        inv_pi, pi, *polynomial_coefficients = constants
        periods = ov.opset13.round(ov.opset13.multiply(value, inv_pi), "half_to_even")
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


def _pack_dacvae_polynomials(ov: Any, model: Any) -> dict[str, dict[str, Any]]:
    op = ov.opset13
    binary = {"Add": op.add, "Multiply": op.multiply, "Subtract": op.subtract}
    strategy = {}
    finals = [
        node
        for node in model.get_ordered_ops()
        if node.get_type_name() == "Multiply"
        and "/aten.pow.Tensor_Scalar/Power" in node.get_friendly_name()
    ]
    for final in finals:
        squared = final.input_value(0).get_node()
        reduced = squared.input_value(0).get_node()
        value = reduced.input_value(0)
        shape = list(value.get_shape())
        height = {192: 75, 96: 150}.get(shape[1])
        if height is None or shape[-1] % (height * 16):
            continue
        width = shape[-1] // height
        prefix = f"{final.get_friendly_name()}/minimax/"
        scaled_periods = reduced.input_value(1).get_node()
        periods = scaled_periods.input_value(0).get_node()
        names = {
            final: "poly/out",
            squared: "poly/q",
            reduced: "range/reduced",
            scaled_periods: "range/period_pi",
            periods: "range/round",
            periods.input_value(0).get_node(): "range/scaled",
        }
        polynomial = final.input_value(1).get_node()
        for index in (2, 1, 0):
            names[polynomial] = f"poly/add_{index}"
            product = polynomial.input_value(1).get_node()
            names[product] = f"poly/mul_{index}"
            polynomial = product.input_value(1).get_node()

        packed_shape = op.constant(shape[:2] + [height, width], name=prefix + "packed_shape")
        copied = {value.get_node(): op.reshape(value, packed_shape, False, name=prefix + "pack")}

        def clone(node, copied=copied, names=names, prefix=prefix):
            if node in copied:
                return copied[node]
            kind = node.get_type_name()
            inputs = [clone(port.get_node()) for port in node.input_values()]
            if kind == "Constant":
                result = op.constant(
                    node.get_data().reshape(1, 1, 1, 1), node.get_output_element_type(0)
                )
            elif kind in binary:
                result = binary[kind](*inputs)
            elif kind == "Round":
                result = op.round(inputs[0], node.get_attributes()["mode"])
            else:
                raise RuntimeError(f"Unexpected DACVAE polynomial operation: {kind}")
            result.set_friendly_name(prefix + names.get(node, node.get_friendly_name()))
            copied[node] = result
            return result

        original_shape = op.constant(shape, name=prefix + "original_shape")
        result = op.reshape(clone(final), original_shape, False)
        result.set_friendly_name(prefix + "restore_shape")
        result.output(0).set_names(final.output(0).get_names())
        final.output(0).replace(result.output(0))
        locations = (
            "range/scaled?t_Multiply/as_scaleshift",
            "range/round?t_Round",
            "range/period_pi?t_Multiply/neg/as_scaleshift",
            "range/reduced?t_Subtract/as_add",
            "poly/q?t_Multiply",
            "poly/add_0?t_Add",
            "poly/mul_1?t_Multiply",
            "poly/add_1?t_Add",
            "poly/mul_2?t_Multiply",
            "poly/add_2?t_Add",
            "poly/out?t_Multiply",
        )
        for location in locations:
            strategy[prefix + location] = {
                "multiClusterStrategy": "SplitOverHeight",
                "verticalFusion": "True",
                "verticalFusionHash": prefix,
                "VFScenario": "MINIMAL",
                # 16-wide tiles leave CMX space for q and the live intermediates.
                "tilingStrategy": {"N": 1, "C": 1, "H": 1, "W": width // 16},
            }
    model.validate_nodes_and_infer_types()
    return strategy


def _codec_context_frames(model: Any, frame_size: int) -> int:
    # Follow the longest convolution path, including residual branches, in input units.
    fields: dict[Any, tuple[float, float, float]] = {model.input(0).get_node(): (1.0, 0.0, 0.0)}
    for node in model.get_ordered_ops():
        if node in fields or node.get_type_name() == "Constant":
            continue
        inputs = [
            fields[value.get_node()] for value in node.input_values() if value.get_node() in fields
        ]
        if not inputs or len(node.output(0).get_shape()) != 3:
            continue
        step = inputs[0][0]
        left = min(field[1] for field in inputs)
        right = max(field[2] for field in inputs)
        kind = node.get_type_name()
        if kind in {"Convolution", "ConvolutionBackpropData"}:
            attrs = node.get_attributes()
            stride = attrs["strides"][0]
            padding = attrs["pads_begin"][0]
            kernel = (node.input_value(1).get_shape()[-1] - 1) * attrs["dilations"][0]
            if kind == "Convolution":
                left -= padding * step
                right += (kernel - padding) * step
                step *= stride
            else:
                step /= stride
                left += (padding - kernel) * step
                right += padding * step
        fields[node] = (step, left, right)
    _, left, right = fields[model.output(0).get_node()]
    # Include a full frame around fractional positions in transposed convolutions.
    return math.ceil(max(-left, right) / frame_size) + 1


def _mask_codec_boundaries(ov: Any, model: Any, frames: int):
    # Clear biased padding at every resolution before later convolutions can consume it.
    masks: dict[int, Any] = {}
    convolutions = [
        node
        for node in model.get_ordered_ops()
        if node.get_type_name() in {"Convolution", "ConvolutionBackpropData"}
    ]
    for conv in convolutions:
        output = conv.output(0)
        targets = list(output.get_target_inputs())
        if len(targets) == 1:
            target = targets[0]
            bias = target.get_node()
            if (
                bias.get_type_name() == "Add"
                and bias.input_value(1 - target.get_index()).get_node().get_type_name()
                == "Constant"
            ):
                output = bias.output(0)
        length = output.get_shape()[-1]
        if length % frames:
            raise RuntimeError("DACVAE convolution lengths must align with latent frames.")
        scale = length // frames
        if scale not in masks:
            masks[scale] = ov.opset13.parameter(
                [1, 1, length], output.get_element_type(), name=f"codec_valid_{scale}"
            )
        targets = list(output.get_target_inputs())
        masked = ov.opset13.multiply(output, masks[scale])
        masked.set_friendly_name(f"{conv.get_friendly_name()}/boundary_mask")
        for target in targets:
            target.replace_source_output(masked.output(0))
    result = ov.Model(
        model.get_results(),
        model.get_parameters() + [masks[scale] for scale in sorted(masks)],
        model.get_friendly_name(),
    )
    result.validate_nodes_and_infer_types()
    return result


class _OpenVINOBackendBase(ABC):
    def __init__(self, *, device: str, checkpoint_path: str | Path, core: Any = None) -> None:
        try:
            import openvino as ov
        except ImportError as exc:
            raise RuntimeError(
                "OpenVINO is required for NPU inference. Install the openvino package."
            ) from exc

        self.ov = ov
        self.device = str(device).strip().upper()
        self.core = ov.Core() if core is None else core
        if self.device not in self.core.available_devices:
            available = ", ".join(self.core.available_devices) or "none"
            raise RuntimeError(
                f"OpenVINO device {self.device!r} is unavailable. Available devices: {available}."
            )
        self._ir_dir: Path | None = None
        self._ir_model = None
        self._dynamic_ir_unavailable = False
        self._compiled_models: OrderedDict[
            tuple[tuple[tuple[int, ...], torch.dtype], ...], Any
        ] = OrderedDict()
        self._resident_model_limit = _MAX_RESIDENT_NPU_MODELS if self.device == "NPU" else None
        self._warmed_profiles: set[tuple[tuple[tuple[int, ...], torch.dtype], ...]] = set()
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

    @property
    def _cache_mode(self) -> str:
        return "OPTIMIZE_SIZE"

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
            properties = {self.ov.properties.cache_dir(): str(compiled_dir)}
            if self.device == "NPU":
                properties[self.ov.properties.cache_mode()] = self._cache_mode
            self.core.set_property(self.device, properties)
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

    def _prepare_model(self, inputs: tuple[torch.Tensor, ...]):
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
        return ov_model

    def _compile(self, inputs: tuple[torch.Tensor, ...]):
        return self.core.compile_model(self._prepare_model(inputs), self.device)

    def _get_compiled_model(self, inputs: tuple[torch.Tensor, ...]):
        signature = tuple((tuple(tensor.shape), tensor.dtype) for tensor in inputs)
        compiled_model = self._compiled_models.get(signature)
        if compiled_model is None:
            if (
                self._resident_model_limit is not None
                and len(self._compiled_models) >= self._resident_model_limit
            ):
                # Release the device allocation before loading the next cached profile.
                self._compiled_models.popitem(last=False)
            compiled_model = self._compile(inputs)
            self._compiled_models[signature] = compiled_model
        else:
            self._compiled_models.move_to_end(signature)
        return compiled_model

    def _run_batched(self, inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
        outputs = []
        for batch_inputs in _iter_batch_inputs(inputs):
            compiled_model = self._get_compiled_model(batch_inputs)
            outputs.append(torch.from_numpy(compiled_model(batch_inputs)[0].copy()))
        return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)

    def _warmup(self, compiled_model, inputs: tuple[torch.Tensor, ...]) -> None:
        compiled_model(inputs)

    def _prepare_profiles(
        self,
        profiles: list[tuple[torch.Tensor, ...]],
        *,
        log_fn: Callable[[str], None] | None,
    ) -> None:
        if log_fn is None:

            def log_fn(message: str) -> None:
                print(message, flush=True)

        for index, inputs in enumerate(profiles, start=1):
            signature = tuple((tuple(tensor.shape), tensor.dtype) for tensor in inputs)
            if signature in self._warmed_profiles:
                continue
            prefix = f"[openvino] {self._ir_filename} [{index}/{len(profiles)}]"
            log_fn(f"{prefix} prepare: {[tuple(tensor.shape) for tensor in inputs]}")
            started = time.perf_counter()
            compiled_model = self._get_compiled_model(inputs)
            compile_seconds = time.perf_counter() - started
            started = time.perf_counter()
            self._warmup(compiled_model, inputs)
            self._warmed_profiles.add(signature)
            log_fn(
                f"{prefix} ready: cache={compiled_model.get_property('LOADED_FROM_CACHE')}, "
                f"load/compile={compile_seconds:.2f}s, warmup={time.perf_counter() - started:.2f}s"
            )
            del compiled_model


class _OpenVINODACVAEBackend(_OpenVINOBackendBase):
    def __init__(self, *, device: str, checkpoint_path: str | Path) -> None:
        self._context_frames = 0
        super().__init__(device=device, checkpoint_path=checkpoint_path)

    @property
    def _cache_mode(self) -> str:
        # Retain transformed convolution weights to keep codec compilation inexpensive.
        return "OPTIMIZE_SPEED"

    def _prepare_model(self, inputs: tuple[torch.Tensor, ...]):
        model = super()._prepare_model(inputs)
        self._context_frames = _codec_context_frames(model, self._frame_size)
        return _mask_codec_boundaries(self.ov, model, inputs[0].shape[-1] // self._frame_size)

    def _masked_inputs(
        self,
        compiled_model,
        value: torch.Tensor,
        valid_frames: int,
    ) -> list[torch.Tensor]:
        inputs = [value]
        width = value.shape[-1] // self._frame_size
        for port in compiled_model.inputs[1:]:
            scale = int(port.any_name.removeprefix("codec_valid_"))
            mask = value.new_zeros((1, 1, width * scale))
            mask[..., : valid_frames * scale] = 1
            inputs.append(mask)
        return inputs

    def _warmup(self, compiled_model, inputs: tuple[torch.Tensor, ...]) -> None:
        value = inputs[0]
        compiled_model(self._masked_inputs(compiled_model, value, value.shape[-1] // self._frame_size))

    def _run_bucketed(self, value: torch.Tensor, buckets: tuple[int, ...]) -> torch.Tensor:
        frames = value.shape[-1] // self._frame_size
        if frames == 0:
            raise ValueError("DACVAE inputs must contain at least one latent frame.")
        pieces = []
        cursor = 0
        while cursor < frames:
            remaining = frames - cursor + (self._context_frames if cursor else 0)
            width = next((capacity for capacity in buckets if capacity >= remaining), buckets[-1])
            # Align the final window with the real boundary and crop overlapping context.
            origin = max(0, min(cursor - self._context_frames, frames - width))
            window = value[..., origin * self._frame_size : (origin + width) * self._frame_size]
            valid_frames = window.shape[-1] // self._frame_size
            if valid_frames < width:
                padding = window.new_zeros(
                    (*window.shape[:-1], (width - valid_frames) * self._frame_size)
                )
                window = torch.cat((window, padding), dim=-1)
            batch_pieces = []
            for batch_inputs in _iter_batch_inputs((window,)):
                compiled_model = self._get_compiled_model(batch_inputs)
                if frames > width and width <= 2 * self._context_frames:
                    raise RuntimeError("DACVAE bucket is too short for its convolution context.")
                inputs = self._masked_inputs(compiled_model, batch_inputs[0], valid_frames)
                output = compiled_model(inputs)[0]
                output_frame_size = output.shape[-1] // width
                end = frames if origin + width >= frames else origin + width - self._context_frames
                piece = output[
                    ..., (cursor - origin) * output_frame_size : (end - origin) * output_frame_size
                ]
                batch_pieces.append(torch.from_numpy(piece.copy()))
            pieces.append(
                batch_pieces[0] if len(batch_pieces) == 1 else torch.cat(batch_pieces, dim=0)
            )
            cursor = end
        return torch.cat(pieces, dim=-1)


def _unpack_context_kv(
    model: TextToLatentRFDiT, states: tuple[torch.Tensor, ...]
) -> list[tuple[torch.Tensor, ...]]:
    cfg = model.cfg
    states = tuple(
        state.unflatten(-1, (2 * cfg.num_layers, cfg.num_heads, model.head_dim)) for state in states
    )
    return [
        tuple(state[:, :, 2 * layer + key] for state in states for key in (0, 1))
        for layer in range(cfg.num_layers)
    ]


class _ContextKVExportWrapper(torch.nn.Module):
    def __init__(self, model: TextToLatentRFDiT) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, text_state: torch.Tensor, speaker_state: torch.Tensor, caption_state: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        cfg = self.model.cfg
        cache = self.model.build_context_kv_cache(
            text_state,
            speaker_state if cfg.use_speaker_condition_resolved else None,
            caption_state if cfg.use_caption_condition else None,
        )
        count = 1 + int(cfg.use_speaker_condition_resolved) + int(cfg.use_caption_condition)
        return tuple(
            torch.stack(
                [layer[2 * condition + key].flatten(2) for layer in cache for key in (0, 1)],
                dim=2,
            ).flatten(2)
            for condition in range(count)
        )


class _OpenVINOContextKVCache(list[tuple[torch.Tensor, ...]]):
    """Layer views support speaker scaling; packed tensors are bound directly to the DiT."""

    def __init__(
        self,
        model: TextToLatentRFDiT,
        packed: tuple[torch.Tensor, ...],
        lengths: list[int],
        profile: tuple[int, int, int],
        batch_rows: tuple[int, ...],
    ) -> None:
        super().__init__(
            _unpack_context_kv(
                model,
                tuple(state[:, :length] for state, length in zip(packed, lengths, strict=True)),
            )
        )
        self.profile = profile
        active = (True, model.cfg.use_speaker_condition_resolved, model.cfg.use_caption_condition)
        states = iter(packed)
        self.packed_states = tuple(
            next(states) if enabled else packed[0].new_zeros((packed[0].shape[0], 1, 1))
            for enabled in active
        )
        self._batch_rows = batch_rows
        self._versions = tuple(state._version for state in self.packed_states)

    def batch_rows(self) -> tuple[int, ...]:
        versions = tuple(state._version for state in self.packed_states)
        if versions != self._versions:
            rows = []
            for index in range(self.packed_states[0].shape[0]):
                rows.append(
                    next(
                        (
                            previous
                            for previous in range(index)
                            if all(
                                torch.equal(state[previous], state[index])
                                for state in self.packed_states
                            )
                        ),
                        index,
                    )
                )
            self._batch_rows = tuple(rows)
            self._versions = versions
        return self._batch_rows


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
        states = (text_state,)
        if self.use_speaker:
            states += (speaker_state,)
        if self.use_caption:
            states += (caption_state,)
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
            context_kv_cache=_unpack_context_kv(self.model, states),
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
        states = (text_state,)
        if self.use_speaker:
            states += (speaker_state,)
        if self.use_caption:
            states += (caption_state,)
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
            context_kv_cache=_unpack_context_kv(self.model, states),
        )


class _DACVAEEncoderExportWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.encoder = model.encoder
        self.in_proj = model.quantizer.in_proj

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.in_proj(self.encoder(waveform))


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


class _SilentCipherLayerExportWrapper(torch.nn.Module):
    def __init__(self, layer: torch.nn.Module) -> None:
        super().__init__()
        self.conv = layer.conv
        self.gate = layer.gate
        self.register_buffer("weight", layer.bn.weight.detach().reshape(1, -1, 1, 1))
        self.register_buffer("bias", layer.bn.bias.detach().reshape(1, -1, 1, 1))
        self.eps = layer.bn.eps

    def forward(
        self, value: torch.Tensor, mask: torch.Tensor, valid_fraction: torch.Tensor
    ) -> torch.Tensor:
        hidden = self.conv(value) * torch.sigmoid(self.gate(value))
        scale = (hidden * mask).abs().amax(dim=(0, 2, 3), keepdim=True).clamp_min(64.0) / 64.0
        scaled = hidden / scale
        mean = (scaled * mask).mean(dim=(0, 2, 3), keepdim=True) / valid_fraction
        centered = (scaled - mean) * mask
        variance = (
            centered.square().mean(dim=(0, 2, 3), keepdim=True) / valid_fraction * scale * scale
        )
        normalized = centered * scale * torch.rsqrt(variance + self.eps) * self.weight + self.bias
        return normalized * mask


class _SilentCipherExportWrapper(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module, decoder: torch.nn.Module) -> None:
        super().__init__()
        self.encoder = torch.nn.ModuleList(
            _SilentCipherLayerExportWrapper(layer) for layer in encoder.main
        )
        self.decoder = torch.nn.ModuleList(
            _SilentCipherLayerExportWrapper(layer) for layer in decoder.main
        )
        self.projection = encoder.linear
        self.frequency_padding = encoder.n_fft // 2 + 1 - encoder.message_band_size
        self.channels = encoder.main[-1].conv.out_channels

    def forward(
        self,
        carrier: torch.Tensor,
        messages: torch.Tensor,
        mask: torch.Tensor,
        valid_fraction: torch.Tensor,
    ) -> torch.Tensor:
        encoded = carrier * mask
        for layer in self.encoder:
            encoded = layer(encoded, mask, valid_fraction)
        projected = self.projection(messages.transpose(2, 3)).transpose(2, 3)
        projected = torch.nn.functional.pad(projected, (0, 0, 0, self.frequency_padding)) * mask
        hidden = (
            torch.cat(
                (
                    encoded,
                    carrier.repeat(1, self.channels, 1, 1),
                    projected.repeat(1, self.channels, 1, 1),
                ),
                dim=1,
            )
            * mask
        )
        for layer in self.decoder:
            hidden = layer(hidden, mask, valid_fraction)
        return hidden


class _OpenVINOContextKVBackend(_OpenVINOBackendBase):
    def __init__(
        self,
        model: TextToLatentRFDiT,
        *,
        device: str,
        checkpoint_path: str | Path,
        core: Any,
    ) -> None:
        self.model = model
        self.wrapper = _ContextKVExportWrapper(model).eval()
        super().__init__(device=device, checkpoint_path=checkpoint_path, core=core)
        if self.device == "NPU":
            self._resident_model_limit = 1

    @property
    def _ir_filename(self) -> str:
        return "context_kv.xml"

    def _supports_dynamic_ir(self, inputs: tuple[torch.Tensor, ...]) -> bool:
        return all(tensor.shape[1] <= _MAX_DYNAMIC_SEQUENCE_LENGTH for tensor in inputs)

    def _export_dynamic_ir(self, inputs: tuple[torch.Tensor, ...]):
        cfg = self.model.cfg
        batch = torch.export.Dim("batch", min=1)
        enabled = (True, cfg.use_speaker_condition_resolved, cfg.use_caption_condition)
        examples = tuple(
            tensor.new_zeros((2, max(2, tensor.shape[1]) if active else 1, tensor.shape[2]))
            for tensor, active in zip(inputs, enabled, strict=True)
        )
        shapes = tuple(
            {0: batch, 1: torch.export.Dim(name, min=1, max=_MAX_DYNAMIC_SEQUENCE_LENGTH)}
            if active
            else {0: batch}
            for name, active in zip(("text", "speaker", "caption"), enabled, strict=True)
        )
        return self.ov.convert_model(
            torch.export.export(self.wrapper, examples, dynamic_shapes=shapes)
        )

    def _convert_static(self, inputs: tuple[torch.Tensor, ...]):
        return self.ov.convert_model(torch.export.export(self.wrapper, inputs))

    @torch.inference_mode()
    def prepare_eager_profiles(
        self,
        *,
        max_condition_lengths: tuple[int, int, int],
        log_fn: Callable[[str], None] | None,
    ) -> None:
        cfg = self.model.cfg
        parameter = next(self.model.parameters())
        profiles = [
            (
                parameter.new_zeros((1, text, cfg.text_dim)),
                parameter.new_zeros((1, speaker, cfg.speaker_dim))
                if cfg.use_speaker_condition_resolved
                else parameter.new_zeros((1, 1, 1)),
                parameter.new_zeros((1, caption, cfg.caption_dim_resolved))
                if cfg.use_caption_condition
                else parameter.new_zeros((1, 1, 1)),
            )
            for text, speaker, caption in _condition_profiles_up_to(max_condition_lengths)
        ]
        self._prepare_profiles(profiles, log_fn=log_fn)

    @torch.inference_mode()
    def __call__(
        self,
        *,
        text_state: torch.Tensor,
        speaker_state: torch.Tensor | None,
        caption_state: torch.Tensor | None,
    ) -> list[tuple[torch.Tensor, ...]]:
        cfg = self.model.cfg
        states = [text_state]
        if cfg.use_speaker_condition_resolved:
            if speaker_state is None:
                raise ValueError("Speaker state is required by this DiT model.")
            states.append(speaker_state)
        if cfg.use_caption_condition:
            if caption_state is None:
                raise ValueError("Caption state is required by this DiT model.")
            states.append(caption_state)
        lengths = [state.shape[1] for state in states]
        profile = _select_condition_profile(
            text_state.shape[1],
            speaker_state.shape[1] if cfg.use_speaker_condition_resolved else 0,
            caption_state.shape[1] if cfg.use_caption_condition else 0,
        )
        batch = text_state.shape[0]
        inputs = tuple(
            _pad_condition_sequence(
                state, torch.ones(state.shape[:2], dtype=torch.bool, device=state.device), length
            )[0]
            if active
            else text_state.new_zeros((batch, 1, 1))
            for state, length, active in zip(
                (text_state, speaker_state, caption_state),
                profile,
                (True, cfg.use_speaker_condition_resolved, cfg.use_caption_condition),
                strict=True,
            )
        )
        rows = []
        unique = []
        batch_rows = []
        for index, batch_inputs in enumerate(_iter_batch_inputs(inputs)):
            entry = next(
                (
                    (output, row)
                    for previous, output, row in unique
                    if all(
                        torch.equal(left, right)
                        for left, right in zip(previous, batch_inputs, strict=True)
                    )
                ),
                None,
            )
            if entry is None:
                compiled = self._get_compiled_model(batch_inputs)
                output = compiled(batch_inputs)
                # Normal tensors track mutations made by force-speaker K/V scaling.
                with torch.inference_mode(False):
                    projected = tuple(
                        torch.from_numpy(output[index].copy()) for index in range(len(states))
                    )
                unique.append((batch_inputs, projected, index))
                batch_rows.append(index)
            else:
                projected, row = entry
                batch_rows.append(row)
            rows.append(projected)
        with torch.inference_mode(False):
            packed = tuple(
                torch.cat([row[index] for row in rows], dim=0) for index in range(len(states))
            )
            return _OpenVINOContextKVCache(self.model, packed, lengths, profile, tuple(batch_rows))


class _CachedDiTCompiledModel:
    def __init__(self, compiled_model: Any, *, state_indexes: tuple[int, ...]) -> None:
        self.compiled_model = compiled_model
        self.state_indexes = state_indexes
        self._requests: OrderedDict[tuple, tuple[Any, tuple[torch.Tensor, ...]]] = OrderedDict()

    def get_property(self, name: str):
        return self.compiled_model.get_property(name)

    @staticmethod
    def _state_key(states: tuple[torch.Tensor, ...]) -> tuple:
        return tuple(
            (
                tensor.data_ptr(),
                tuple(tensor.shape),
                None if tensor.is_inference() else tensor._version,
            )
            for tensor in states
        )

    def __call__(self, inputs: tuple[torch.Tensor, ...]):
        states = tuple(inputs[index] for index in self.state_indexes)
        key = self._state_key(states)
        entry = self._requests.pop(key, None)
        if entry is None:
            previous_key = next(
                (
                    previous_key
                    for previous_key, (_, previous) in self._requests.items()
                    if self._state_key(previous) == previous_key
                    and all(
                        torch.equal(left, right)
                        for left, right in zip(previous, states, strict=True)
                    )
                ),
                None,
            )
            if previous_key is not None:
                entry = self._requests.pop(previous_key)
        if entry is None:
            if len(self._requests) >= _MAX_CONTEXT_KV_CACHES:
                self._requests.popitem(last=False)
            request = self.compiled_model.create_infer_request()
            # Keep K/V in the request's native device buffers across diffusion steps.
            for index in self.state_indexes:
                request.get_input_tensor(index).data[...] = inputs[index].numpy()
        else:
            request = entry[0]
        self._requests[key] = (request, states)
        for index, tensor in enumerate(inputs):
            if index not in self.state_indexes:
                request.get_input_tensor(index).data[...] = tensor.numpy()
        return request.infer(share_outputs=True)


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
        self._context_kv_backend = _OpenVINOContextKVBackend(
            self.model, device=device, checkpoint_path=checkpoint_path, core=self.core
        )
        self._packed_context_caches: OrderedDict[int, tuple] = OrderedDict()

    @property
    def _ir_filename(self) -> str:
        return "meanflow_dit_kv.xml" if self.is_meanflow else "rf_dit_kv.xml"

    def _compile(self, inputs: tuple[torch.Tensor, ...]):
        offset = 4 if self.is_meanflow else 3
        active = (
            True,
            self.model.cfg.use_speaker_condition_resolved,
            self.model.cfg.use_caption_condition,
        )
        return _CachedDiTCompiledModel(
            super()._compile(inputs),
            state_indexes=tuple(
                offset + 2 * index for index, enabled in enumerate(active) if enabled
            ),
        )

    def _warmup(
        self, compiled_model: _CachedDiTCompiledModel, inputs: tuple[torch.Tensor, ...]
    ) -> None:
        compiled_model(inputs)
        compiled_model._requests.clear()

    def build_context_kv_cache(
        self,
        *,
        text_state: torch.Tensor,
        speaker_state: torch.Tensor | None,
        caption_state: torch.Tensor | None = None,
    ) -> list[tuple[torch.Tensor, ...]]:
        return self._context_kv_backend(
            text_state=text_state, speaker_state=speaker_state, caption_state=caption_state
        )

    def _run_batched(
        self,
        inputs: tuple[torch.Tensor, ...],
        *,
        context_rows: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        outputs = []
        state_indexes = (4, 6, 8) if self.is_meanflow else (3, 5, 7)
        for index, batch_inputs in enumerate(_iter_batch_inputs(inputs)):
            if context_rows is not None:
                # Identical candidates share the same K/V binding, including CFG branches.
                row = context_rows[index]
                batch_inputs = tuple(
                    inputs[position][row : row + 1].contiguous()
                    if position in state_indexes
                    else tensor
                    for position, tensor in enumerate(batch_inputs)
                )
            compiled_model = self._get_compiled_model(batch_inputs)
            outputs.append(torch.from_numpy(compiled_model(batch_inputs)[0].copy()))
        return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)

    def _pack_context_cache(
        self,
        cache: list[tuple[torch.Tensor, ...]],
        profile: tuple[int, int, int],
    ) -> tuple[torch.Tensor, ...]:
        if isinstance(cache, _OpenVINOContextKVCache) and cache.profile == profile:
            return cache.packed_states
        # Force-speaker scaling mutates K/V in place; invalidate packed inputs on mutation.
        try:
            versions = tuple(tensor._version for layer in cache for tensor in layer)
        except RuntimeError:
            # External inference tensors lack version counters and cannot be memoized safely.
            versions = None
        entry = self._packed_context_caches.get(id(cache)) if versions is not None else None
        if entry is not None and entry[1:3] == (profile, versions):
            self._packed_context_caches.move_to_end(id(cache))
            return entry[3]
        cfg = self.model.cfg
        if len(cache) != cfg.num_layers:
            raise ValueError("Context K/V cache must contain one entry per DiT layer.")
        packed = []
        offset = 0
        for length, active in zip(
            profile,
            (True, cfg.use_speaker_condition_resolved, cfg.use_caption_condition),
            strict=True,
        ):
            if active:
                state = torch.stack(
                    [layer[offset + key].flatten(2) for layer in cache for key in (0, 1)], dim=2
                ).flatten(2)
                state = _pad_condition_sequence(
                    state,
                    torch.ones(state.shape[:2], dtype=torch.bool, device=state.device),
                    length,
                )[0]
                offset += 2
            else:
                state = cache[0][0].new_zeros((cache[0][0].shape[0], 1, 1))
            packed.append(state)
        packed = tuple(packed)
        if versions is not None:
            self._packed_context_caches[id(cache)] = (cache, profile, versions, packed)
            self._packed_context_caches.move_to_end(id(cache))
            if len(self._packed_context_caches) > _MAX_CONTEXT_KV_CACHES:
                self._packed_context_caches.popitem(last=False)
        return packed

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
        profile = _select_condition_profile(
            text_state.shape[1],
            speaker_state.shape[1] if self.model.cfg.use_speaker_condition_resolved else 0,
            caption_state.shape[1] if self.model.cfg.use_caption_condition else 0,
        )
        text_state, text_mask = _pad_condition_sequence(text_state, text_mask, profile[0])
        if self.model.cfg.use_speaker_condition_resolved:
            speaker_state, speaker_mask = _pad_condition_sequence(
                speaker_state, speaker_mask, profile[1]
            )
        if self.model.cfg.use_caption_condition:
            caption_state, caption_mask = _pad_condition_sequence(
                caption_state, caption_mask, profile[2]
            )
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

    @torch.inference_mode()
    def prepare_eager_profiles(
        self,
        *,
        max_sequence_length: int = _LATENT_SEQUENCE_BUCKETS[-1],
        max_condition_lengths: tuple[int, int, int] = _CONDITION_SEQUENCE_PROFILES[-1],
        log_fn: Callable[[str], None] | None = None,
    ) -> None:
        cfg = self.model.cfg
        parameter = next(self.model.parameters())
        self._context_kv_backend.prepare_eager_profiles(
            max_condition_lengths=max_condition_lengths, log_fn=log_fn
        )
        packed_dim = 2 * cfg.num_layers * cfg.model_dim

        def condition(length: int, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
            state = parameter.new_zeros((1, length, dim))
            mask = torch.ones((1, length), dtype=torch.bool, device=parameter.device)
            return state, mask

        profiles = []
        latent_buckets = _latent_buckets_up_to(max_sequence_length)
        for text_length, speaker_length, caption_length in _condition_profiles_up_to(
            max_condition_lengths
        ):
            text_state, text_mask = condition(text_length, packed_dim)
            speaker_state, speaker_mask = (
                condition(speaker_length, packed_dim)
                if cfg.use_speaker_condition_resolved
                else (None, None)
            )
            caption_state, caption_mask = (
                condition(caption_length, packed_dim) if cfg.use_caption_condition else (None, None)
            )
            for length in latent_buckets:
                profiles.append(
                    self._prepare_inputs(
                        x_t=parameter.new_zeros((1, length, cfg.patched_latent_dim)),
                        t=parameter.new_full((1,), 0.5),
                        delta_t=parameter.new_full((1,), 0.25) if self.is_meanflow else None,
                        text_state=text_state,
                        text_mask=text_mask,
                        speaker_state=speaker_state,
                        speaker_mask=speaker_mask,
                        caption_state=caption_state,
                        caption_mask=caption_mask,
                    )
                )
        self._prepare_profiles(profiles, log_fn=log_fn)

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
        cfg = self.model.cfg
        for state, mask, enabled in zip(
            (text_state, speaker_state, caption_state),
            (text_mask, speaker_mask, caption_mask),
            (True, cfg.use_speaker_condition_resolved, cfg.use_caption_condition),
            strict=True,
        ):
            if enabled:
                if state is None or mask is None:
                    raise ValueError(
                        "Enabled condition states and masks are required by this DiT model."
                    )
                _pad_condition_sequence(state, mask, state.shape[1])
        profile = _select_condition_profile(
            text_state.shape[1],
            speaker_state.shape[1]
            if cfg.use_speaker_condition_resolved and speaker_state is not None
            else 0,
            caption_state.shape[1]
            if cfg.use_caption_condition and caption_state is not None
            else 0,
        )
        if context_kv_cache is None:
            context_kv_cache = self.build_context_kv_cache(
                text_state=text_state, speaker_state=speaker_state, caption_state=caption_state
            )
        states = self._pack_context_cache(context_kv_cache, profile)
        masks = tuple(
            torch.nn.functional.pad(mask, (0, state.shape[1] - mask.shape[1]))
            if mask is not None
            else None
            for state, mask in zip(states, (text_mask, speaker_mask, caption_mask), strict=True)
        )
        sequence_length = x_t.shape[1]
        inputs = self._prepare_inputs(
            x_t=x_t,
            t=t,
            delta_t=delta_t,
            text_state=states[0],
            text_mask=masks[0],
            speaker_state=states[1],
            speaker_mask=masks[1],
            caption_state=states[2],
            caption_mask=masks[2],
        )
        context_rows = (
            context_kv_cache.batch_rows()
            if isinstance(context_kv_cache, _OpenVINOContextKVCache)
            else None
        )
        output = self._run_batched(inputs, context_rows=context_rows)
        return output[:, :sequence_length].to(
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

    @torch.inference_mode()
    def prepare_eager_profiles(
        self,
        *,
        max_sequence_lengths: tuple[int, ...] = (_CONDITION_SEQUENCE_PROFILES[-1][0],),
        log_fn: Callable[[str], None] | None = None,
    ) -> None:
        if not max_sequence_lengths or min(max_sequence_lengths) <= 0:
            raise ValueError("Maximum condition sequence lengths must be positive.")
        lengths = _condition_profiles_up_to((max(max_sequence_lengths), 0, 0))
        device = next(self.backbone.parameters()).device
        profiles = [
            (
                torch.zeros((1, profile[0]), dtype=torch.long, device=device),
                torch.ones((1, profile[0]), dtype=torch.bool, device=device),
            )
            for profile in lengths
        ]
        self._prepare_profiles(profiles, log_fn=log_fn)

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
        active_rows = mask.any(dim=1)
        if not bool(active_rows.any()):
            return torch.zeros(
                (*input_ids.shape, self.hidden_size),
                dtype=self.dtype,
                device=input_ids.device,
            )
        sequence_length = input_ids.shape[1]
        bucket_length = _select_condition_profile(sequence_length, 0, 0)[0]
        inputs = _pad_condition_sequence(input_ids[active_rows], mask[active_rows], bucket_length)
        output = self._run_batched(inputs)
        output = output[:, :sequence_length].to(device=input_ids.device, dtype=self.dtype)
        if bool(active_rows.all()):
            return output
        state = output.new_zeros((*input_ids.shape, self.hidden_size))
        state[active_rows] = output
        return state


class OpenVINODACVAEEncoderBackend(_OpenVINODACVAEBackend):
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        device: str,
        checkpoint_path: str | Path,
        deterministic_encode: bool,
    ) -> None:
        self.model = model.eval()
        self.wrapper = _DACVAEEncoderExportWrapper(self.model).eval()
        self.dtype = next(self.model.parameters()).dtype
        self.hop_length = int(self.model.hop_length)
        self._frame_size = self.hop_length
        self.deterministic_encode = deterministic_encode
        super().__init__(device=device, checkpoint_path=checkpoint_path)

    @property
    def _ir_filename(self) -> str:
        return "dacvae_encoder_minimax.xml"

    def _convert_exported(self, exported: torch.export.ExportedProgram):
        model = self.ov.convert_model(exported)
        _replace_squared_sine_with_polynomial(self.ov, model)
        return model

    def _supports_dynamic_ir(self, inputs: tuple[torch.Tensor, ...]) -> bool:
        return inputs[0].shape[2] // self.hop_length <= _MAX_DYNAMIC_SEQUENCE_LENGTH

    def _export_dynamic_ir(self, inputs: tuple[torch.Tensor, ...]):
        waveform = inputs[0]
        export_input = torch.zeros(
            (2, 1, self.hop_length * max(2, waveform.shape[2] // self.hop_length)),
            dtype=waveform.dtype,
            device=waveform.device,
        )
        batch = torch.export.Dim("batch", min=1)
        latent = torch.export.Dim("latent", min=1, max=_MAX_DYNAMIC_SEQUENCE_LENGTH)
        with torch.inference_mode():
            exported = torch.export.export(
                self.wrapper,
                (export_input,),
                dynamic_shapes=({0: batch, 2: self.hop_length * latent},),
            )
        return self._convert_exported(exported)

    def _convert_static(self, inputs: tuple[torch.Tensor, ...]):
        with torch.inference_mode():
            exported = torch.export.export(self.wrapper, inputs)
        return self._convert_exported(exported)

    @torch.inference_mode()
    def prepare_eager_profiles(self, *, log_fn: Callable[[str], None] | None = None) -> None:
        parameter = next(self.model.parameters())
        profiles = [
            (parameter.new_zeros((1, 1, frames * self.hop_length)),)
            for frames in _CODEC_ENCODER_BUCKETS
        ]
        self._prepare_profiles(profiles, log_fn=log_fn)

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim != 3 or waveform.shape[1] != 1:
            raise ValueError(f"Expected mono waveform shape (B,1,T), got {tuple(waveform.shape)}")
        waveform = waveform.contiguous().to(device="cpu", dtype=self.dtype)
        waveform = self.model._pad(waveform)
        output = self._run_bucketed(waveform, _CODEC_ENCODER_BUCKETS)
        mean, scale = output.chunk(2, dim=1)
        if self.deterministic_encode:
            encoded = mean
        else:
            encoded, _ = self.model.quantizer._vae_sample(mean, scale)
        return encoded.transpose(1, 2).contiguous()


class OpenVINODACVAEDecoderBackend(_OpenVINODACVAEBackend):
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
        self._frame_size = 1
        super().__init__(device=device, checkpoint_path=checkpoint_path)

    @property
    def _ir_filename(self) -> str:
        return "dacvae_decoder_minimax.xml"

    def _convert_exported(self, exported: torch.export.ExportedProgram):
        model = self.ov.convert_model(exported)
        _replace_squared_sine_with_polynomial(self.ov, model)
        return model

    def _compile(self, inputs: tuple[torch.Tensor, ...]):
        model = self._prepare_model(inputs)
        strategy = _pack_dacvae_polynomials(self.ov, model) if self.device == "NPU" else {}
        if not strategy:
            return self.core.compile_model(model, self.device)

        strategy_json = json.dumps(strategy, sort_keys=True)
        strategy_hash = hashlib.sha256(strategy_json.encode()).hexdigest()[:16]
        with tempfile.TemporaryDirectory(prefix="irodori-dacvae-") as directory:
            root = Path(directory)
            model_path = root / "model.xml"
            self.ov.save_model(model, model_path, compress_to_fp16=False)
            (root / "strategy_in.json").write_text(strategy_json + "\n", encoding="utf-8")
            # OpenVINO hashes the model and checks compiler/driver compatibility;
            # its cache key does not include the contents of strategy_in.json.
            cache_dir = (self._ir_dir or root) / "compiled" / f"dacvae-fusion-{strategy_hash}"
            config = {
                "CACHE_DIR": str(cache_dir.resolve()),
                "CACHE_MODE": "OPTIMIZE_SPEED",
                "NPU_COMPILATION_MODE_PARAMS": "read-strategy-from-json=true",
            }
            (root / "config.json").write_text(json.dumps(config) + "\n", encoding="utf-8")
            # The compiler reads a fixed relative filename. Isolate its cwd from
            # the application's other threads, then load the populated OV cache.
            subprocess.run(
                [sys.executable, str(Path(__file__).with_name("_openvino_compile.py").resolve())],
                cwd=root,
                check=True,
            )
            return self.core.compile_model(self.core.read_model(model_path), self.device, config)

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

    @torch.inference_mode()
    def prepare_eager_profiles(self, *, log_fn: Callable[[str], None] | None = None) -> None:
        parameter = next(self.model.parameters())
        latent_dim = int(self.model.quantizer.out_proj.in_channels)
        profiles = [
            (parameter.new_zeros((1, latent_dim, frames)),) for frames in _CODEC_DECODER_BUCKETS
        ]
        self._prepare_profiles(profiles, log_fn=log_fn)

    def __call__(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 3:
            raise ValueError(f"Expected latent ndim=3, got shape={tuple(latent.shape)}")
        z = latent.transpose(1, 2).contiguous().to(device="cpu", dtype=self.dtype)
        return self._run_bucketed(z, _CODEC_DECODER_BUCKETS)


class OpenVINOSilentCipherBackend(_OpenVINOBackendBase):
    def __init__(
        self,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        *,
        device: str,
        checkpoint_path: str | Path,
    ) -> None:
        self.wrapper = _SilentCipherExportWrapper(encoder, decoder).eval()
        self.dtype = next(self.wrapper.parameters()).dtype
        self.frequency_bins = encoder.n_fft // 2 + 1
        self.message_dim = encoder.linear.in_features
        self.max_sequence_length = _WATERMARK_SEQUENCE_BUCKETS[-1]
        super().__init__(device=device, checkpoint_path=checkpoint_path)

    @property
    def _ir_filename(self) -> str:
        return "silentcipher.xml"

    @property
    def _cache_mode(self) -> str:
        return "OPTIMIZE_SPEED"

    def _supports_dynamic_ir(self, inputs: tuple[torch.Tensor, ...]) -> bool:
        return inputs[0].shape[-1] <= self.max_sequence_length

    def _export_dynamic_ir(self, inputs: tuple[torch.Tensor, ...]):
        sequence = torch.export.Dim("frames", min=1, max=self.max_sequence_length)
        with torch.inference_mode():
            exported = torch.export.export(
                self.wrapper,
                inputs,
                dynamic_shapes=({3: sequence}, {3: sequence}, {3: sequence}, {}),
            )
        return self.ov.convert_model(exported)

    def _convert_static(self, inputs: tuple[torch.Tensor, ...]):
        with torch.inference_mode():
            exported = torch.export.export(self.wrapper, inputs)
        return self.ov.convert_model(exported)

    @torch.inference_mode()
    def prepare_eager_profiles(self, *, log_fn: Callable[[str], None] | None = None) -> None:
        parameter = next(self.wrapper.parameters())
        profiles = [
            (
                parameter.new_zeros((1, 1, self.frequency_bins, frames)),
                parameter.new_zeros((1, 1, self.message_dim, frames)),
                parameter.new_ones((1, 1, 1, frames)),
                parameter.new_ones((1, 1, 1, 1)),
            )
            for frames in _WATERMARK_SEQUENCE_BUCKETS
        ]
        self._prepare_profiles(profiles, log_fn=log_fn)

    @torch.inference_mode()
    def __call__(self, carrier: torch.Tensor, messages: torch.Tensor) -> torch.Tensor:
        if carrier.ndim != 4 or carrier.shape[:3] != (1, 1, self.frequency_bins):
            raise ValueError(
                f"Expected carrier shape (1,1,{self.frequency_bins},T), got {carrier.shape}."
            )
        frames = carrier.shape[-1]
        if messages.shape != (1, 1, self.message_dim, frames):
            raise ValueError(
                f"Expected message shape (1,1,{self.message_dim},{frames}), got {messages.shape}."
            )
        if not 0 < frames <= self.max_sequence_length:
            raise ValueError(
                f"Watermark sequence length must be between 1 and {self.max_sequence_length}."
            )
        capacity = next(length for length in _WATERMARK_SEQUENCE_BUCKETS if frames <= length)
        carrier = carrier.contiguous().to(device="cpu", dtype=self.dtype)
        messages = messages.contiguous().to(device="cpu", dtype=self.dtype)
        mask = carrier.new_zeros((1, 1, 1, capacity))
        mask[..., :frames] = 1
        inputs = (
            torch.nn.functional.pad(carrier, (0, capacity - frames)),
            torch.nn.functional.pad(messages, (0, capacity - frames)),
            mask,
            carrier.new_full((1, 1, 1, 1), frames / capacity),
        )
        return self._run_batched(inputs)[..., :frames].contiguous()


def create_dit_backend(
    model: TextToLatentRFDiT,
    *,
    device: str,
    checkpoint_path: str | Path,
) -> OpenVINODiTBackend:
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


def create_dacvae_encoder_backend(
    model: torch.nn.Module,
    *,
    device: str,
    checkpoint_path: str | Path,
    deterministic_encode: bool,
) -> Callable[[torch.Tensor], torch.Tensor]:
    return OpenVINODACVAEEncoderBackend(
        model,
        device=device,
        checkpoint_path=checkpoint_path,
        deterministic_encode=deterministic_encode,
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


def create_silentcipher_backend(
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    *,
    device: str,
    checkpoint_path: str | Path,
) -> OpenVINOSilentCipherBackend:
    return OpenVINOSilentCipherBackend(
        encoder,
        decoder,
        device=device,
        checkpoint_path=checkpoint_path,
    )
