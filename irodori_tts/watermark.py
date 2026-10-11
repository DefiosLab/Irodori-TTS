from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .openvino_backend import OpenVINOSilentCipherBackend

logger = logging.getLogger(__name__)

IRODORI_WATERMARK_PAYLOAD = (73, 82, 68, 84, 83)  # "IRDTS"


def _as_single_channel_vector(audio: torch.Tensor) -> torch.Tensor | None:
    squeezed = audio.detach().float().squeeze()
    if squeezed.ndim == 0 or squeezed.numel() == 0:
        return None
    if squeezed.ndim == 1:
        return squeezed
    return squeezed.reshape(-1)


def _match_original_rank(audio: torch.Tensor, *, reference: torch.Tensor) -> torch.Tensor:
    if reference.ndim == 2:
        return audio.reshape(1, -1)
    return audio.reshape(-1)


class SilentCipherWatermarker:
    def __init__(self, *, device: str, model_type: str = "44.1k") -> None:
        npu = str(device).strip().lower() == "npu"
        self.model = self._load_backend(device="cpu" if npu else device, model_type=model_type)
        self._npu_backend: OpenVINOSilentCipherBackend | None = None
        if npu and self.model is not None:
            from .openvino_backend import create_silentcipher_backend

            self._npu_backend = create_silentcipher_backend(
                self.model.enc_c,
                self.model.dec_c,
                device="NPU",
                checkpoint_path=Path(self.model.config.load_ckpt) / "enc_c.ckpt",
            )

    @staticmethod
    def _load_backend(*, device: str, model_type: str):
        try:
            import silentcipher
        except ImportError:
            logger.warning(
                "SilentCipher package is unavailable; generated audio will not be watermarked."
            )
            return None

        try:
            return silentcipher.get_model(model_type=model_type, device=device)
        except Exception as exc:
            logger.warning(
                "SilentCipher model could not be loaded (%s); generated audio will not be "
                "watermarked.",
                exc,
            )
            return None

    @property
    def ready(self) -> bool:
        return self.model is not None

    def prepare_eager_profiles(self, *, log_fn: Callable[[str], None] | None = None) -> None:
        if self._npu_backend is not None:
            self._npu_backend.prepare_eager_profiles(log_fn=log_fn)

    @torch.inference_mode()
    def _encode_npu(
        self, vector: torch.Tensor, *, sample_rate: int, payload: list[int]
    ) -> torch.Tensor:
        import torchaudio

        model = self.model
        wave = vector.to(device="cpu")
        if sample_rate != model.sr:
            wave = torchaudio.functional.resample(wave, sample_rate, model.sr)
        power = wave.square().mean()
        if power == 0:
            return vector
        normalized = wave * torch.sqrt(model.average_energy_VCTK / power)
        magnitude, phase = model.stft.transform(normalized.unsqueeze(0))
        carrier = magnitude.unsqueeze(1)
        bits = "".join(f"{value:08b}" for value in payload)
        symbols = [int(bits[index : index + 2], 2) for index in range(0, len(bits) // 2 * 2, 2)]
        messages, _ = model.letters_encoding(carrier.shape[-1], [symbols])
        messages = torch.as_tensor(messages, dtype=torch.float32).unsqueeze(0)
        info = self._npu_backend(carrier, messages)
        if model.config.ensure_negative_message:
            info = info.abs()
        info[:, :, model.config.message_band_size :, :] = 0
        if not model.config.no_normalization:
            info = info / info.square().mean(dim=2, keepdim=True).sqrt()
            info = info / (10 ** (model.config.message_sdr / 20))
        if model.config.frame_level_normalization:
            info = info * carrier.square().mean(dim=2, keepdim=True).sqrt()
        elif model.config.utterance_level_normalization:
            info = info * carrier.square().mean(dim=(2, 3), keepdim=True).sqrt()
        if model.config.ensure_negative_message:
            spectrum = torch.nn.functional.relu(carrier - info)
        elif model.config.ensure_constrained_message:
            spectrum = carrier + info.clamp(min=-carrier, max=carrier)
        else:
            spectrum = (carrier + info).abs()
        model.stft.num_samples = wave.numel()
        encoded = model.stft.inverse(spectrum.squeeze(1), phase)[0, 0]
        encoded = encoded * torch.sqrt(power / model.average_energy_VCTK)
        if sample_rate != model.sr:
            encoded = torchaudio.functional.resample(encoded, model.sr, sample_rate)
        return encoded[: vector.numel()]

    def encode_one(
        self,
        audio: torch.Tensor,
        *,
        sample_rate: int,
        payload: Iterable[int] = IRODORI_WATERMARK_PAYLOAD,
    ) -> torch.Tensor:
        if self.model is None:
            return audio

        vector = _as_single_channel_vector(audio)
        if vector is None:
            return audio

        if self._npu_backend is not None:
            encoded = self._encode_npu(vector, sample_rate=int(sample_rate), payload=list(payload))
        else:
            encoded, _ = self.model.encode_wav(
                vector.to(self.model.device),
                int(sample_rate),
                list(payload),
                calc_sdr=False,
            )
        encoded_audio = torch.as_tensor(encoded, dtype=torch.float32, device="cpu")
        return _match_original_rank(encoded_audio, reference=audio)

    def encode_batch(self, audios: list[torch.Tensor], *, sample_rate: int) -> list[torch.Tensor]:
        if self.model is None:
            return audios
        return [self.encode_one(audio, sample_rate=sample_rate) for audio in audios]
