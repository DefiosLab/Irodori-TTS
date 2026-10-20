# Irodori-TTS-Server on Intel NPU

Use this NPU-enabled Irodori-TTS checkout with
[Irodori-TTS-Server](https://github.com/Aratako/Irodori-TTS-Server) to serve the
OpenAI-compatible speech API. This guide covers the NPU-specific setup for a fresh
local Python installation. See the server's README for general setup and usage.

The [server patch](patches/tts-server-npu.patch) targets server commit
`61012c760f22f7b4a6c21c5c5f8f9e148120b6f9` (`Support MeanFlow inference defaults`).
It makes three changes:

- Uses the sibling Irodori-TTS checkout as an editable dependency.
- Adds a server `npu` extra that selects Irodori-TTS's `npu` dependencies and CPU
  PyTorch wheels. Dependency version constraints remain in Irodori-TTS.
- Uses `acquire_runtime()` so model loading includes NPU compilation and warmup
  before the server reports the runtime as loaded.

## Prerequisites and Layout

Install the Intel NPU driver and keep the two repositories in sibling directories:

```text
parent/
├── Irodori-TTS/          # This NPU-enabled checkout
└── Irodori-TTS-Server/
```

The relative dependency path in the patch is `../Irodori-TTS`. Change it if you use
a different layout. The server environment installs this checkout directly; the
virtual environment used for standalone Irodori-TTS inference is separate.

## Apply the Patch and Install

From the server directory, select the patch's target commit and apply the patch
before installing the NPU extra:

```bash
git switch --detach 61012c760f22f7b4a6c21c5c5f8f9e148120b6f9
git apply --check ../Irodori-TTS/docs/patches/tts-server-npu.patch
git apply ../Irodori-TTS/docs/patches/tts-server-npu.patch
uv sync --extra npu
```

Confirm that imports resolve to this checkout and that OpenVINO lists `NPU`:

```bash
uv run --no-sync python -c "import irodori_tts, openvino as ov; print(irodori_tts.__file__); print(ov.Core().available_devices)"
```

Use `uv run --no-sync` after installation to preserve the selected backend extra.

## Configure the Server

Set the following values in the server's `.env`:

```dotenv
IRODORI_HF_CHECKPOINT=Aratako/Irodori-TTS-v4.1-Small
IRODORI_MODEL_DEVICE=npu
IRODORI_CODEC_DEVICE=npu
IRODORI_MODEL_PRECISION=fp32
IRODORI_CODEC_PRECISION=fp32
IRODORI_COMPILE_MODEL=false
IRODORI_PRELOAD=true
```

`IRODORI_PRELOAD=true` completes model loading and NPU preparation during startup.
With `false`, this work happens on the first speech request. Initial compilation
can take longer than a client's request timeout. Preparation logs use the
`[openvino]` prefix and finish with `[openvino] NPU ready (...)`.

## NPU Notes

- Select `npu` explicitly for each target. `auto` does not select NPU. The model and
  codec targets can differ, for example model `npu` with codec `cpu`.
- Keep NPU target precision settings at `fp32`. They control CPU-side PyTorch
  weights and floating-point tensors; Intel NPU floating-point computation runs
  in `fp16`.
- Keep `IRODORI_COMPILE_MODEL=false` when the model target is `npu`.
- Per-request `irodori.lora_adapter` is unavailable when the model target is `npu`.
  [Merge the adapter into a checkpoint](../README.md#6-convert-a-training-checkpoint)
  first, then select it with `IRODORI_CHECKPOINT`.
- NPU caches are stored beside the model, codec, and watermark checkpoints in
  `<checkpoint-name>.openvino/` directories. Keep those locations writable so
  subsequent launches can reuse the caches.
