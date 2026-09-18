# Sooktam2: AOTI and TensorRT-LLM/Triton serving

This repository provides installation and serving instructions for the final
Sooktam2 Hindi CLS optimized paths:

- TensorRT-LLM FP32 DiT + TensorRT Vocos served through Triton.
- A BF16-autocast FastAPI service whose DiT ODE step is a real dynamic AOTInductor shared library.

The native Hugging Face eager harness remains in the repository only as the
benchmark baseline; this README does not provide an eager installation path.

The model parameters remain FP32 in every validated path. “AOTI BF16” means BF16 autocast execution with FP32 parameter storage; it is **not** a BF16 checkpoint conversion.

## CLS-cache and prepared-DiT result

Hardware: one NVIDIA A100 80GB PCIe. This is not an L20 result.

Workload: the Sooktam2 repository's bundled `ref.wav` and transcript, eight fixed varied-length Hindi target sentences, `tokenizer="cls"`, Hindi CLS language, and 32 NFE. One warmup request is excluded from each row. All eight measured WAVs in every row were finite and had RMS >= 0.001.

RTF below is the comparable system E2E value: timed workload wall time divided by total generated audio duration. Mean latency is individual request E2E latency and therefore includes queueing at C=2.

| Mode | Concurrency | E2E RTF | Throughput | Mean latency | P95 latency | Quality |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| HF eager, FP32 | 1 | 0.8940 | 0.239 req/s | 4.188 s | 5.368 s | Pass: 8/8 |
| TRT-LLM FP32 DiT + TRT Vocos + Triton, CLS cache + prepared inputs | 1 | 0.6335 | 0.317 req/s | 3.115 s | 4.712 s | Pass: 8/8 |
| TRT-LLM FP32 DiT + TRT Vocos + Triton, CLS cache + prepared inputs | 2 | 0.6032 | 0.333 req/s | 5.320 s | 7.942 s | Pass: 8/8 |
| AOTI BF16 autocast DiT + server, CLS cache | 1 | **0.2523** | **0.846 req/s** | **1.179 s** | **1.493 s** | Pass: 8/8 |
| AOTI BF16 autocast DiT + server, CLS cache | 2 | **0.2518** | **0.847 req/s** | **2.158 s** | **2.762 s** | Pass: 8/8 |

The TensorRT/Triton change improved system RTF by 16.1% at C=1 and 16.8% at
C=2 versus its immediately preceding matched NFE=32 run. AOTI was already
largely dominated by the compiled DiT; its C=1 result improved by 1.0% and C=2
was statistically flat (+0.2% RTF). The before/after summaries and raw
quality-gated request data are in [`results/clsopt_nfe32`](results/clsopt_nfe32).

## Repository layout

```text
aoti_bf16_server.py             BF16-autocast AOTInductor FastAPI service
validate_cls_cache.py            Exact CLS cache parity test for Hindi-8
aoti_preflight.py               Dynamic-shape AOTI capability check
start_aoti_bf16_server.sh       Starts the service in a remote screen session
hindi8_workload.py              Shared non-FLEURS Hindi prompt/text workload
http_benchmark_client.py        Quality-gated Triton or AOTI HTTP benchmark client
summarize_final.py              Produces FINAL.md and final_summary.json
references/benchmark_bf16.py    Supplied reference implementation, unmodified
results/                        Final measured table and machine-readable rows
triton/f5_tts/1/                Deployed BLS frontend and TensorRT-LLM DiT adapter
sooktam2_src/f5_tts/infer/      Shared CLS prefix-cache source
```

The model checkpoint, TensorRT-LLM engine, TensorRT Vocos engine, and generated AOTI `.so` are intentionally excluded from Git.

## Prerequisites

The validated setup used:

- Python 3.12, PyTorch 2.5.1 + CUDA 12.4, `torch._export.aot_compile`, FastAPI, Uvicorn, SoundFile, Requests, Transformers, and `python-multipart`.
- A local Sooktam2 checkout that supports `AutoModel.from_pretrained(..., trust_remote_code=True)` and exposes `model.tts`.
- The Sooktam2/F5-TTS Python package on `PYTHONPATH`.
- For the TRT row: an already validated NFE=32 TensorRT-LLM **FP32 DiT** engine, TensorRT Vocos engine, and Triton `f5_tts` BLS model. The active endpoint used here was `http://127.0.0.1:8004`.

The TRT engine is a deployment prerequisite, not a generic TensorRT export: it must preserve Sooktam2 CLS tokenization, CFG, duration calculation, the FP32 DiT contract, and Vocos decoding. See the upstream [F5-TTS Triton TensorRT-LLM runtime](https://github.com/SWivid/F5-TTS/tree/main/src/f5_tts/runtime/triton_trtllm) for its baseline layout.

### Materialize the Sooktam2 checkpoint

The model checkout must contain real model weights. A file beginning with
`version https://git-lfs.github.com/spec` is an LFS pointer, not a usable
checkpoint.

```bash
sudo apt-get update
sudo apt-get install -y git-lfs
git lfs install
cd /home/ubuntu/optimize-sooktam2/src/sooktam2
git lfs pull
```

### Install the AOTI service dependencies

Keep the validated CUDA-enabled PyTorch in place; do **not** install a generic
new Torch wheel. The AOTI service needs `torch._export.aot_compile` and
`aot_load` from that existing build.

```bash
ROOT=/home/ubuntu/optimize-sooktam2
BUILDER=sooktam2-build-cuda12

docker start "$BUILDER" 2>/dev/null || true
docker inspect -f '{{.State.Running}}' "$BUILDER" | grep -qx true
docker exec "$BUILDER" python3 -m pip install \
  fastapi 'uvicorn[standard]' python-multipart \
  numpy soundfile torchdiffeq transformers requests

# Only required if `import f5_tts` is not already available in the builder.
docker exec "$BUILDER" python3 -m pip install -e /workspace/src/F5-TTS

docker exec "$BUILDER" env PYTHONPATH=/workspace/src/sooktam2/src python3 - <<'PY'
import torch
from torch._export import aot_compile, aot_load
from f5_tts.model.backbones.dit import DiT
from f5_tts.model.cfm import CFM
assert torch.cuda.is_available(), "CUDA-enabled PyTorch is required"
print(torch.__version__, torch.version.cuda)
print("AOTI imports: OK")
PY
```

### Install the CLS cache source

Copy `sooktam2_src/f5_tts/infer/cls_token_cache.py` into the Sooktam2
checkout at `src/f5_tts/infer/cls_token_cache.py`. It caches only CLS token
strings for the immutable prompt prefix; vocabulary lookup and all model
weights stay unchanged. Before serving, prove that cached and upstream tokens
are identical for both API and legacy Triton text joins:

```bash
docker exec "$BUILDER" env PYTHONPATH=/workspace/src/sooktam2/src:/workspace \
  python3 /workspace/validate_cls_cache.py
```

The expected result reports `token_parity: true` for `aoti_api_join` and
`triton_legacy_join`. Do not enable the cache if that check fails.

## Reproduce the benchmark

The following commands assume the validated remote layout. Run host-side commands from `/home/ubuntu/optimize-sooktam2`; use a remote `screen` session for each long-running process.

```bash
ROOT=/home/ubuntu/optimize-sooktam2
BUILDER=sooktam2-build-cuda12

# The repo files must be visible in the builder at /workspace.
docker exec "$BUILDER" env PYTHONPATH=/workspace python3 -m py_compile \
  /workspace/aoti_bf16_server.py /workspace/http_benchmark_client.py \
  /workspace/hindi8_workload.py

# Required once for FastAPI file uploads.
docker exec "$BUILDER" python3 -m pip install python-multipart

# Verify the installed PyTorch can produce and load a dynamic AOTI shared object.
mkdir -p "$ROOT/logs"
screen -dmS sooktam2_aoti_preflight bash -lc \
  "docker exec $BUILDER env PYTHONPATH=/workspace/src/sooktam2/src \
   python3 /workspace/aoti_preflight.py | tee $ROOT/logs/aoti_preflight.log"
```

### 1. Start AOTI BF16 service

```bash
bash "$ROOT/start_aoti_bf16_server.sh"
curl -fsS http://127.0.0.1:8010/healthz
```

Expected health metadata includes `backend: aoti-bf16`, `nfe_steps: 32`, and `model_boundary: serialized`. The first launch creates `artifacts/aoti_bf16/sooktam2_dit_bf16_aoti.so`; later launches load it.
It also reports `cls_prefix_cache`; benchmark requests should increase its hit
count without increasing misses for the fixed prompt.

### 2. Run AOTI rows

```bash
# Wait for this session to finish before launching the C=2 command below.
screen -dmS sooktam2_aoti_c1 bash -lc \
  "docker exec $BUILDER env PYTHONPATH=/workspace python3 /workspace/http_benchmark_client.py \
    --backend aoti --endpoint http://127.0.0.1:8010 \
    --reference-audio /workspace/src/sooktam2/ref.wav --concurrency 1 \
    --output-dir /workspace/artifacts/bench_hindi8_nfe32/aoti_bf16_c1"

screen -dmS sooktam2_aoti_c2 bash -lc \
  "docker exec $BUILDER env PYTHONPATH=/workspace python3 /workspace/http_benchmark_client.py \
    --backend aoti --endpoint http://127.0.0.1:8010 \
    --reference-audio /workspace/src/sooktam2/ref.wav --concurrency 2 \
    --output-dir /workspace/artifacts/bench_hindi8_nfe32/aoti_bf16_c2"
```

Run those commands sequentially, not simultaneously, to avoid cross-run interference.

### 3. Install and launch the TensorRT-LLM/Triton stack

The upstream [F5-TTS Triton TensorRT-LLM runtime](https://github.com/SWivid/F5-TTS/tree/main/src/f5_tts/runtime/triton_trtllm)
is a structural baseline, not a drop-in converter for Sooktam2. The custom
Sooktam2 conversion and BLS model must preserve Hindi CLS, CFG, duration
calculation, FP32 DiT, and Vocos decoding.

The measured service used `optimize-sooktam2-triton-trtllm:24.12-cuda12` on
the tested R565 driver stack. Do not substitute a newer TensorRT/CUDA stack
without rebuilding and revalidating the engine.

Before launch, provide these non-Git artifacts and verify them as a unit:

```text
$ROOT/artifacts/tllm_sooktam2_fp32/       FP32 TensorRT-LLM DiT engine directory
$ROOT/artifacts/vocos_sooktam2_fp16.plan  TensorRT Vocos engine
$ROOT/model_repository_sooktam2/f5_tts/   Triton BLS model plus config.pbtxt
```

The custom DiT conversion must include the equivalent of `--dtype float32`.
The engine profiles must cover the longest prompt-plus-generation sequence, and
the BLS configuration must explicitly retain `tokenizer="cls"`, Hindi CLS, and
`nfe_steps: 32`. Do not run the upstream `MODEL=F5TTS_v1_Base` recipe against a
Sooktam2 checkpoint.

Launch the validated custom image only after the artifacts above are ready:

```bash
TRITON_IMAGE=optimize-sooktam2-triton-trtllm:24.12-cuda12
MODEL_REPOSITORY="$ROOT/artifacts/model_repository_nfe32"

docker rm -f sooktam2-triton 2>/dev/null || true
docker run -d --name sooktam2-triton --gpus all --net host --shm-size=2g \
  -v "$ROOT:/workspace" \
  "$TRITON_IMAGE" \
  bash -lc "export PYTHONPATH=/workspace/src/sooktam2/src:\${PYTHONPATH:-}; \
    exec tritonserver --model-repository=$MODEL_REPOSITORY --grpc-port=8003 --http-port=8004 --metrics-port=8005"
```

Start the validated NFE=32 Triton model repository and verify it before testing:

```bash
curl -fsS http://127.0.0.1:8004/v2/health/ready
curl -fsS http://127.0.0.1:8004/v2/models/f5_tts/config
```

Its model config must identify Hindi CLS, `nfe_steps: 32`, a TensorRT-LLM FP32 DiT engine, and Vocos. Then run:

```bash
# Run C=1 to completion before C=2.
screen -dmS sooktam2_triton_c1 bash -lc \
  "docker exec -w /workspace sooktam2-triton env PYTHONPATH=/workspace python3 /workspace/http_benchmark_client.py \
    --backend triton --endpoint http://127.0.0.1:8004 \
    --reference-audio /workspace/src/sooktam2/ref.wav --concurrency 1 \
    --output-dir /workspace/artifacts/bench_hindi8_nfe32/triton_fp32_c1"

screen -dmS sooktam2_triton_c2 bash -lc \
  "docker exec -w /workspace sooktam2-triton env PYTHONPATH=/workspace python3 /workspace/http_benchmark_client.py \
    --backend triton --endpoint http://127.0.0.1:8004 \
    --reference-audio /workspace/src/sooktam2/ref.wav --concurrency 2 \
    --output-dir /workspace/artifacts/bench_hindi8_nfe32/triton_fp32_c2"
```

The deployed BLS API exposes waveform only, so its report uses the same system E2E RTF boundary as the other modes rather than inventing a backend-only latency.

### 4. Validate and summarize

Each client writes WAVs, `requests.jsonl`, and `summary.json`. It exits non-zero if any output is empty, non-finite, or has RMS below 0.001.

```bash
docker exec "$BUILDER" env PYTHONPATH=/workspace python3 /workspace/summarize_final.py \
  --root /workspace/artifacts/bench_hindi8_nfe32
cat "$ROOT/artifacts/bench_hindi8_nfe32/FINAL.md"
```

## Important scope notes

- This workload replaced FLEURS completely; it is eight fixed Hindi targets to make runs repeatable.
- Compare RTF only within the stated end-to-end boundary. Triton did not expose an internal DiT/Vocos duration tensor in its public model schema.
- The generated audio duration differed slightly across the BLS and Python paths, so RTF is normalized by each row's actual output duration.
- AOTI has a serialized model boundary. Real server-side microbatching would need independent-reference batching, duration bucketing, and deterministic per-request RNG handling.
- The TensorRT adapter's `prepare_condition` builds text embeddings, CFG lanes,
  RoPE, and input lengths exactly once. Its `forward` reuses the bound engine
  inputs and output buffers through every NFE step, updating only noise and
  timestep contents.

For the upstream F5-TTS Triton structure and the open items that make custom
checkpoint deployment non-trivial, see the [Triton runtime README](https://github.com/SWivid/F5-TTS/tree/main/src/f5_tts/runtime/triton_trtllm)
and [issue #1182](https://github.com/SWivid/F5-TTS/issues/1182).
