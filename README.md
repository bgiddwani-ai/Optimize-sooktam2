# Sooktam2: AOTI, TensorRT-LLM/Triton, and HF eager benchmark

This repository contains the final Sooktam2 Hindi CLS benchmark harnesses:

- Native Hugging Face eager FP32 inference.
- TensorRT-LLM FP32 DiT + TensorRT Vocos served through Triton.
- A BF16-autocast FastAPI service whose DiT ODE step is a real dynamic AOTInductor shared library.

The model parameters remain FP32 in every validated path. “AOTI BF16” means BF16 autocast execution with FP32 parameter storage; it is **not** a BF16 checkpoint conversion.

## Final measured result

Hardware: one NVIDIA A100 80GB PCIe. This is not an L20 result.

Workload: the Sooktam2 repository's bundled `ref.wav` and transcript, eight fixed varied-length Hindi target sentences, `tokenizer="cls"`, Hindi CLS language, and 32 NFE. One warmup request is excluded from each row. All eight measured WAVs in every row were finite and had RMS >= 0.001.

RTF below is the comparable system E2E value: timed workload wall time divided by total generated audio duration. Mean latency is individual request E2E latency and therefore includes queueing at C=2.

| Mode | Concurrency | E2E RTF | Throughput | Mean latency | P95 latency | Quality |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| HF eager, FP32 | 1 | 0.8940 | 0.239 req/s | 4.188 s | 5.368 s | Pass: 8/8 |
| HF eager, FP32 | 2 | 0.8923 | 0.239 req/s | 7.644 s | 9.913 s | Pass: 8/8 |
| TRT-LLM FP32 DiT + TRT Vocos + Triton | 1 | 0.7552 | 0.266 req/s | 3.722 s | 5.319 s | Pass: 8/8 |
| TRT-LLM FP32 DiT + TRT Vocos + Triton | 2 | 0.7247 | 0.277 req/s | 6.453 s | 9.152 s | Pass: 8/8 |
| AOTI BF16 autocast DiT + server | 1 | **0.2547** | **0.837 req/s** | **1.189 s** | **1.523 s** | Pass: 8/8 |
| AOTI BF16 autocast DiT + server | 2 | **0.2512** | **0.849 req/s** | **2.151 s** | **2.767 s** | Pass: 8/8 |

The source data for the table is saved in [`results/final_summary.json`](results/final_summary.json).

## Repository layout

```text
aoti_bf16_server.py             BF16-autocast AOTInductor FastAPI service
aoti_preflight.py               Dynamic-shape AOTI capability check
start_aoti_bf16_server.sh       Starts the service in a remote screen session
hindi8_workload.py              Shared non-FLEURS Hindi prompt/text workload
http_benchmark_client.py        Quality-gated Triton or AOTI HTTP benchmark client
native_hf_eager_benchmark.py    Quality-gated native eager FP32 benchmark
summarize_final.py              Produces FINAL.md and final_summary.json
references/benchmark_bf16.py    Supplied reference implementation, unmodified
results/                        Final measured table and machine-readable rows
```

The model checkpoint, TensorRT-LLM engine, TensorRT Vocos engine, and generated AOTI `.so` are intentionally excluded from Git.

## Prerequisites

The validated setup used:

- Python 3.12, PyTorch 2.5.1 + CUDA 12.4, `torch._export.aot_compile`, FastAPI, Uvicorn, SoundFile, Requests, Transformers, and `python-multipart`.
- A local Sooktam2 checkout that supports `AutoModel.from_pretrained(..., trust_remote_code=True)` and exposes `model.tts`.
- The Sooktam2/F5-TTS Python package on `PYTHONPATH`.
- For the TRT row: an already validated NFE=32 TensorRT-LLM **FP32 DiT** engine, TensorRT Vocos engine, and Triton `f5_tts` BLS model. The active endpoint used here was `http://127.0.0.1:8004`.

The TRT engine is a deployment prerequisite, not a generic TensorRT export: it must preserve Sooktam2 CLS tokenization, CFG, duration calculation, the FP32 DiT contract, and Vocos decoding. See the upstream [F5-TTS Triton TensorRT-LLM runtime](https://github.com/SWivid/F5-TTS/tree/main/src/f5_tts/runtime/triton_trtllm) for its baseline layout.

## Reproduce the benchmark

The following commands assume the validated remote layout. Run host-side commands from `/home/ubuntu/optimize-sooktam2`; use a remote `screen` session for each long-running process.

```bash
ROOT=/home/ubuntu/optimize-sooktam2
BUILDER=sooktam2-build-cuda12

# The repo files must be visible in the builder at /workspace.
docker exec "$BUILDER" env PYTHONPATH=/workspace python3 -m py_compile \
  /workspace/aoti_bf16_server.py /workspace/http_benchmark_client.py \
  /workspace/native_hf_eager_benchmark.py /workspace/hindi8_workload.py

# Required once for FastAPI file uploads.
docker exec "$BUILDER" python3 -m pip install python-multipart

# Verify the installed PyTorch can produce and load a dynamic AOTI shared object.
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

### 3. Run native HF eager FP32 rows

```bash
# Run C=1 to completion before C=2.
screen -dmS sooktam2_hf_c1 bash -lc \
  "docker exec $BUILDER env PYTHONPATH=/workspace:/workspace/src/sooktam2/src \
    python3 /workspace/native_hf_eager_benchmark.py \
    --model-dir /workspace/src/sooktam2 \
    --reference-audio /workspace/src/sooktam2/ref.wav --concurrency 1 \
    --output-dir /workspace/artifacts/bench_hindi8_nfe32/hf_eager_fp32_c1"

screen -dmS sooktam2_hf_c2 bash -lc \
  "docker exec $BUILDER env PYTHONPATH=/workspace:/workspace/src/sooktam2/src \
    python3 /workspace/native_hf_eager_benchmark.py \
    --model-dir /workspace/src/sooktam2 \
    --reference-audio /workspace/src/sooktam2/ref.wav --concurrency 2 \
    --output-dir /workspace/artifacts/bench_hindi8_nfe32/hf_eager_fp32_c2"
```

This path disables TF32 and autocast, so it is native eager FP32. C=2 is client concurrency over a single locked model, not unsafe parallel calls to the mutable sampler.

### 4. Start the prebuilt Triton stack and run its rows

Start the validated NFE=32 Triton model repository and verify it before testing:

```bash
curl -fsS http://127.0.0.1:8004/v2/health/ready
curl -fsS http://127.0.0.1:8004/v2/models/f5_tts/config
```

Its model config must identify Hindi CLS, `nfe_steps: 32`, a TensorRT-LLM FP32 DiT engine, and Vocos. Then run:

```bash
# Run C=1 to completion before C=2.
screen -dmS sooktam2_triton_c1 bash -lc \
  "docker exec $BUILDER env PYTHONPATH=/workspace python3 /workspace/http_benchmark_client.py \
    --backend triton --endpoint http://127.0.0.1:8004 \
    --reference-audio /workspace/src/sooktam2/ref.wav --concurrency 1 \
    --output-dir /workspace/artifacts/bench_hindi8_nfe32/triton_fp32_c1"

screen -dmS sooktam2_triton_c2 bash -lc \
  "docker exec $BUILDER env PYTHONPATH=/workspace python3 /workspace/http_benchmark_client.py \
    --backend triton --endpoint http://127.0.0.1:8004 \
    --reference-audio /workspace/src/sooktam2/ref.wav --concurrency 2 \
    --output-dir /workspace/artifacts/bench_hindi8_nfe32/triton_fp32_c2"
```

The deployed BLS API exposes waveform only, so its report uses the same system E2E RTF boundary as the other modes rather than inventing a backend-only latency.

### 5. Validate and summarize

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
- AOTI and native eager have a serialized model boundary. Real server-side microbatching would need independent-reference batching, duration bucketing, and deterministic per-request RNG handling.
