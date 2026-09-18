# Sooktam2: AOTI, TensorRT-LLM/Triton, and HF eager benchmark

Three benchmark harnesses for Sooktam2 Hindi CLS inference:

- Native Hugging Face eager FP32.
- TensorRT-LLM FP32 DiT + TensorRT Vocos served through Triton.
- A BF16-autocast FastAPI service whose DiT ODE step is a real dynamic AOTInductor shared library.

Model parameters stay FP32 in every validated path. "AOTI BF16" means BF16 autocast execution with FP32 parameter storage; it is **not** a BF16 checkpoint conversion.

## CLS-cache and prepared-DiT result

Hardware: one NVIDIA A100 80GB PCIe. This is not an L20 result.

Workload: the bundled `ref.wav` and transcript, eight fixed varied-length Hindi target sentences, `tokenizer="cls"`, Hindi CLS language, 32 NFE. One warmup request is excluded from each row. All eight measured WAVs in every row were finite with RMS >= 0.001.

E2E RTF is timed workload wall time divided by total generated audio duration. Mean latency is per-request E2E latency, so it includes queueing at C=2.

| Mode | Concurrency | E2E RTF | Throughput | Mean latency | P95 latency | Quality |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| HF eager, FP32 | 1 | 0.8940 | 0.239 req/s | 4.188 s | 5.368 s | Pass: 8/8 |
| TRT-LLM FP32 DiT + TRT Vocos + Triton, CLS cache + prepared inputs | 1 | 0.6335 | 0.317 req/s | 3.115 s | 4.712 s | Pass: 8/8 |
| TRT-LLM FP32 DiT + TRT Vocos + Triton, CLS cache + prepared inputs | 2 | 0.6032 | 0.333 req/s | 5.320 s | 7.942 s | Pass: 8/8 |
| AOTI BF16 autocast DiT + server, CLS cache | 1 | **0.2523** | **0.846 req/s** | **1.179 s** | **1.493 s** | Pass: 8/8 |
| AOTI BF16 autocast DiT + server, CLS cache | 2 | **0.2518** | **0.847 req/s** | **2.158 s** | **2.762 s** | Pass: 8/8 |

The TensorRT/Triton change improved system RTF by 16.1% at C=1 and 16.8% at
C=2 against its immediately preceding matched NFE=32 run. AOTI was already
compiled and static-condition hoisted, so the CLS cache changed C=1 by 1.0%
and C=2 was effectively flat. The TensorRT figures are a combined result for
exact CLS prefix caching plus prepared-input/buffer reuse, not an isolated
engine-kernel speedup.

Source data: [`results/clsopt_nfe32`](results/clsopt_nfe32).

## Repository layout

```text
aoti_bf16_server.py             BF16-autocast AOTInductor FastAPI service
validate_cls_cache.py            Exact CLS cache parity test for Hindi-8
aoti_preflight.py               Dynamic-shape AOTI capability check
start_aoti_bf16_server.sh       Server launch wrapper (holds the exact server flags)
hindi8_workload.py              Shared Hindi prompt/text workload
http_benchmark_client.py        Quality-gated Triton or AOTI HTTP benchmark client
summarize_final.py              Produces FINAL.md and final_summary.json
references/benchmark_bf16.py    Supplied reference implementation, unmodified
results/                        Final measured table and machine-readable rows
triton/f5_tts/1/                Deployed BLS frontend and TensorRT-LLM DiT adapter
sooktam2_src/f5_tts/infer/      Shared CLS prefix-cache source
```

The model checkpoint, TensorRT-LLM engine, TensorRT Vocos engine, and generated AOTI `.so` are excluded from Git.

## Prerequisites

The validated setup used:

- Python 3.12, PyTorch 2.5.1 + CUDA 12.4, `torch._export.aot_compile`, FastAPI, Uvicorn, SoundFile, Requests, Transformers, `python-multipart`.
- A local Sooktam2 checkout supporting `AutoModel.from_pretrained(..., trust_remote_code=True)` and exposing `model.tts`.
- The Sooktam2/F5-TTS Python package on `PYTHONPATH`.
- For the TRT row only: an already validated NFE=32 TensorRT-LLM **FP32 DiT** engine, TensorRT Vocos engine, and Triton `f5_tts` BLS model, served at `http://127.0.0.1:8004`.

The TRT engine is a deployment prerequisite, not a generic TensorRT export. It must preserve Sooktam2 CLS tokenization, CFG, duration calculation, the FP32 DiT contract, and Vocos decoding. See the upstream [F5-TTS Triton TensorRT-LLM runtime](https://github.com/SWivid/F5-TTS/tree/main/src/f5_tts/runtime/triton_trtllm) for the baseline layout.

## Conventions used below

Commands are split into two kinds:

- **Host** — run from `/home/ubuntu/optimize-sooktam2` on the machine running Docker.
- **Container** — run inside the builder container, after opening a shell into it.

The repo must be visible inside the container at `/workspace`.

The AOTI server needs to keep running while its benchmarks execute, so it gets its own container shell. Everything else runs in the foreground, one command at a time. Benchmark rows must not overlap.

---

## Step 0 — Open a container shell

**Host:**

```bash
docker exec -it sooktam2-build-cuda12 bash
```

**Container** (run once per shell you open):

```bash
cd /workspace
export PYTHONPATH=/workspace:/workspace/src/sooktam2/src
```

## Step 1 — One-time setup and checks

**Container:**

```bash
# Required for FastAPI file uploads.
python3 -m pip install python-multipart

# Syntax check.
python3 -m py_compile aoti_bf16_server.py http_benchmark_client.py \
  native_hf_eager_benchmark.py hindi8_workload.py

# Verify the installed PyTorch can produce and load a dynamic AOTI shared object.
python3 aoti_preflight.py 2>&1 | tee logs/aoti_preflight.log

# Cache activation is conditional on exact upstream token parity for both
# the AOTI API join and the Triton legacy BLS join.
python3 validate_cls_cache.py
```

Expected output reports `token_parity: true` for both joins. The cache stores
only immutable Hindi CLS token-string prefixes; weights, token IDs, and target
tokenization remain unchanged.

## Step 2 — AOTI BF16 rows

### 2a. Start the server

Open a second container shell (Step 0) and leave this running:

**Container (shell 2):**

```bash
python3 aoti_bf16_server.py
```

`start_aoti_bf16_server.sh` holds the exact flags and log paths if you need them. To run detached instead of holding a shell:

```bash
mkdir -p logs
nohup python3 aoti_bf16_server.py > logs/aoti_bf16_server.log 2>&1 &
```

The first launch builds `artifacts/aoti_bf16/sooktam2_dit_bf16_aoti.so`; later launches load it.

### 2b. Confirm it is up

**Container (shell 1):**

```bash
curl -fsS http://127.0.0.1:8010/healthz
```

Health metadata should include `backend: aoti-bf16`, `nfe_steps: 32`, and `model_boundary: serialized`.

### 2c. Run the rows

**Container (shell 1)** — C=1 first, then C=2:

```bash
python3 http_benchmark_client.py \
  --backend aoti --endpoint http://127.0.0.1:8010 \
  --reference-audio src/sooktam2/ref.wav --concurrency 1 \
  --output-dir artifacts/bench_hindi8_nfe32/aoti_bf16_c1

python3 http_benchmark_client.py \
  --backend aoti --endpoint http://127.0.0.1:8010 \
  --reference-audio src/sooktam2/ref.wav --concurrency 2 \
  --output-dir artifacts/bench_hindi8_nfe32/aoti_bf16_c2
```

Stop the server (Ctrl-C in shell 2, or `pkill -f aoti_bf16_server.py`) before moving on.

## Step 3 — Native HF eager FP32 rows

**Container** — C=1 first, then C=2:

```bash
python3 native_hf_eager_benchmark.py \
  --model-dir src/sooktam2 \
  --reference-audio src/sooktam2/ref.wav --concurrency 1 \
  --output-dir artifacts/bench_hindi8_nfe32/hf_eager_fp32_c1

python3 native_hf_eager_benchmark.py \
  --model-dir src/sooktam2 \
  --reference-audio src/sooktam2/ref.wav --concurrency 2 \
  --output-dir artifacts/bench_hindi8_nfe32/hf_eager_fp32_c2
```

The custom DiT conversion must include the equivalent of `--dtype float32`.
The engine profiles must cover the longest prompt-plus-generation sequence, and
the BLS configuration must explicitly retain `tokenizer="cls"`, Hindi CLS, and
`nfe_steps: 32`. Do not run the upstream `MODEL=F5TTS_v1_Base` recipe against a
Sooktam2 checkpoint.

## Step 4 — Triton rows

Start the validated NFE=32 Triton model repository outside this container, then verify it.

**Container:**

```bash
curl -fsS http://127.0.0.1:8004/v2/health/ready
curl -fsS http://127.0.0.1:8004/v2/models/f5_tts/config
```

The config must identify Hindi CLS, `nfe_steps: 32`, a TensorRT-LLM FP32 DiT engine, and Vocos.

**Container** — C=1 first, then C=2:

```bash
python3 http_benchmark_client.py \
  --backend triton --endpoint http://127.0.0.1:8004 \
  --reference-audio src/sooktam2/ref.wav --concurrency 1 \
  --output-dir artifacts/bench_hindi8_nfe32/triton_fp32_c1

python3 http_benchmark_client.py \
  --backend triton --endpoint http://127.0.0.1:8004 \
  --reference-audio src/sooktam2/ref.wav --concurrency 2 \
  --output-dir artifacts/bench_hindi8_nfe32/triton_fp32_c2
```

The deployed BLS API exposes waveform only, so this row uses the same system E2E RTF boundary as the others rather than inventing a backend-only latency.

## Step 5 — Summarize

Each client writes WAVs, `requests.jsonl`, and `summary.json`, and exits non-zero if any output is empty, non-finite, or has RMS below 0.001.

**Container:**

```bash
python3 summarize_final.py --root artifacts/bench_hindi8_nfe32
cat artifacts/bench_hindi8_nfe32/FINAL.md
```

## Scope notes

- Compare RTF only within the stated end-to-end boundary. Triton did not expose an internal DiT/Vocos duration tensor in its public model schema.
- Generated audio duration differed slightly across the BLS and Python paths, so RTF is normalized by each row's actual output duration.
- AOTI and native eager have a serialized model boundary. Real server-side microbatching would need independent-reference batching, duration bucketing, and deterministic per-request RNG handling.
