# Sooktam2 Hindi8 benchmark (NFE=32)

GPU: NVIDIA A100 80GB PCIe

RTF is timed run wall seconds / generated audio seconds. One warmup request is excluded; C=2 mean latency includes queueing.

| Mode | Concurrency | E2E RTF | Throughput | Mean latency | P95 latency | Quality |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| HF eager, FP32 | 1 | 0.8940 | 0.239 req/s | 4.188 s | 5.368 s | Pass: 8/8 |
| HF eager, FP32 | 2 | 0.8923 | 0.239 req/s | 7.644 s | 9.913 s | Pass: 8/8 |
| TRT-LLM FP32 DiT + TRT Vocos + Triton | 1 | 0.7552 | 0.266 req/s | 3.722 s | 5.319 s | Pass: 8/8 |
| TRT-LLM FP32 DiT + TRT Vocos + Triton | 2 | 0.7247 | 0.277 req/s | 6.453 s | 9.152 s | Pass: 8/8 |
| AOTI BF16 autocast DiT + server | 1 | 0.2547 | 0.837 req/s | 1.189 s | 1.523 s | Pass: 8/8 |
| AOTI BF16 autocast DiT + server | 2 | 0.2512 | 0.849 req/s | 2.151 s | 2.767 s | Pass: 8/8 |
