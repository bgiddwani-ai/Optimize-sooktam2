# CLS cache + prepared DiT benchmark

One A100 80GB PCIe, fixed Hindi-8 workload, Hindi CLS, NFE=32, one excluded
warmup, and the same quality gate in every row (finite, non-empty WAV and RMS
at least 0.001).

RTF is whole-run wall seconds divided by actual generated audio seconds. Mean
latency is per-request E2E latency, including queueing at C=2.

| Runtime | C | Before RTF | After RTF | Before / after throughput | Before / after mean latency | Quality |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| AOTI BF16 autocast | 1 | 0.2547 | 0.2523 | 0.837 / 0.846 req/s | 1.189 / 1.179 s | 8/8 / 8/8 |
| AOTI BF16 autocast | 2 | 0.2512 | 0.2518 | 0.849 / 0.847 req/s | 2.151 / 2.158 s | 8/8 / 8/8 |
| TRT-LLM FP32 DiT + TRT Vocos + Triton | 1 | 0.7552 | 0.6335 | 0.266 / 0.317 req/s | 3.722 / 3.115 s | 8/8 / 8/8 |
| TRT-LLM FP32 DiT + TRT Vocos + Triton | 2 | 0.7247 | 0.6032 | 0.277 / 0.333 req/s | 6.453 / 5.320 s | 8/8 / 8/8 |

The TensorRT/Triton improvements are 16.1% (C=1) and 16.8% (C=2) in system
RTF. The AOTI C=1 result improves 1.0%; C=2 is effectively unchanged within
run variation. Raw `summary.json` copies for every before/after row are next to
this file.
