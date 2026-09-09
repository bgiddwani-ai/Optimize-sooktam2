# Sooktam2 optimization learnings

## Precision and correctness

1. Keep Sooktam2 DiT parameters in FP32. The validated AOTI path stores parameters in FP32 and uses `torch.autocast(..., bfloat16)` only for execution. This preserved finite, non-silent output; a lower-precision checkpoint conversion was not accepted as a replacement for FP32 DiT.
2. Treat DiT and vocoder as independent precision contracts. The optimized Triton path uses FP32 DiT with TensorRT Vocos. A numerical result is not valid until the complete waveform passes finite, non-empty, and RMS checks.
3. `tokenizer="cls"` plus the implementation's Hindi CLS setting (`cls_language="hindi"`) is a functional requirement. Do not silently substitute a generic tokenizer or an inferred language.

## Where AOTI helped

1. At 32 NFE, the same DiT conditioning work would otherwise recur at every ODE step. Precompute the conditional/unconditional text embeddings and RoPE once per request, then make the exported step accept those tensors.
2. The installed rotary embedding returned `(frequency_tensor, 1.0)`, not two tensors. Exporting the scale as a dynamic tensor was incorrect; it must remain the constant scalar in the compiled module.
3. The model does not expose public `mel_dim` or text-width fields. Derive them from `transformer.proj_out.out_features` and `transformer.text_embed.text_embed.embedding_dim` when building the AOTI example inputs.
4. A real dynamic shared object is preferable to describing `torch.compile` as AOTI. The validated route was `torch._export.aot_compile` followed by `aot_load`; the generated `.so` is GPU/PyTorch/ABI specific and must be rebuilt after those change.
5. AOTI BF16 improved C=1 throughput from 0.239 req/s (native FP32 eager) to 0.837 req/s on the measured A100: roughly 3.5x. It also substantially outperformed the current Triton BLS configuration on this workload.

## Concurrency and batching

1. A client concurrency value is not proof of model batching. Both native eager and AOTI serialize at the model boundary because the sampler/cache/RNG state is mutable. C=2 increased per-request latency while throughput remained nearly flat.
2. Triton's preferred batch size of 2 did not materially improve this workload: 0.266 req/s at C=1 and 0.277 req/s at C=2. A dynamic-batching configuration alone is not a speedup.
3. Safe independent-request batching needs a redesigned sampler boundary: per-request random generators, independent reference audio/text preprocessing, padding/duration bucketing, valid CFG lane packing, and output-order restoration. Do not share the global RNG/cache state between requests.

## Benchmarking discipline

1. Use a fixed non-FLEURS workload for repeatability. The final workload uses the model's own prompt audio/transcript plus eight static varied-length Hindi targets.
2. Finite output is insufficient. Gate each WAV on: non-empty samples, all finite values, and RMS >= 0.001. All final rows passed 8/8.
3. Report system RTF as timed wall seconds divided by generated audio seconds, and report request latency separately. At C=2, per-request latency includes queueing while system RTF captures overall throughput.
4. Normalize by actual generated duration. The Triton BLS path produced a slightly different total audio duration than the Python paths, so text length alone is not a sufficient denominator.
5. The public Triton schema exposed waveform only. Without explicit backend timing outputs, use an E2E timing label instead of claiming isolated DiT or Vocos latency.

## Deployment pitfalls

1. FastAPI `File`/`Form` routes require `python-multipart`; this missing dependency allows model/AOTI initialization to succeed but prevents the HTTP route from starting.
2. The AOTI service needs a sequence limit matched to both the exported dynamic shape and the sampler clamp. The validated service used a maximum of 4096 frames.
3. The benchmark GPU was an A100 80GB PCIe, not an L20. Re-run all rows on an L20 before using the numbers as L20 capacity guidance.
