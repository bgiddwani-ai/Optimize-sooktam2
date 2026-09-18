# Sooktam2 optimization learnings

## Precision and correctness

1. Keep Sooktam2 DiT parameters in FP32. The validated AOTI path stores parameters in FP32 and uses `torch.autocast(..., bfloat16)` only for execution. This preserved finite, non-silent output; a lower-precision checkpoint conversion was not accepted as a replacement for FP32 DiT.
2. Treat DiT and vocoder as independent precision contracts. The optimized Triton path uses FP32 DiT with TensorRT Vocos. A numerical result is not valid until the complete waveform passes finite, non-empty, and RMS checks.
3. `tokenizer="cls"` plus the implementation's Hindi CLS setting (`cls_language="hindi"`) is a functional requirement. Do not silently substitute a generic tokenizer or an inferred language.

## CLS frontend cache

1. Native NFE=16 profiling attributed about 729 ms (29.6% of request wall
   time) to Hindi CLS tokenization, almost entirely `wordparse`; transliteration
   and nasal normalization were negligible. This cost is CPU-side and does not
   get faster merely by using TensorRT.
2. The prompt transcript is immutable across the Hindi-8 run. Cache its CLS
   token-string prefix, not embeddings or token IDs, and tokenize only the
   changing target. The validation run was token-identical for all eight pairs
   under both the API join and the legacy Triton BLS join; it reduced isolated
   CLS work from about 5.8 s to 1.6 s across eight targets (3.6x).
3. The AOTI API does not concatenate the caller's raw `reference_text`
   directly. `preprocess_ref_audio_text` first adds sentence punctuation and
   then `infer_batch_process` adds terminal whitespace. The cache key must
   reproduce that exact `".  "`-style join. Caching `reference_text + " "`
   looks plausible but is a miss and produces no latency benefit.
4. Do not concatenate cached pieces blindly. CLS tokenization has different
   semantics for whitespace and no-whitespace joins. The cache admits only a
   prefix that has passed exact equality against upstream `cls_tokenize_text`.

## Where AOTI helped

1. At 32 NFE, the same DiT conditioning work would otherwise recur at every ODE step. Precompute the conditional/unconditional text embeddings and RoPE once per request, then make the exported step accept those tensors.
2. The installed rotary embedding returned `(frequency_tensor, 1.0)`, not two tensors. Exporting the scale as a dynamic tensor was incorrect; it must remain the constant scalar in the compiled module.
3. The model does not expose public `mel_dim` or text-width fields. Derive them from `transformer.proj_out.out_features` and `transformer.text_embed.text_embed.embedding_dim` when building the AOTI example inputs.
4. A real dynamic shared object is preferable to describing `torch.compile` as AOTI. The validated route was `torch._export.aot_compile` followed by `aot_load`; the generated `.so` is GPU/PyTorch/ABI specific and must be rebuilt after those change.
5. AOTI BF16 improved C=1 throughput from 0.239 req/s (native FP32 eager) to 0.837 req/s on the measured A100: roughly 3.5x. It also substantially outperformed the current Triton BLS configuration on this workload.
6. AOTI already hoisted text embedding and RoPE outside its ODE loop, so adding
   CLS caching had a small but real C=1 improvement (0.2547 to 0.2523 E2E RTF,
   0.837 to 0.846 req/s); C=2 was effectively unchanged. This is expected once
   compiled DiT and waveform decoding dominate the remaining wall time.

## TensorRT-LLM prepared DiT runtime

1. Use the same architecture split as AOTI in the TensorRT adapter:
   `prepare_condition` performs text embedding, CFG conditioning, RoPE, and
   schedule preparation once; `forward` performs only the iterative DiT/CFG
   update. The engine already accepts these prepared tensors, so no model
   weight conversion or engine rebuild is required.
2. Avoid reallocating output buffers and rebinding a fresh noise/time tensor on
   every NFE. Use stable input addresses, bind shapes once for a request shape,
   and update noise and timestep contents in place. On the measured A100 this
   changed Triton system RTF from 0.7552 to 0.6335 at C=1 and from 0.7247 to
   0.6032 at C=2 (16.1% and 16.8% better), with all 8/8 audio quality gates
   passing in both rows.
3. Those TensorRT figures are a combined before/after result for exact CLS
   prefix caching plus prepared-DiT/buffer reuse. They should not be reported
   as an isolated engine-kernel speedup without a separate ablation. No engine
   rebuild was needed because the existing FP32 DiT engine already accepted the
   prepared condition, RoPE, noise, timestep, and length tensors.

## Concurrency and batching

1. A client concurrency value is not proof of model batching. Both native eager and AOTI serialize at the model boundary because the sampler/cache/RNG state is mutable. C=2 increased per-request latency while throughput remained nearly flat.
2. After the prepared-DiT update, Triton improved from 0.317 req/s at C=1 to
   0.333 req/s at C=2 (about 5%). C=2 nevertheless raised mean request latency
   from 3.115 s to 5.321 s because it includes queue/batch wait. Report both
   system RTF (0.6335 and 0.6032) and latency; a dynamic-batching configuration
   alone is not a guarantee of a useful throughput gain.
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
