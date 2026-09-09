#!/usr/bin/env python3
"""FP32 HF-eager Sooktam2 benchmark on the shared non-FLEURS Hindi8 workload."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import statistics
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModel

from hindi8_workload import HINDI_TARGETS, REFERENCE_TEXT, request_seed, workload_metadata


QUALITY_RMS_FLOOR = 1.0e-3


def percentile(values: list[float], fraction: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def inspect_waveform(waveform: np.ndarray) -> dict[str, object]:
    finite = bool(waveform.size and np.isfinite(waveform).all())
    rms = float(np.sqrt(np.mean(np.square(waveform)))) if finite else float("nan")
    return {
        "finite": finite,
        "samples": int(waveform.size),
        "rms": rms,
        "peak": float(np.max(np.abs(waveform))) if finite else float("nan"),
        "passed": bool(finite and rms >= QUALITY_RMS_FLOOR),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("/workspace/src/sooktam2"))
    parser.add_argument("--reference-audio", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, choices=(1, 2), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=1)
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=False)

    # This is native eager FP32: no torch.compile, no autocast, and no TF32
    # reduced-precision matmuls.  The upstream model has mutable sampling/cache
    # state, so C=2 is an honest client-concurrency measurement with a locked
    # single eager model rather than an unsafe concurrent model call.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    model = AutoModel.from_pretrained(
        str(arguments.model_dir), trust_remote_code=True, local_files_only=True
    ).eval()
    tts = model.tts
    model_lock = threading.Lock()

    def infer(text: str) -> tuple[np.ndarray, int, float, float]:
        queued_at = time.perf_counter()
        with model_lock:
            started = time.perf_counter()
            waveform, sample_rate, _ = tts.infer(
                ref_file=str(arguments.reference_audio),
                ref_text=REFERENCE_TEXT,
                gen_text=text,
                tokenizer="cls",
                cls_language="hindi",
                nfe_step=32,
                speed=1.0,
                cfg_strength=2.0,
                seed=request_seed(text),
                remove_silence=False,
                show_info=lambda *_args, **_kwargs: None,
                progress=None,
            )
            torch.cuda.synchronize()
            model_seconds = time.perf_counter() - started
        return np.asarray(waveform, dtype=np.float32), int(sample_rate), time.perf_counter() - queued_at, model_seconds

    for _ in range(arguments.warmup):
        infer(HINDI_TARGETS[0])

    def run_one(index: int, text: str) -> dict[str, object]:
        waveform, sample_rate, client_seconds, model_seconds = infer(text)
        result: dict[str, object] = {
            "index": index,
            "target_text": text,
            "seed": request_seed(text),
            "client_latency_ms": client_seconds * 1000.0,
            "model_compute_ms": model_seconds * 1000.0,
            "sample_rate": sample_rate,
            "audio_seconds": waveform.size / sample_rate,
            **inspect_waveform(waveform),
        }
        sf.write(arguments.output_dir / f"{index:02d}.wav", waveform, sample_rate, subtype="PCM_16")
        return result

    benchmark_started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=arguments.concurrency) as executor:
        results = [future.result() for future in [
            executor.submit(run_one, index, text) for index, text in enumerate(HINDI_TARGETS)
        ]]
    wall_seconds = time.perf_counter() - benchmark_started
    results.sort(key=lambda item: int(item["index"]))
    with (arguments.output_dir / "requests.jsonl").open("w", encoding="utf-8") as handle:
        for item in results:
            handle.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")

    latencies = [float(item["client_latency_ms"]) for item in results]
    model_seconds = sum(float(item["model_compute_ms"]) for item in results) / 1000.0
    end_to_end_seconds = sum(float(item["client_latency_ms"]) for item in results) / 1000.0
    audio_seconds = sum(float(item["audio_seconds"]) for item in results)
    passed = sum(bool(item["passed"]) for item in results)
    summary = {
        **workload_metadata(),
        "backend": "hf_eager_fp32",
        "concurrency": arguments.concurrency,
        "model_boundary": "serialized_single_eager_model",
        "warmup_requests_excluded": arguments.warmup,
        "wall_seconds": wall_seconds,
        "throughput_requests_per_second": len(results) / wall_seconds,
        "mean_latency_ms": statistics.fmean(latencies),
        "p50_latency_ms": percentile(latencies, 0.5),
        "p90_latency_ms": percentile(latencies, 0.9),
        "p95_latency_ms": percentile(latencies, 0.95),
        "processing_rtf": model_seconds / audio_seconds if audio_seconds else math.nan,
        "end_to_end_rtf": end_to_end_seconds / audio_seconds if audio_seconds else math.nan,
        "timing_scope": "hf_eager_model_and_e2e",
        "audio_seconds_total": audio_seconds,
        "quality": {"passed": passed, "total": len(results), "rms_floor": QUALITY_RMS_FLOOR},
    }
    with (arguments.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
    if passed != len(results):
        raise SystemExit("quality gate failed")


if __name__ == "__main__":
    main()
