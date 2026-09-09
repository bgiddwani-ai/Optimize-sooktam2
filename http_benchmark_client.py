#!/usr/bin/env python3
"""Quality-gated eight-sentence HTTP benchmark for Triton or the AOTI server."""

from __future__ import annotations

import argparse
import concurrent.futures
import io
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests
import soundfile as sf

from hindi8_workload import HINDI_TARGETS, REFERENCE_TEXT, request_seed, workload_metadata


QUALITY_RMS_FLOOR = 1.0e-3


def percentile(values: list[float], fraction: float) -> float:
    values = sorted(values)
    if not values:
        return float("nan")
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


class Client:
    def __init__(self, backend: str, endpoint: str, reference_audio: Path) -> None:
        self.backend = backend
        self.endpoint = endpoint.rstrip("/")
        waveform, sample_rate = sf.read(reference_audio, dtype="float32", always_2d=False)
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
        self.reference_waveform = np.asarray(waveform, dtype=np.float32)
        self.reference_sample_rate = int(sample_rate)
        self.reference_audio = reference_audio

    def infer(self, target_text: str) -> tuple[np.ndarray, int, float, float]:
        """Return waveform, sample rate, end-to-end seconds, model seconds."""

        seed = request_seed(target_text)
        started = time.perf_counter()
        if self.backend == "triton":
            body: dict[str, Any] = {
                "inputs": [
                    {
                        "name": "reference_wav",
                        "shape": [1, int(self.reference_waveform.size)],
                        "datatype": "FP32",
                        "data": self.reference_waveform[None].tolist(),
                    },
                    {
                        "name": "reference_wav_len",
                        "shape": [1, 1],
                        "datatype": "INT32",
                        "data": [[int(self.reference_waveform.size)]],
                    },
                    {"name": "reference_text", "shape": [1, 1], "datatype": "BYTES", "data": [[REFERENCE_TEXT]]},
                    {"name": "target_text", "shape": [1, 1], "datatype": "BYTES", "data": [[target_text]]},
                ],
                # The deployed BLS model intentionally exposes only waveform.
                # End-to-end client timing is therefore the comparable timing
                # boundary for every backend in this report.
                "outputs": [{"name": "waveform"}],
            }
            response = requests.post(
                self.endpoint + "/v2/models/f5_tts/infer", json=body, timeout=900
            )
            elapsed = time.perf_counter() - started
            if not response.ok:
                raise RuntimeError(f"Triton HTTP {response.status_code}: {response.text}")
            outputs = {entry["name"]: entry for entry in response.json()["outputs"]}
            waveform = np.asarray(outputs["waveform"]["data"], dtype=np.float32).reshape(-1)
            model_seconds = elapsed
            sample_rate = 24000
        elif self.backend == "aoti":
            with self.reference_audio.open("rb") as handle:
                response = requests.post(
                    self.endpoint + "/v1/infer",
                    files={"reference_audio": (self.reference_audio.name, handle, "audio/wav")},
                    data={
                        "reference_text": REFERENCE_TEXT,
                        "target_text": target_text,
                        "seed": str(seed),
                    },
                    timeout=900,
                )
            elapsed = time.perf_counter() - started
            response.raise_for_status()
            waveform, sample_rate = sf.read(io.BytesIO(response.content), dtype="float32", always_2d=False)
            if waveform.ndim == 2:
                waveform = waveform.mean(axis=1)
            waveform = np.asarray(waveform, dtype=np.float32)
            model_seconds = float(response.headers["X-Sooktam-Server-Latency-Ms"]) / 1000.0
        else:
            raise ValueError(f"unsupported backend: {self.backend}")
        return waveform, int(sample_rate), elapsed, model_seconds


def quality(waveform: np.ndarray) -> dict[str, object]:
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
    parser.add_argument("--backend", choices=("triton", "aoti"), required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--reference-audio", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, choices=(1, 2), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=1)
    arguments = parser.parse_args()

    arguments.output_dir.mkdir(parents=True, exist_ok=False)
    client = Client(arguments.backend, arguments.endpoint, arguments.reference_audio)

    # Warmup is deliberately outside the timed/quality-gated eight requests.
    for _ in range(arguments.warmup):
        client.infer(HINDI_TARGETS[0])

    def run_one(index: int, text: str) -> dict[str, object]:
        waveform, sample_rate, elapsed, model_seconds = client.infer(text)
        audio_seconds = waveform.size / sample_rate
        result: dict[str, object] = {
            "index": index,
            "target_text": text,
            "seed": request_seed(text),
            "client_latency_ms": elapsed * 1000.0,
            "model_compute_ms": model_seconds * 1000.0,
            "sample_rate": sample_rate,
            "audio_seconds": audio_seconds,
            **quality(waveform),
        }
        sf.write(arguments.output_dir / f"{index:02d}.wav", waveform, sample_rate, subtype="PCM_16")
        return result

    benchmark_started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=arguments.concurrency) as executor:
        futures = [executor.submit(run_one, index, text) for index, text in enumerate(HINDI_TARGETS)]
        results = [future.result() for future in futures]
    wall_seconds = time.perf_counter() - benchmark_started
    results.sort(key=lambda item: int(item["index"]))
    with (arguments.output_dir / "requests.jsonl").open("w", encoding="utf-8") as handle:
        for item in results:
            handle.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")

    latencies = [float(item["client_latency_ms"]) for item in results]
    model_seconds = sum(float(item["model_compute_ms"]) for item in results) / 1000.0
    end_to_end_seconds = sum(float(item["client_latency_ms"]) for item in results) / 1000.0
    audio_seconds = sum(float(item["audio_seconds"]) for item in results)
    passing = sum(bool(item["passed"]) for item in results)
    summary = {
        **workload_metadata(),
        "backend": arguments.backend,
        "endpoint": arguments.endpoint,
        "concurrency": arguments.concurrency,
        "warmup_requests_excluded": arguments.warmup,
        "wall_seconds": wall_seconds,
        "throughput_requests_per_second": len(results) / wall_seconds,
        "mean_latency_ms": statistics.fmean(latencies),
        "p50_latency_ms": percentile(latencies, 0.50),
        "p90_latency_ms": percentile(latencies, 0.90),
        "p95_latency_ms": percentile(latencies, 0.95),
        "processing_rtf": model_seconds / audio_seconds if audio_seconds else math.nan,
        "end_to_end_rtf": end_to_end_seconds / audio_seconds if audio_seconds else math.nan,
        "timing_scope": "triton_e2e_request" if arguments.backend == "triton" else "aoti_server_and_e2e",
        "audio_seconds_total": audio_seconds,
        "quality": {"passed": passing, "total": len(results), "rms_floor": QUALITY_RMS_FLOOR},
    }
    with (arguments.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
    if passing != len(results):
        raise SystemExit("quality gate failed")


if __name__ == "__main__":
    main()
