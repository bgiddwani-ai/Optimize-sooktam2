#!/usr/bin/env python3
"""Produce the one comparable final Hindi8 benchmark table from run summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


RUNS = (
    ("HF eager, FP32", "hf_eager_fp32_c1", 1),
    ("HF eager, FP32", "hf_eager_fp32_c2", 2),
    ("TRT-LLM FP32 DiT + TRT Vocos + Triton", "triton_fp32_c1", 1),
    ("TRT-LLM FP32 DiT + TRT Vocos + Triton", "triton_fp32_c2", 2),
    ("AOTI BF16 DiT + server", "aoti_bf16_c1", 1),
    ("AOTI BF16 DiT + server", "aoti_bf16_c2", 2),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--gpu", default="NVIDIA A100 80GB PCIe")
    arguments = parser.parse_args()

    rows = []
    for mode, directory, concurrency in RUNS:
        summary_path = arguments.root / directory / "summary.json"
        with summary_path.open(encoding="utf-8") as handle:
            summary = json.load(handle)
        quality = summary["quality"]
        if quality["passed"] != quality["total"]:
            raise RuntimeError(f"quality gate failed: {summary_path}")
        # This is the system-level RTF comparable at either concurrency: timed
        # wall time divided by the generated audio duration.  Per-request mean
        # latency remains separately reported and includes queueing at C=2.
        system_e2e_rtf = summary["wall_seconds"] / summary["audio_seconds_total"]
        rows.append(
            {
                "mode": mode,
                "concurrency": concurrency,
                "system_e2e_rtf": system_e2e_rtf,
                "throughput_requests_per_second": summary["throughput_requests_per_second"],
                "mean_latency_ms": summary["mean_latency_ms"],
                "p95_latency_ms": summary["p95_latency_ms"],
                "audio_seconds_total": summary["audio_seconds_total"],
                "quality": f"Pass: {quality['passed']}/{quality['total']}",
                "source": str(summary_path),
            }
        )

    report = {
        "gpu": arguments.gpu,
        "workload": "model-bundled Hindi prompt audio plus 8 fixed varied-length Hindi targets; CLS; NFE=32",
        "warmup_requests_excluded_per_run": 1,
        "rtf_definition": "timed run wall seconds divided by generated audio seconds",
        "latency_definition": "per-request end-to-end latency; includes queueing at C=2",
        "rows": rows,
    }
    with (arguments.root / "final_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    lines = [
        "# Sooktam2 Hindi8 benchmark (NFE=32)",
        "",
        f"GPU: {arguments.gpu}",
        "",
        "RTF is timed run wall seconds / generated audio seconds. One warmup request is excluded; C=2 mean latency includes queueing.",
        "",
        "| Mode | Concurrency | E2E RTF | Throughput | Mean latency | P95 latency | Quality |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['mode']} | {row['concurrency']} | {row['system_e2e_rtf']:.4f} | "
            f"{row['throughput_requests_per_second']:.3f} req/s | "
            f"{row['mean_latency_ms'] / 1000:.3f} s | {row['p95_latency_ms'] / 1000:.3f} s | {row['quality']} |"
        )
    (arguments.root / "FINAL.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
