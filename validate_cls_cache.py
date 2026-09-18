#!/usr/bin/env python3
"""Validate exact CLS cache parity for the Hindi-8 benchmark workload."""

from __future__ import annotations

import json
import time

from f5_tts.infer.cls_token_cache import (
    CLSReferencePrefixCache,
    assert_reference_prefix_token_parity,
)
from f5_tts.infer.cls_tokenizer_v2 import cls_tokenize_text
from hindi8_workload import HINDI_TARGETS, REFERENCE_TEXT


def aoti_prefix(text: str) -> str:
    if not text.endswith(". ") and not text.endswith("。"):
        text = text + (" " if text.endswith(".") else ". ")
    return text + " " if len(text[-1].encode("utf-8")) == 1 else text


def elapsed_ms(function) -> float:
    started = time.perf_counter()
    function()
    return (time.perf_counter() - started) * 1000.0


def main() -> None:
    results: dict[str, object] = {}
    for label, prefix in {
        "aoti_api_join": aoti_prefix(REFERENCE_TEXT),
        "triton_legacy_join": REFERENCE_TEXT,
    }.items():
        cache = CLSReferencePrefixCache()
        assert_reference_prefix_token_parity(cache, prefix, HINDI_TARGETS, "hindi")
        upstream_ms = elapsed_ms(
            lambda: [cls_tokenize_text(prefix + target, "hindi") for target in HINDI_TARGETS]
        )
        cached_ms = elapsed_ms(
            lambda: [cache.tokenize(prefix + target, "hindi") for target in HINDI_TARGETS]
        )
        results[label] = {
            "token_parity": True,
            "upstream_ms": round(upstream_ms, 3),
            "cached_ms": round(cached_ms, 3),
            "speedup": round(upstream_ms / cached_ms, 3) if cached_ms else None,
            "cache": vars(cache.stats()),
        }
    print(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
