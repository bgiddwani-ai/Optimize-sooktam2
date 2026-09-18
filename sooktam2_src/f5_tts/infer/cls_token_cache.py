"""Exact, thread-safe cache for the Sooktam2 CLS front end.

The CLS tokenizer works word by word and emits a single ``" "`` token between
words.  A TTS request always supplies a fixed prompt transcript followed by
the new target transcript.  Caching the prompt token prefix therefore avoids
re-running ``indic_unified_parser.wordparse`` over the prompt for every
request, while preserving precisely the same tokens as tokenizing the joined
string.

The cache deliberately stores token *strings*, rather than IDs or embeddings:
the vocabulary lookup, model precision, and DiT inputs are unchanged.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock

from f5_tts.infer.cls_tokenizer_v2 import cls_tokenize_text


@dataclass(frozen=True)
class CLSCacheStats:
    """A snapshot suitable for a health endpoint or benchmark evidence."""

    hits: int
    misses: int
    entries: int


class CLSReferencePrefixCache:
    """Bounded cache for a known CLS prompt prefix.

    ``prime_reference`` receives the complete prefix as it will appear before
    target text (normally ``preprocessed_ref_text + " "``).  ``tokenize`` is
    a drop-in ``cls_tokenizer_fn`` for ``F5TTS.infer``: it safely falls back to
    the upstream tokenizer when no cached prefix matches.
    """

    def __init__(self, max_entries: int = 64) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        self.max_entries = max_entries
        self._prefixes: OrderedDict[tuple[str, str], tuple[str, ...]] = OrderedDict()
        self._lock = Lock()
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _key(prefix: str, language: str) -> tuple[str, str]:
        return language.strip().lower(), prefix

    def prime_reference(self, prefix: str, language: str) -> None:
        """Cache tokens for a prompt prefix exactly as it precedes target text."""

        if not prefix:
            raise ValueError("CLS reference prefix must not be empty")
        key = self._key(prefix, language)
        with self._lock:
            if key in self._prefixes:
                self._prefixes.move_to_end(key)
                return
        # Whitespace is ignored by the upstream tokenizer, so a separator is
        # represented explicitly when the cached prefix ends in whitespace.
        tokens = tuple(cls_tokenize_text(prefix, key[0]))
        if not tokens:
            raise RuntimeError("CLS reference prefix produced no tokens")
        with self._lock:
            self._prefixes[key] = tokens
            self._prefixes.move_to_end(key)
            while len(self._prefixes) > self.max_entries:
                self._prefixes.popitem(last=False)

    def tokenize(self, text: str, language: str) -> list[str]:
        """Return upstream-equivalent tokens, using the longest cached prefix."""

        normalized_language = language.strip().lower()
        with self._lock:
            matches = [
                (prefix, tokens)
                for (cached_language, prefix), tokens in self._prefixes.items()
                if cached_language == normalized_language and text.startswith(prefix)
            ]
        if not matches:
            tokens = cls_tokenize_text(text, normalized_language)
            with self._lock:
                self._misses += 1
            return tokens

        prefix, reference_tokens = max(matches, key=lambda item: len(item[0]))
        target_text = text[len(prefix) :]
        target_tokens = cls_tokenize_text(target_text, normalized_language) if target_text else []
        # ``get_cls_token_list`` inserts one space between words.  Add it only
        # when the original joined string contains an inter-text whitespace
        # boundary.  The no-whitespace form supports the legacy Triton BLS
        # request contract and is parity-checked before it is enabled.
        separator = [" "] if target_tokens and prefix[-1].isspace() else []
        merged = [*reference_tokens, *separator, *target_tokens]
        with self._lock:
            self._hits += 1
        return merged

    def tokenize_reference_target(self, reference_prefix: str, target_text: str, language: str) -> list[str]:
        """Prime and tokenize an explicit prompt/target pair for Triton BLS."""

        self.prime_reference(reference_prefix, language)
        return self.tokenize(reference_prefix + target_text, language)

    def stats(self) -> CLSCacheStats:
        with self._lock:
            return CLSCacheStats(self._hits, self._misses, len(self._prefixes))


def assert_reference_prefix_token_parity(
    cache: CLSReferencePrefixCache,
    reference_prefix: str,
    target_texts: list[str] | tuple[str, ...],
    language: str,
) -> None:
    """Fail closed if cached concatenation diverges from the upstream CLS path."""

    cache.prime_reference(reference_prefix, language)
    for target_text in target_texts:
        upstream = cls_tokenize_text(reference_prefix + target_text, language)
        cached = cache.tokenize(reference_prefix + target_text, language)
        if cached != upstream:
            raise RuntimeError(
                "CLS prefix cache token parity failed; refusing to serve a changed token sequence"
            )
