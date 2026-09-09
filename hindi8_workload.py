"""Reproducible, non-FLEURS Hindi workload used by every benchmark path."""

from __future__ import annotations

import hashlib
from pathlib import Path


# This is the prompt audio and transcript shipped in the Sooktam2 repository's
# own ``test.py``.  Keeping it fixed makes C=1/C=2 comparisons meaningful while
# avoiding any FLEURS-derived audio or transcripts.
REFERENCE_AUDIO = Path("/workspace/src/sooktam2/ref.wav")
REFERENCE_TEXT = (
    "सर, मैं तब से यह कह रहा हूँ कि मैंने अपना टिकट कैंसल कर दिया है, "
    "लेकिन अब तक मेरे पैसे वापस नहीं आए हैं। आप इस मामले को देखेंगे भी या नहीं?"
)

# Eight fixed sentences with deliberately varied lengths.  They are fixed (not
# freshly randomized on each run) so a result can be reproduced exactly.
HINDI_TARGETS = (
    "नमस्ते।",
    "आज दिल्ली में हल्की बारिश हुई।",
    "कृपया बैठक शुरू होने से पहले रिपोर्ट ईमेल कर दीजिए।",
    "मुझे स्टेशन तक पहुँचने के लिए सबसे तेज़ और सुरक्षित रास्ता बताइए।",
    "हमारी टीम अगले सप्ताह नए उत्पाद का परीक्षण करेगी।",
    "अगर मौसम साफ रहा, तो हम रविवार को परिवार के साथ पार्क में पिकनिक मनाएँगे।",
    "किसान ने बताया कि समय पर सिंचाई, सही बीज और मिट्टी की जाँच से फसल की गुणवत्ता बेहतर होती है।",
    "सरकार ने कहा कि छोटे व्यवसायों को डिजिटल भुगतान अपनाने, ग्राहकों तक बेहतर सेवा पहुँचाने और अपने काम को पारदर्शी रखने में मदद दी जाएगी।",
)


def request_seed(target_text: str) -> int:
    """Stable per-sentence sampling seed shared by eager, Triton, and AOTI."""

    digest = hashlib.blake2b(
        f"{REFERENCE_TEXT}\0{target_text}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "little") & ((1 << 63) - 1)


def workload_metadata() -> dict[str, object]:
    return {
        "dataset": "fixed_hindi8_non_fleurs",
        "target_count": len(HINDI_TARGETS),
        "target_characters": [len(item) for item in HINDI_TARGETS],
        "reference_audio": str(REFERENCE_AUDIO),
        "tokenizer": "cls",
        "cls_language": "hindi",
        "nfe_steps": 32,
    }
