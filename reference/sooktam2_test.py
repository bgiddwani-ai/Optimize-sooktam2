import os
from transformers import AutoModel

# --- Paths / model id (adjust if needed) ---
REPO_DIR = "."
MODEL_ID =  "bharatgenai/sooktam2"

REF_AUDIO = "./ref.wav"
REF_TEXT = "सर, मैं तब से यह कह रहा हूँ कि मैंने अपना टिकट कैंसल कर दिया है, लेकिन अब तक मेरे पैसे वापस नहीं आए हैं। आप इस मामले को देखेंगे भी या नहीं?"
GEN_TEXT = "यह एक टेस्ट वाक्य है जिसे आवाज़ में बदलना है।"

OUT_DIR = os.path.join(REPO_DIR, "outputs")
OUT_WAV = os.path.join(OUT_DIR, "sooktam_cls.wav")

# CLS tokenization is handled inside utils_infer via cls_tokenizer_v2.

# --- Load TTS model via AutoModel (auto-download ckpt + vocab from HF) ---
model = AutoModel.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)

os.makedirs(OUT_DIR, exist_ok=True)

wav, sr, _ = model.infer(
    ref_file=REF_AUDIO,
    ref_text=REF_TEXT,
    gen_text=GEN_TEXT,
    tokenizer="cls",
    cls_language="hindi",
    file_wave=OUT_WAV,
)

print("Saved:", OUT_WAV, "sample_rate:", sr, "samples:", len(wav))
