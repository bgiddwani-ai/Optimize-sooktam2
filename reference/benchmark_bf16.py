"""
BF16 / TF32 optimization benchmark for Sooktam2 on RTX Pro 6000.

Tests three progressive configs beyond the FP32 baseline:
  A. FP32 + TF32  — enable allow_tf32, model stays FP32
  B. BF16 autocast — torch.autocast(bfloat16) around infer(), no compile
  C. BF16 autocast + fullgraph compile — forward_compiled hoists text_embed+rope
     out of ODE loop and compiles the inner DiT with fullgraph=True

RTF = wall_time / audio_duration  (lower = better; <1.0 = faster than realtime)
"""
import os
import time
import json
import glob
import torch
import statistics
import numpy as np
import soundfile as sf
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Optional

# Monkey-patch torchaudio.load to use soundfile (avoids torchcodec dep in torchaudio 2.11)
import torchaudio as _torchaudio
import soundfile as _sf
def _sf_load(filepath, *args, **kwargs):
    data, sr = _sf.read(str(filepath), dtype='float32', always_2d=True)
    return torch.from_numpy(data.T), sr
_torchaudio.load = _sf_load

# ── TF32: big free win on Blackwell/Ampere+ — tensor cores for FP32 matmuls ──
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ── Find checkpoint ──────────────────────────────────────────────────────────
SNAP_GLOB = "/ephemeral/cache/huggingface/hub/models--bharatgenai--sooktam2/snapshots/*/sooktam.safetensors"
snaps = glob.glob(SNAP_GLOB)
if not snaps:
    snaps = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--bharatgenai--sooktam2/snapshots/*/sooktam.safetensors"
    ))
if not snaps:
    from huggingface_hub import snapshot_download
    snap_dir = snapshot_download("bharatgenai/sooktam2", ignore_patterns=["model_1250000.pt"])
    CKPT = os.path.join(snap_dir, "sooktam.safetensors")
    VOCAB = os.path.join(snap_dir, "vocab.txt")
else:
    CKPT = snaps[0]
    VOCAB = os.path.join(os.path.dirname(CKPT), "vocab.txt")

print(f"Checkpoint: {CKPT}")

# ── Monkey-patch DiT.forward_compiled ────────────────────────────────────────
# Adds a stateless forward method with pre-computed text_embed and rope passed in.
# This enables fullgraph=True compilation because the text_embed cache assignment
# (self.text_cond = ...) no longer appears inside the compiled region.

from f5_tts.model.backbones.dit import DiT

def _forward_compiled(
    self,
    x,                  # [b, n, mel_dim]  noisy input
    cond,               # [b, n, mel_dim]  conditioning
    text_embed_cond,    # [b, n, text_dim]  pre-computed, drop_text=False
    text_embed_uncond,  # [b, n, text_dim]  pre-computed, drop_text=True
    time,               # scalar or [b]
    rope,               # (freqs, xpos_scale) from rotary_embed.forward_from_seq_len
    mask=None,          # [b, n] bool or None
    cfg_infer=True,     # True = pack cond+uncond in batch dim (standard inference)
):
    batch = x.shape[0]
    if time.ndim == 0:
        time = time.repeat(batch)
    t = self.time_embed(time)
    if cfg_infer:
        x_cond   = self.input_embed(x, cond, text_embed_cond,   drop_audio_cond=False)
        x_uncond = self.input_embed(x, cond, text_embed_uncond, drop_audio_cond=True)
        x = torch.cat((x_cond, x_uncond), dim=0)
        t = torch.cat((t, t), dim=0)
        if mask is not None:
            mask = torch.cat((mask, mask), dim=0)
    else:
        x = self.input_embed(x, cond, text_embed_cond, drop_audio_cond=False)

    if self.long_skip_connection is not None:
        residual = x

    for block in self.transformer_blocks:
        x = block(x, t, mask=mask, rope=rope)

    if self.long_skip_connection is not None:
        x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

    x = self.norm_out(x, t)
    return self.proj_out(x)

DiT.forward_compiled = _forward_compiled

# ── Monkey-patch CFM.sample to use forward_compiled ──────────────────────────
# Hoists text_embed and rope computation out of the ODE loop.
# The fn(t, x) callback then calls transformer.forward_compiled instead of
# transformer(..., cache=True), removing the cache-check Python overhead and
# enabling the compiled path when F5_COMPILE=1.

from f5_tts.model.cfm import CFM
from torch.nn.utils.rnn import pad_sequence
from f5_tts.model.utils import (
    default, exists, get_epss_timesteps, lens_to_mask,
    list_str_to_idx, list_str_to_tensor, mask_from_frac_lengths,
)
import torch.nn.functional as F
from torchdiffeq import odeint

@torch.no_grad()
def _sample_patched(
    self,
    cond,
    text,
    duration,
    *,
    lens=None,
    steps=32,
    cfg_strength=1.0,
    sway_sampling_coef=None,
    seed=None,
    max_duration=4096,
    vocoder=None,
    use_epss=True,
    no_ref_audio=False,
    duplicate_test=False,
    t_inter=0.1,
    edit_mask=None,
):
    self.eval()

    if cond.ndim == 2:
        cond = self.mel_spec(cond)
        cond = cond.permute(0, 2, 1)
        assert cond.shape[-1] == self.num_channels

    cond = cond.to(next(self.parameters()).dtype)

    batch, cond_seq_len, device = *cond.shape[:2], cond.device
    if not exists(lens):
        lens = torch.full((batch,), cond_seq_len, device=device, dtype=torch.long)

    if isinstance(text, list):
        if exists(self.vocab_char_map):
            text = list_str_to_idx(text, self.vocab_char_map).to(device)
        else:
            text = list_str_to_tensor(text).to(device)
        assert text.shape[0] == batch

    cond_mask = lens_to_mask(lens)
    if edit_mask is not None:
        cond_mask = cond_mask & edit_mask

    if isinstance(duration, int):
        duration = torch.full((batch,), duration, device=device, dtype=torch.long)

    duration = torch.maximum(
        torch.maximum((text != -1).sum(dim=-1), lens) + 1, duration
    )
    duration = duration.clamp(max=max_duration)
    max_duration = duration.amax()

    if duplicate_test:
        test_cond = F.pad(cond, (0, 0, cond_seq_len, max_duration - 2 * cond_seq_len), value=0.0)

    cond = F.pad(cond, (0, 0, 0, max_duration - cond_seq_len), value=0.0)
    if no_ref_audio:
        cond = torch.zeros_like(cond)

    cond_mask = F.pad(cond_mask, (0, max_duration - cond_mask.shape[-1]), value=False)
    cond_mask = cond_mask.unsqueeze(-1)
    step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))

    if batch > 1:
        mask = lens_to_mask(duration)
    else:
        mask = None

    # ── Hoist text_embed and rope outside the ODE loop ───────────────────────
    # Original code recomputed these inside fn() on every ODE step (or cached
    # via self.text_cond, which blocks fullgraph compilation).
    seq_len = step_cond.shape[1]
    _text_embed_cond   = self.transformer.text_embed(text, seq_len, drop_text=False, audio_mask=None)
    _text_embed_uncond = self.transformer.text_embed(text, seq_len, drop_text=True,  audio_mask=None)
    _rope = self.transformer.rotary_embed.forward_from_seq_len(seq_len)

    def fn(t, x):
        if cfg_strength < 1e-5:
            pred = self.transformer.forward_compiled(
                x=x, cond=step_cond,
                text_embed_cond=_text_embed_cond,
                text_embed_uncond=_text_embed_cond,
                time=t, rope=_rope, mask=mask, cfg_infer=False,
            )
            return pred
        pred_cfg = self.transformer.forward_compiled(
            x=x, cond=step_cond,
            text_embed_cond=_text_embed_cond,
            text_embed_uncond=_text_embed_uncond,
            time=t, rope=_rope, mask=mask, cfg_infer=True,
        )
        pred, null_pred = torch.chunk(pred_cfg, 2, dim=0)
        return pred + (pred - null_pred) * cfg_strength

    y0 = []
    for dur in duration:
        if exists(seed):
            torch.manual_seed(seed)
        y0.append(torch.randn(dur, self.num_channels, device=self.device, dtype=step_cond.dtype))
    y0 = pad_sequence(y0, padding_value=0, batch_first=True)

    t_start = 0

    if duplicate_test:
        t_start = t_inter
        y0 = (1 - t_start) * y0 + t_start * test_cond
        steps = int(steps * (1 - t_start))

    if t_start == 0 and use_epss:
        t = get_epss_timesteps(steps, device=self.device, dtype=step_cond.dtype)
    else:
        t = torch.linspace(t_start, 1, steps + 1, device=self.device, dtype=step_cond.dtype)
    if sway_sampling_coef is not None:
        t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

    trajectory = odeint(fn, y0, t, **self.odeint_kwargs)
    # No clear_cache() needed — we don't set self.text_cond/text_uncond

    sampled = trajectory[-1]
    out = sampled
    out = torch.where(cond_mask, cond, out)

    if exists(vocoder):
        out = out.permute(0, 2, 1)
        out = vocoder(out)

    return out, trajectory

CFM.sample = _sample_patched

# ── Load model in FP32 ───────────────────────────────────────────────────────
from f5_tts.api import F5TTS

NFE = int(os.environ.get("NFE_STEPS", "16"))
print(f"\nLoading F5TTS model (FP32, no compile initially, NFE={NFE})...")
t0 = time.perf_counter()
tts = F5TTS(
    model="F5TTS_v1_Base",
    ckpt_file=CKPT,
    vocab_file=VOCAB,
    ode_method="euler",
    use_ema=True,
    device="cuda",
)
# Model stays in FP32 — TF32 uses tensor cores, autocast handles BF16
load_time = time.perf_counter() - t0
print(f"Model loaded in {load_time:.1f}s  (FP32, eager)")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")

REF_WAV = "/home/shadeform/sooktam/benchmark/ref.wav"
REF_TEXT = "यह संदर्भ ऑडियो है।"

TEXTS = [
    "नमस्ते।",
    "आज मौसम अच्छा है।",
    "मुझे खाना पसंद है।",
    "कृत्रिम बुद्धिमत्ता भारतीय भाषाओं के विकास में महत्वपूर्ण भूमिका निभा रही है।",
    "प्रौद्योगिकी की मदद से हम एक बेहतर समाज बना सकते हैं जहाँ सभी को समान अवसर मिलें।",
    "भारत एक विविधताओं से भरा देश है जहाँ अनेक भाषाएँ और संस्कृतियाँ एक साथ फलती-फूलती हैं।",
    "विज्ञान और तकनीक के क्षेत्र में भारत ने अभूतपूर्व प्रगति की है। हमारे वैज्ञानिकों ने अंतरिक्ष अनुसंधान, परमाणु ऊर्जा और सूचना प्रौद्योगिकी में विश्व स्तर पर अपनी पहचान बनाई है। आने वाले वर्षों में भारत और भी आगे बढ़ेगा।",
]


@dataclass
class Result:
    text_len: int
    wall_s: float
    audio_dur_s: float
    rtf: float


@dataclass
class Report:
    label: str
    n: int
    nfe: int
    dtype: str
    compiled: bool
    results: List[Result] = field(default_factory=list)
    latency_mean_ms: float = 0
    latency_p50_ms: float = 0
    latency_p90_ms: float = 0
    latency_p95_ms: float = 0
    latency_p99_ms: float = 0
    latency_min_ms: float = 0
    latency_max_ms: float = 0
    rtf_mean: float = 0
    rtf_p50: float = 0
    rtf_p90: float = 0
    rps: float = 0
    audio_s_per_s: float = 0
    total_wall_s: float = 0


def pct(lst, p):
    if not lst:
        return 0.0
    s = sorted(lst)
    return s[min(int(len(s)*p/100), len(s)-1)]


def print_report(r: Report):
    print(f"\n{'='*60}")
    print(f"  {r.label}")
    print(f"  NFE={r.nfe}  dtype={r.dtype}  compiled={r.compiled}  N={r.n}")
    print(f"  total_wall={r.total_wall_s:.1f}s")
    print(f"{'─'*60}")
    print(f"  Latency (wall-clock):")
    print(f"    mean  = {r.latency_mean_ms:.0f} ms")
    print(f"    p50   = {r.latency_p50_ms:.0f} ms")
    print(f"    p90   = {r.latency_p90_ms:.0f} ms")
    print(f"    p95   = {r.latency_p95_ms:.0f} ms")
    print(f"    p99   = {r.latency_p99_ms:.0f} ms")
    print(f"    min   = {r.latency_min_ms:.0f} ms  max = {r.latency_max_ms:.0f} ms")
    print(f"  RTF = wall_time / audio_duration  (LOWER IS BETTER):")
    print(f"    mean  = {r.rtf_mean:.4f}")
    print(f"    p50   = {r.rtf_p50:.4f}")
    print(f"    p90   = {r.rtf_p90:.4f}  (worst 10%)")
    print(f"  Throughput:")
    print(f"    RPS   = {r.rps:.2f} req/s")
    print(f"    audio = {r.audio_s_per_s:.2f} audio-s/wall-s")
    print(f"{'='*60}")


def run_tts(text: str, nfe: int, use_autocast: bool) -> tuple:
    t0 = time.perf_counter()
    if use_autocast:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            wav, sr, _ = tts.infer(
                ref_file=REF_WAV, ref_text=REF_TEXT, gen_text=text,
                nfe_step=nfe, speed=1.0, cfg_strength=2.0, remove_silence=False,
            )
    else:
        wav, sr, _ = tts.infer(
            ref_file=REF_WAV, ref_text=REF_TEXT, gen_text=text,
            nfe_step=nfe, speed=1.0, cfg_strength=2.0, remove_silence=False,
        )
    wall = time.perf_counter() - t0
    return wav, sr, wall


def run_sequential(n: int, nfe: int, label: str, dtype: str, compiled: bool, use_autocast: bool) -> Report:
    results = []
    t_total = time.perf_counter()
    for i in range(n):
        text = TEXTS[i % len(TEXTS)]
        wav, sr, wall = run_tts(text, nfe, use_autocast)
        dur = len(wav) / sr
        rtf = wall / dur
        results.append(Result(text_len=len(text), wall_s=wall, audio_dur_s=dur, rtf=rtf))
        print(f"  [{i+1}/{n}] wall={wall*1000:.0f}ms  audio={dur:.2f}s  RTF={rtf:.4f}")
    total_wall = time.perf_counter() - t_total

    walls = [r.wall_s for r in results]
    rtfs  = [r.rtf for r in results]
    rep = Report(label=label, n=len(results), nfe=nfe, dtype=dtype, compiled=compiled,
                 results=results, total_wall_s=total_wall)
    rep.latency_mean_ms = statistics.mean(walls)*1000
    rep.latency_p50_ms  = pct(walls, 50)*1000
    rep.latency_p90_ms  = pct(walls, 90)*1000
    rep.latency_p95_ms  = pct(walls, 95)*1000
    rep.latency_p99_ms  = pct(walls, 99)*1000
    rep.latency_min_ms  = min(walls)*1000
    rep.latency_max_ms  = max(walls)*1000
    rep.rtf_mean        = statistics.mean(rtfs)
    rep.rtf_p50         = pct(rtfs, 50)
    rep.rtf_p90         = pct(rtfs, 90)
    rep.rps             = len(results) / total_wall
    rep.audio_s_per_s   = sum(r.audio_dur_s for r in results) / total_wall
    return rep


all_reports = []


# ── CONFIG A: FP32 + TF32 ────────────────────────────────────────────────────
# TF32 already enabled globally above. Model is FP32. No autocast.
print(f"\n{'#'*60}")
print("  CONFIG A: FP32 + TF32 (warm-up 3 runs)")
print(f"{'#'*60}")
for i in range(3):
    wav, sr, wall = run_tts(TEXTS[i % len(TEXTS)], NFE, use_autocast=False)
    dur = len(wav)/sr
    print(f"  warmup {i+1}: wall={wall*1000:.0f}ms  RTF={wall/dur:.4f}")

print(f"\n{'#'*60}")
print(f"  CONFIG A: FP32 + TF32 (N=20, NFE={NFE})")
print(f"{'#'*60}")
rep = run_sequential(20, NFE, f"FP32_TF32_NFE{NFE}", "float32+tf32", False, False)
print_report(rep)
all_reports.append(rep)


# ── CONFIG B: BF16 autocast ───────────────────────────────────────────────────
# Same FP32 model, but inference runs under torch.autocast(bfloat16).
# Matmuls/attention use BF16 tensor cores; accumulation stays FP32.
print(f"\n{'#'*60}")
print("  CONFIG B: BF16 autocast (warm-up 3 runs)")
print(f"{'#'*60}")
for i in range(3):
    wav, sr, wall = run_tts(TEXTS[i % len(TEXTS)], NFE, use_autocast=True)
    dur = len(wav)/sr
    print(f"  warmup {i+1}: wall={wall*1000:.0f}ms  RTF={wall/dur:.4f}")

print(f"\n{'#'*60}")
print(f"  CONFIG B: BF16 autocast eager (N=20, NFE={NFE})")
print(f"{'#'*60}")
rep = run_sequential(20, NFE, f"BF16_autocast_NFE{NFE}", "bfloat16", False, True)
print_report(rep)
all_reports.append(rep)


# ── CONFIG C: BF16 autocast + fullgraph compile ───────────────────────────────
# Compile transformer.forward_compiled with fullgraph=True.
# The patched CFM.sample hoists text_embed+rope outside the ODE loop so the
# compiled region has no Python graph-breaks from cache assignment.
print(f"\n{'#'*60}")
print("  CONFIG C: BF16 autocast + fullgraph compile (compiling...)")
print(f"{'#'*60}")

dit = tts.ema_model.transformer
print("Compiling forward_compiled with fullgraph=True, max-autotune-no-cudagraphs...")
t_compile = time.perf_counter()
dit.forward_compiled = torch.compile(
    dit.forward_compiled.__get__(dit, type(dit)),  # bound method
    mode="max-autotune-no-cudagraphs",
    dynamic=True,
    fullgraph=True,
)
print(f"torch.compile setup done in {time.perf_counter()-t_compile:.1f}s (actual compilation happens on first call)")

print("  Warm-up 5 runs (triggers JIT compilation)...")
for i in range(5):
    wav, sr, wall = run_tts(TEXTS[i % len(TEXTS)], NFE, use_autocast=True)
    dur = len(wav)/sr
    print(f"  warmup {i+1}: wall={wall*1000:.0f}ms  RTF={wall/dur:.4f}")

print(f"\n{'#'*60}")
print(f"  CONFIG C: BF16 + fullgraph compile (N=20, NFE={NFE})")
print(f"{'#'*60}")
rep = run_sequential(20, NFE, f"BF16_fullgraph_compile_NFE{NFE}", "bfloat16", True, True)
print_report(rep)
all_reports.append(rep)


# ── NFE ablation on best config ───────────────────────────────────────────────
print(f"\n{'#'*60}")
print("  NFE ABLATION on BF16+compile (N=10 each)")
print(f"{'#'*60}")
for nfe in [8, 16, 32]:
    rep = run_sequential(10, nfe, f"BF16_compile_NFE{nfe}", "bfloat16", True, True)
    print_report(rep)
    all_reports.append(rep)


# ── Save results ──────────────────────────────────────────────────────────────
OUT_DIR = Path("/home/shadeform/sooktam/benchmark/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)
out_path = OUT_DIR / "rtx_pro_6000_bf16.json"

def serialize(obj):
    if isinstance(obj, (np.int64, np.float64)): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    return str(obj)

with open(out_path, "w") as f:
    json.dump([asdict(r) for r in all_reports], f, indent=2, default=serialize)

print(f"\n\nResults saved to {out_path}")

# ── Final summary table ───────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  FINAL COMPARISON — NFE={NFE}")
print(f"{'='*70}")
print(f"  {'Config':<35} {'Mean ms':>8} {'RTF mean':>10} {'RPS':>6}")
print(f"  {'─'*35} {'─'*8} {'─'*10} {'─'*6}")
for r in all_reports:
    if str(NFE) in r.label and "ablation" not in r.label.lower():
        print(f"  {r.label:<35} {r.latency_mean_ms:>8.0f} {r.rtf_mean:>10.4f} {r.rps:>6.2f}")
print(f"{'='*70}")
print("BF16 BENCHMARK COMPLETE")
