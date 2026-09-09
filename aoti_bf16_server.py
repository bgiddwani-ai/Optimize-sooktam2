#!/usr/bin/env python3
"""BF16 AOTInductor HTTP service for Sooktam2.

The DiT checkpoint remains FP32.  Inference uses BF16 autocast, TF32 is enabled
for FP32 matmuls, and the DiT ODE step is exported with ``torch._export`` then
loaded as an AOTInductor shared library.  Text embedding and RoPE are computed
once per request, outside the ODE loop, following the supplied reference.

The model has process-global RNG and cache state.  Requests are therefore
serialized at the model boundary; client concurrency is still measured end to
end, but is not falsely presented as unsafe model microbatching.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from torch._export import aot_compile, aot_load
from torch.export import Dim
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint
from transformers import AutoModel

from f5_tts.model.backbones.dit import DiT
from f5_tts.model.cfm import CFM
from f5_tts.model.utils import (
    exists,
    get_epss_timesteps,
    lens_to_mask,
    list_str_to_idx,
    list_str_to_tensor,
)


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")


def _forward_compiled(
    self: DiT,
    x: torch.Tensor,
    cond: torch.Tensor,
    text_embed_cond: torch.Tensor,
    text_embed_uncond: torch.Tensor,
    time_value: torch.Tensor,
    rope: tuple[torch.Tensor, float],
    mask: torch.Tensor | None = None,
    cfg_infer: bool = True,
) -> torch.Tensor:
    """Stateless DiT forward used by AOTI.

    This is the supplied reference update with the text-embedding cache removed
    from the compiled graph.
    """

    batch = x.shape[0]
    if time_value.ndim == 0:
        time_value = time_value.repeat(batch)
    time_embedding = self.time_embed(time_value)

    if cfg_infer:
        x_cond = self.input_embed(x, cond, text_embed_cond, drop_audio_cond=False)
        x_uncond = self.input_embed(x, cond, text_embed_uncond, drop_audio_cond=True)
        x = torch.cat((x_cond, x_uncond), dim=0)
        time_embedding = torch.cat((time_embedding, time_embedding), dim=0)
        if mask is not None:
            mask = torch.cat((mask, mask), dim=0)
    else:
        x = self.input_embed(x, cond, text_embed_cond, drop_audio_cond=False)

    residual = x if self.long_skip_connection is not None else None
    for block in self.transformer_blocks:
        x = block(x, time_embedding, mask=mask, rope=rope, skip_flash_attn=True)
    if self.long_skip_connection is not None:
        x = self.long_skip_connection(torch.cat((x, residual), dim=-1))
    return self.proj_out(self.norm_out(x, time_embedding))


class _DiTStep(torch.nn.Module):
    """Exportable CFG DiT step; all inputs are tensors for AOTI."""

    def __init__(self, dit: DiT) -> None:
        super().__init__()
        self.dit = dit

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        text_embed_cond: torch.Tensor,
        text_embed_uncond: torch.Tensor,
        time_value: torch.Tensor,
        rope_freqs: torch.Tensor,
    ) -> torch.Tensor:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return self.dit.forward_compiled(
                x,
                cond,
                text_embed_cond,
                text_embed_uncond,
                time_value,
                # The installed x-transformers RotaryEmbedding returns a
                # dynamic frequency tensor plus the constant scale ``1.0``.
                # Keep that scalar inside the exported module.
                (rope_freqs, 1.0),
                mask=None,
                cfg_infer=True,
            )


class AOTIDiTStep:
    """Persistent dynamic-sequence AOTInductor DiT step."""

    def __init__(self, dit: DiT, artifact_dir: Path, max_sequence: int) -> None:
        self.dit = dit
        self.artifact_dir = artifact_dir
        self.max_sequence = max_sequence
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.library_path = self.artifact_dir / "sooktam2_dit_bf16_aoti.so"
        self._callable: Callable[..., object] | None = None

    def build_or_load(self) -> None:
        if self.library_path.exists():
            self._callable = aot_load(str(self.library_path), "cuda")
            return

        sequence_length = min(512, self.max_sequence)
        device = torch.device("cuda")
        model_dtype = next(self.dit.parameters()).dtype
        # DiT stores these on submodules, not as public ``mel_dim`` / text-dim
        # fields.  These exact accessors were validated against Sooktam2.
        text_dim = self.dit.text_embed.text_embed.embedding_dim
        mel_dim = self.dit.proj_out.out_features
        rope_freqs, rope_scale = self.dit.rotary_embed.forward_from_seq_len(sequence_length)
        if rope_scale != 1.0:
            raise RuntimeError(f"unsupported non-constant RoPE scale: {rope_scale!r}")
        example_args = (
            torch.zeros((1, sequence_length, mel_dim), device=device, dtype=model_dtype),
            torch.zeros((1, sequence_length, mel_dim), device=device, dtype=model_dtype),
            torch.zeros((1, sequence_length, text_dim), device=device, dtype=model_dtype),
            torch.zeros((1, sequence_length, text_dim), device=device, dtype=model_dtype),
            torch.tensor(0.5, device=device, dtype=model_dtype),
            rope_freqs.to(device=device, dtype=model_dtype),
        )
        sequence = Dim("sequence", min=8, max=self.max_sequence)
        dynamic_shapes = (
            {1: sequence},
            {1: sequence},
            {1: sequence},
            {1: sequence},
            None,
            {1: sequence},
        )
        generated_library = aot_compile(
            _DiTStep(self.dit).eval(),
            example_args,
            dynamic_shapes=dynamic_shapes,
        )
        shutil.copy2(generated_library, self.library_path)
        self._callable = aot_load(str(self.library_path), "cuda")

    def __call__(self, *args: torch.Tensor) -> torch.Tensor:
        if self._callable is None:
            raise RuntimeError("AOTI step has not been loaded")
        output = self._callable(*args)
        if isinstance(output, (list, tuple)):
            output = output[0]
        if not isinstance(output, torch.Tensor):
            raise TypeError(f"Unexpected AOTI output type: {type(output)!r}")
        return output


@torch.no_grad()
def _sample_aoti(
    self: CFM,
    cond: torch.Tensor,
    text: torch.Tensor | list[str] | list[list[str]],
    duration: int | torch.Tensor,
    *,
    lens: torch.Tensor | None = None,
    steps: int = 32,
    cfg_strength: float = 1.0,
    sway_sampling_coef: float | None = None,
    seed: int | None = None,
    max_duration: int = 4096,
    vocoder=None,
    use_epss: bool = True,
    no_ref_audio: bool = False,
    duplicate_test: bool = False,
    t_inter: float = 0.1,
    edit_mask: torch.Tensor | None = None,
):
    """CFM sampler with text/RoPE hoisted and its DiT step routed to AOTI."""

    del duplicate_test, t_inter, edit_mask
    self.eval()
    if cond.ndim == 2:
        cond = self.mel_spec(cond).permute(0, 2, 1)
        assert cond.shape[-1] == self.num_channels
    cond = cond.to(next(self.parameters()).dtype)
    batch, cond_sequence_length, device = *cond.shape[:2], cond.device
    if batch != 1:
        raise ValueError("The AOTI service deliberately serializes model calls; batch must be one")
    if lens is None:
        lens = torch.full((batch,), cond_sequence_length, device=device, dtype=torch.long)
    if isinstance(text, list):
        if self.vocab_char_map is not None:
            text = list_str_to_idx(text, self.vocab_char_map).to(device)
        else:
            text = list_str_to_tensor(text).to(device)
    if isinstance(duration, int):
        duration = torch.full((batch,), duration, device=device, dtype=torch.long)
    duration = torch.maximum(torch.maximum((text != -1).sum(dim=-1), lens) + 1, duration)
    duration = duration.clamp(max=max_duration)
    max_duration = int(duration.amax().item())

    cond = F.pad(cond, (0, 0, 0, max_duration - cond_sequence_length), value=0.0)
    if no_ref_audio:
        cond = torch.zeros_like(cond)
    cond_mask = lens_to_mask(lens)
    cond_mask = F.pad(cond_mask, (0, max_duration - cond_mask.shape[-1]), value=False).unsqueeze(-1)
    step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))

    text_embed_cond = self.transformer.text_embed(text, max_duration, drop_text=False, audio_mask=None)
    text_embed_uncond = self.transformer.text_embed(text, max_duration, drop_text=True, audio_mask=None)
    rope_freqs, rope_scale = self.transformer.rotary_embed.forward_from_seq_len(max_duration)
    if rope_scale != 1.0:
        raise RuntimeError(f"unsupported non-constant RoPE scale: {rope_scale!r}")

    if seed is not None:
        torch.manual_seed(seed)
    y0 = torch.randn((batch, max_duration, self.num_channels), device=device, dtype=step_cond.dtype)
    aoti_step = getattr(self.transformer, "_sooktam_aoti_step", None)
    if aoti_step is None:
        raise RuntimeError("AOTI DiT step is not installed")

    if use_epss:
        timesteps = get_epss_timesteps(steps, device=device, dtype=step_cond.dtype)
    else:
        timesteps = torch.linspace(0, 1, steps + 1, device=device, dtype=step_cond.dtype)
    if sway_sampling_coef is not None:
        timesteps = timesteps + sway_sampling_coef * (torch.cos(torch.pi / 2 * timesteps) - 1 + timesteps)

    def ode_function(timestep: torch.Tensor, noisy: torch.Tensor) -> torch.Tensor:
        prediction_cfg = aoti_step(
            noisy,
            step_cond,
            text_embed_cond,
            text_embed_uncond,
            timestep,
            rope_freqs,
        )
        prediction, null_prediction = torch.chunk(prediction_cfg, 2, dim=0)
        return prediction + (prediction - null_prediction) * cfg_strength

    trajectory = odeint(ode_function, y0, timesteps, **self.odeint_kwargs)
    # Match upstream CFM.sample: use the full generated trajectory rather than
    # splicing the prompt mel back into the output.
    sampled = trajectory[-1]
    if vocoder is not None:
        sampled = vocoder(sampled.permute(0, 2, 1))
    return sampled, trajectory


def _request_seed(reference_text: str, target_text: str, supplied_seed: int | None) -> int:
    if supplied_seed is not None:
        return supplied_seed
    digest = hashlib.blake2b(
        f"{reference_text}\0{target_text}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "little") & ((1 << 63) - 1)


class SooktamAOTIService:
    def __init__(self, model_dir: Path, artifact_dir: Path, nfe_steps: int, max_sequence: int) -> None:
        self.model_dir = model_dir
        self.artifact_dir = artifact_dir
        self.nfe_steps = nfe_steps
        self.max_sequence = max_sequence
        self.lock = asyncio.Lock()
        self.model = None
        self.tts = None
        self.compile_seconds: float | None = None

    def initialize(self) -> None:
        started = time.perf_counter()
        self.model = AutoModel.from_pretrained(
            str(self.model_dir), trust_remote_code=True, local_files_only=True
        ).eval()
        self.tts = self.model.tts
        DiT.forward_compiled = _forward_compiled
        CFM.sample = _sample_aoti
        accelerator = AOTIDiTStep(self.tts.ema_model.transformer, self.artifact_dir, self.max_sequence)
        accelerator.build_or_load()
        self.tts.ema_model.transformer._sooktam_aoti_step = accelerator
        self.compile_seconds = time.perf_counter() - started

    def infer_sync(self, reference_audio: bytes, reference_text: str, target_text: str, seed: int | None) -> tuple[bytes, float, int]:
        if self.tts is None:
            raise RuntimeError("service is not initialized")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temporary:
            temporary.write(reference_audio)
            reference_path = temporary.name
        try:
            started = time.perf_counter()
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                waveform, sample_rate, _ = self.tts.infer(
                    ref_file=reference_path,
                    ref_text=reference_text,
                    gen_text=target_text,
                    tokenizer="cls",
                    cls_language="hindi",
                    nfe_step=self.nfe_steps,
                    speed=1.0,
                    cfg_strength=2.0,
                    seed=_request_seed(reference_text, target_text, seed),
                    remove_silence=False,
                    show_info=lambda *_args, **_kwargs: None,
                    progress=None,
                )
            elapsed = time.perf_counter() - started
            waveform = np.asarray(waveform, dtype=np.float32)
            if not waveform.size or not np.isfinite(waveform).all():
                raise RuntimeError("model returned an empty or non-finite waveform")
            payload = io.BytesIO()
            sf.write(payload, waveform, sample_rate, format="WAV", subtype="PCM_16")
            return payload.getvalue(), elapsed, sample_rate
        finally:
            os.unlink(reference_path)


def create_app(service: SooktamAOTIService) -> FastAPI:
    app = FastAPI(title="Sooktam2 BF16 AOTI", version="1")

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        return JSONResponse(
            {
                "ready": service.tts is not None,
                "backend": "aoti-bf16",
                "nfe_steps": service.nfe_steps,
                "max_sequence": service.max_sequence,
                "compile_seconds": service.compile_seconds,
                "model_boundary": "serialized",
            }
        )

    @app.post("/v1/infer")
    async def infer(
        reference_audio: UploadFile = File(...),
        reference_text: str = Form(...),
        target_text: str = Form(...),
        seed: int | None = Form(default=None),
    ) -> Response:
        audio_bytes = await reference_audio.read()
        if not audio_bytes or not reference_text.strip() or not target_text.strip():
            raise HTTPException(status_code=400, detail="reference audio, reference text, and target text are required")
        async with service.lock:
            try:
                payload, elapsed, sample_rate = await asyncio.to_thread(
                    service.infer_sync, audio_bytes, reference_text, target_text, seed
                )
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"inference failed: {type(exc).__name__}: {exc}") from exc
        return Response(
            content=payload,
            media_type="audio/wav",
            headers={
                "X-Sooktam-Backend": "aoti-bf16",
                "X-Sooktam-Server-Latency-Ms": f"{elapsed * 1000:.3f}",
                "X-Sooktam-Sample-Rate": str(sample_rate),
            },
        )

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("/workspace/src/sooktam2"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("/workspace/artifacts/aoti_bf16"))
    parser.add_argument("--nfe-steps", type=int, default=32)
    parser.add_argument("--max-sequence", type=int, default=4096)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8010)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    service = SooktamAOTIService(
        arguments.model_dir, arguments.artifact_dir, arguments.nfe_steps, arguments.max_sequence
    )
    service.initialize()
    import uvicorn

    uvicorn.run(create_app(service), host=arguments.host, port=arguments.port, log_level="info")
