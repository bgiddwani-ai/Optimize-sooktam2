#!/usr/bin/env python3
"""Minimal dynamic-shape AOTInductor preflight for the deployed PyTorch build."""

from pathlib import Path

import torch
from torch._export import aot_compile, aot_load
from torch.export import Dim


def main() -> None:
    torch.manual_seed(0)
    module = torch.nn.Linear(64, 64).cuda().eval()
    example = torch.randn(1, 16, 64, device="cuda")
    sequence = Dim("sequence", min=4, max=64)
    library = aot_compile(
        module,
        (example,),
        dynamic_shapes=({1: sequence},),
    )
    loaded = aot_load(library, "cuda")
    output = loaded(torch.randn(1, 24, 64, device="cuda"))
    if isinstance(output, (list, tuple)):
        output = output[0]
    assert output.shape == (1, 24, 64), output.shape
    print(Path(library).resolve())
    print("AOTI_DYNAMIC_PREFLIGHT_PASS")


if __name__ == "__main__":
    main()
