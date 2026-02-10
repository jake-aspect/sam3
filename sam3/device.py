# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
"""
Centralized device selection for SAM3.

``COMPUTE_DEVICE`` is the preferred accelerator (CUDA when available,
otherwise CPU).  Import it wherever you need a sensible default device
instead of hard-coding ``"cuda"``.
"""

import torch

COMPUTE_DEVICE: torch.device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
