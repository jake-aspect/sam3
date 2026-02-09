# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
"""
SAM3 (Segment Anything Model 3) -- Windows/CPU compatible fork.

Public API is loaded lazily so that ``import sam3`` never triggers heavy
transitive dependencies (triton, CUDA kernels, etc.).
"""

__version__ = "0.1.0"

__all__ = ["build_sam3_image_model"]


def __getattr__(name: str):
    if name == "build_sam3_image_model":
        from .model_builder import build_sam3_image_model

        return build_sam3_image_model

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
