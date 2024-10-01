from typing import Literal
import importlib

Module = Literal[
    "",
    "Conv2d",
    "CausalConv3d",
    "AttnBlock",
    "AttnBlock3D",
    "Upsample",
    "Downsample",
    "SpatialUpsample2x",
    "SpatialDownsample2x",
    "Spatial2xTime2x3DUpsample",
    "Spatial2xTime2x3DDownsample",
]

MODULES_BASE = "algorithms.vae.common.modules"


def resolve_str_to_module(name: str) -> Module:
    if name == "":
        raise ValueError("Empty string is not a valid module name.")
    module = importlib.import_module(MODULES_BASE)
    return getattr(module, name)
