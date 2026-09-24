"""Fusion models."""
from .av_f0_gated_late import WSMAVGatedLateFusionF0Model, wsm_av_f0_gated_late_model
from .av_f1_shared_mtl import WSMAVSharedMTLF1Model, wsm_av_f1_shared_mtl_model

__all__ = [
    "WSMAVGatedLateFusionF0Model", "wsm_av_f0_gated_late_model",
    "WSMAVSharedMTLF1Model", "wsm_av_f1_shared_mtl_model",
]
