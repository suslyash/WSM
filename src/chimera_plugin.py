from __future__ import annotations

import importlib
import warnings


_MODULES_TO_REGISTER: tuple[str, ...] = (
    "common.callbacks.wsm_segment_callback",
    "common.callbacks.wsm_summary_callback",
    "audio.data.wsm_audio_segment_datamodule",
    "audio.models.audio_mamba_segment",
    "audio.loss.wsm_audio_loss",
    "fusion.models.av_sync_mamba_segment"
)


def register() -> None:
    for module_name in _MODULES_TO_REGISTER:
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name in {"mamba_ssm", "torch", "pandas", "numpy", "matplotlib"}:
                warnings.warn(f"Optional dependency is missing while importing '{module_name}': {exc}", stacklevel=2)
                continue

            warnings.warn(f"Failed to import '{module_name}': {exc}", stacklevel=2)
        except Exception as exc:
            warnings.warn(f"Failed to import '{module_name}': {exc}", stacklevel=2)
