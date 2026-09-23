"""Deterministic, frozen CLIP frame-feature extraction for WSM V1.

This module intentionally has no registry, label, split, or experiment-specific
logic.  It consumes a video path and produces temporal image features only.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


PREPROCESSING_VERSION = "wsm-video-v1"
SAMPLING_METHOD = "uniform_time_inclusive_endpoints"


def uniform_frame_indices(frame_count: int, target_frames: int) -> list[int]:
    """Return deterministic indices spanning the complete available video."""
    if frame_count < 0 or target_frames <= 0:
        raise ValueError("frame_count must be non-negative and target_frames positive")
    if frame_count == 0:
        return []
    if frame_count <= target_frames:
        return list(range(frame_count))
    if target_frames == 1:
        return [0]
    return [round(i * (frame_count - 1) / (target_frames - 1)) for i in range(target_frames)]


def _package_versions() -> dict[str, str]:
    versions = {}
    for name in ("numpy", "opencv-python", "torch", "transformers"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unavailable"
    return versions


def build_cache_fingerprint(
    *,
    manifest_fingerprint: str,
    segment_id: str,
    source_path: str,
    model_name: str,
    model_revision: str,
    target_frames: int,
    preprocessing_version: str = PREPROCESSING_VERSION,
    processor_identity: str = "transformers.CLIPProcessor",
    sampling_method: str = SAMPLING_METHOD,
    package_versions: dict[str, str] | None = None,
) -> str:
    payload = {
        "manifest_fingerprint": manifest_fingerprint,
        "segment_id": segment_id,
        "source_path": source_path,
        "model_name": model_name,
        "model_revision": model_revision,
        "target_frames": target_frames,
        "preprocessing_version": preprocessing_version,
        "processor_identity": processor_identity,
        "sampling_method": sampling_method,
        "package_versions": package_versions if package_versions is not None else _package_versions(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass
class ExtractionResult:
    """Success or explicit failure; failures never contain fake features."""

    success: bool
    features: Any = None
    valid_mask: Any = None
    error: str | None = None
    frame_count: int = 0
    sampled_indices: list[int] | None = None


def _rgb_frame(frame: Any) -> Any:
    """Convert a numpy frame to RGB without adding model-visible metadata."""
    import numpy as np

    array = np.asarray(frame)
    if array.ndim == 2:
        return np.repeat(array[..., None], 3, axis=-1)
    if array.ndim != 3 or array.shape[-1] not in (3, 4):
        raise ValueError(f"unsupported frame shape: {array.shape}")
    return array[..., :3]


def read_video_rgb(source_path: str | Path) -> list[Any]:
    """Read all frames as RGB arrays; an unreadable/empty source is a failure."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required to read video") from exc
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"video could not be opened: {source_path}")
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"video contains no readable frames: {source_path}")
    return frames


class ClipVideoFeatureExtractor:
    """Frozen CLIP image encoder applied independently to sampled RGB frames."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        model_revision: str = "main",
        target_frames: int = 32,
        device: str = "cpu",
        processor: Any = None,
        model: Any = None,
        frame_reader: Callable[[str | Path], list[Any]] = read_video_rgb,
        local_files_only: bool = True,
    ) -> None:
        if target_frames <= 0:
            raise ValueError("target_frames must be positive")
        self.model_name = model_name
        self.model_revision = model_revision
        self.target_frames = target_frames
        self.device = device
        self.processor = processor
        self.model = model
        self.frame_reader = frame_reader
        self.local_files_only = local_files_only

    def _load(self) -> None:
        if self.processor is not None and self.model is not None:
            return
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise RuntimeError("torch and transformers are required for CLIP extraction") from exc
        self.processor = CLIPProcessor.from_pretrained(
            self.model_name, revision=self.model_revision, local_files_only=self.local_files_only
        )
        self.model = CLIPModel.from_pretrained(
            self.model_name, revision=self.model_revision, local_files_only=self.local_files_only
        )
        self.model.to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def extract(self, source_path: str | Path) -> ExtractionResult:
        try:
            frames = self.frame_reader(source_path)
            if not frames:
                raise RuntimeError("frame reader returned zero frames")
            indices = uniform_frame_indices(len(frames), self.target_frames)
            sampled = [_rgb_frame(frames[index]) for index in indices]
            self._load()
            import torch

            inputs = self.processor(images=sampled, return_tensors="pt")
            inputs = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}
            with torch.inference_mode():
                if hasattr(self.model, "get_image_features"):
                    features = self.model.get_image_features(**inputs)
                else:
                    output = self.model.vision_model(pixel_values=inputs["pixel_values"])
                    features = output.pooler_output
            if features.ndim != 2 or features.shape[0] != len(indices):
                raise RuntimeError(f"CLIP returned invalid feature shape: {tuple(features.shape)}")
            return ExtractionResult(
                success=True,
                features=features.detach().cpu(),
                valid_mask=torch.ones(len(indices), dtype=torch.bool),
                frame_count=len(frames),
                sampled_indices=indices,
            )
        except Exception as exc:  # reportable per-segment failure, never a fake artifact
            return ExtractionResult(success=False, error=f"{type(exc).__name__}: {exc}")


def save_cache_artifact(path: str | Path, *, segment_id: str, source_path: str, result: ExtractionResult, target_frames: int,
                        model_name: str, model_revision: str, cache_fingerprint: str) -> None:
    if not result.success or result.features is None or result.valid_mask is None:
        raise ValueError("refusing to save a failed extraction as a cache artifact")
    import torch

    artifact = {
        "segment_id": segment_id,
        "source_path": source_path,
        "features": result.features,
        "valid_mask": result.valid_mask,
        "model_name": model_name,
        "model_revision": model_revision,
        "preprocessing": {
            "version": PREPROCESSING_VERSION,
            "sampling_method": SAMPLING_METHOD,
            "target_frames": target_frames,
            "rgb": True,
        },
        "cache_fingerprint": cache_fingerprint,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, path)
