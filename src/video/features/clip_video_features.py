"""Deterministic DEPART-style YOLOv8 ROI + frozen CLIP extraction for WSM V1."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
PREPROCESSING_VERSION = "depart-v1-roi"
SAMPLING_METHOD = "uniform_time_inclusive_endpoints"
ROI_POLICY_VERSION = "depart-body-roi-v1"

def uniform_frame_indices(frame_count: int, target_frames: int) -> list[int]:
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
    for name in ("numpy", "opencv-python", "torch", "transformers", "ultralytics"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unavailable"
    return versions

def build_cache_fingerprint(*, manifest_fingerprint: str, segment_id: str, source_path: str,
                            model_name: str, model_revision: str, target_frames: int,
                            preprocessing_version: str = PREPROCESSING_VERSION,
                            processor_identity: str = "transformers.CLIPProcessor",
                            sampling_method: str = SAMPLING_METHOD,
                            package_versions: dict[str, str] | None = None,
                            detector_identity: str | None = None,
                            detector_weights_sha256: str | None = None,
                            detector_confidence: float | None = None,
                            detector_iou: float | None = None,
                            detector_imgsz: int | None = None,
                            roi_policy_version: str | None = None) -> str:
    payload = {
        "manifest_fingerprint": manifest_fingerprint, "segment_id": segment_id,
        "source_path": source_path, "model_name": model_name, "model_revision": model_revision,
        "target_frames": target_frames, "preprocessing_version": preprocessing_version,
        "processor_identity": processor_identity, "sampling_method": sampling_method,
        "package_versions": package_versions if package_versions is not None else _package_versions(),
        "detector_identity": detector_identity, "detector_weights_sha256": detector_weights_sha256,
        "detector_confidence": detector_confidence, "detector_iou": detector_iou,
        "detector_imgsz": detector_imgsz, "roi_policy_version": roi_policy_version,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()

@dataclass
class ExtractionResult:
    success: bool
    features: Any = None
    valid_mask: Any = None
    error: str | None = None
    frame_count: int = 0
    sampled_indices: list[int] | None = None
    selected_boxes: list[tuple[int, int, int, int] | None] | None = None
    detection_confidences: list[float | None] | None = None
    valid_detection_count: int = 0
    detection_coverage: float = 0.0
    detector_identity: str | None = None
    failure_kind: str | None = None

def normalize_clip_image_features(output: Any) -> Any:
    """Normalize supported CLIP image outputs to a rank-2 tensor [B,D]."""
    import torch
    if isinstance(output, torch.Tensor):
        tensor = output
    else:
        image_embeds = getattr(output, "image_embeds", None)
        if isinstance(image_embeds, torch.Tensor):
            tensor = image_embeds
        else:
            pooler_output = getattr(output, "pooler_output", None)
            if isinstance(pooler_output, torch.Tensor):
                tensor = pooler_output
            else:
                raise RuntimeError(f"unsupported CLIP image-feature output type: {type(output).__name__}")
    if tensor.ndim != 2:
        raise RuntimeError(f"CLIP image features must be rank-2 [B,D], got shape {tuple(tensor.shape)}")
    return tensor

def _rgb_frame(frame: Any) -> Any:
    import numpy as np
    array = np.asarray(frame)
    if array.ndim == 2:
        return np.repeat(array[..., None], 3, axis=-1)
    if array.ndim != 3 or array.shape[-1] not in (3, 4):
        raise ValueError(f"unsupported frame shape: {array.shape}")
    return array[..., :3]

def read_video_rgb(source_path: str | Path) -> list[Any]:
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
    """Frozen CLIP image encoder; ROI mode is the default and raw mode is debug-only."""
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", model_revision: str = "main",
                 target_frames: int = 60, device: str = "cpu", processor: Any = None, model: Any = None,
                 frame_reader: Callable[[str | Path], list[Any]] = read_video_rgb,
                 local_files_only: bool = True, detector: Any = None, raw_frame_debug: bool = False) -> None:
        if target_frames <= 0:
            raise ValueError("target_frames must be positive")
        self.model_name, self.model_revision, self.target_frames = model_name, model_revision, target_frames
        self.device, self.processor, self.model = device, processor, model
        self.frame_reader, self.local_files_only = frame_reader, local_files_only
        self.detector, self.raw_frame_debug = detector, raw_frame_debug

    def _load(self) -> None:
        if self.processor is not None and self.model is not None:
            return
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise RuntimeError("torch and transformers are required for CLIP extraction") from exc
        self.processor = CLIPProcessor.from_pretrained(self.model_name, revision=self.model_revision,
                                                       local_files_only=self.local_files_only)
        self.model = CLIPModel.from_pretrained(self.model_name, revision=self.model_revision,
                                               local_files_only=self.local_files_only)
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
            selected_boxes: list[tuple[int, int, int, int] | None] = [None] * len(sampled)
            detection_confidences: list[float | None] = [None] * len(sampled)
            crops = sampled
            if self.detector is not None:
                crops = []
                valid_positions = []
                for position, frame in enumerate(sampled):
                    detection = self.detector.detect(frame)
                    if detection is None:
                        continue
                    selected_boxes[position] = detection.box
                    detection_confidences[position] = detection.confidence
                    x1, y1, x2, y2 = detection.box
                    crops.append(frame[y1:y2, x1:x2])
                    valid_positions.append(position)
                if not crops:
                    return ExtractionResult(False, error="no body detection in any sampled frame",
                                            frame_count=len(frames), sampled_indices=indices,
                                            selected_boxes=selected_boxes,
                                            detection_confidences=detection_confidences,
                                            detector_identity=getattr(self.detector, "identity", None),
                                            failure_kind="no_body_detected")
            elif not self.raw_frame_debug:
                raise RuntimeError("default V1 extraction requires a configured YOLOv8 body detector")
            self._load()
            import torch
            inputs = self.processor(images=crops, return_tensors="pt")
            inputs = {key: value.to(self.device) if hasattr(value, "to") else value
                      for key, value in inputs.items()}
            with torch.inference_mode():
                if hasattr(self.model, "get_image_features"):
                    raw_features = self.model.get_image_features(**inputs)
                else:
                    raw_features = self.model.vision_model(pixel_values=inputs["pixel_values"])
                features = normalize_clip_image_features(raw_features)
            if features.shape[0] != len(crops):
                raise RuntimeError(f"CLIP returned invalid feature shape: {tuple(features.shape)}")
            if self.detector is not None:
                temporal = torch.zeros((len(indices), features.shape[1]), dtype=features.dtype)
                valid_mask = torch.zeros(len(indices), dtype=torch.bool)
                valid_positions = [i for i, box in enumerate(selected_boxes) if box is not None]
                temporal[valid_positions] = features.detach().cpu()
                valid_mask[valid_positions] = True
                features = temporal
            else:
                features = features.detach().cpu()
                valid_mask = torch.ones(len(indices), dtype=torch.bool)
            valid_count = int(valid_mask.sum().item())
            return ExtractionResult(True, features=features, valid_mask=valid_mask,
                                    frame_count=len(frames), sampled_indices=indices,
                                    selected_boxes=selected_boxes,
                                    detection_confidences=detection_confidences,
                                    valid_detection_count=valid_count,
                                    detection_coverage=valid_count / len(indices),
                                    detector_identity=getattr(self.detector, "identity", None))
        except Exception as exc:
            return ExtractionResult(False, error=f"{type(exc).__name__}: {exc}",
                                    failure_kind="video_read_or_model_load")

def save_cache_artifact(path: str | Path, *, segment_id: str, source_path: str, result: ExtractionResult,
                        target_frames: int, model_name: str, model_revision: str,
                        cache_fingerprint: str, detector: Any = None) -> None:
    if not result.success or result.features is None or result.valid_mask is None:
        raise ValueError("refusing to save a failed extraction as a cache artifact")
    import torch
    artifact = {
        "segment_id": segment_id, "source_path": source_path, "features": result.features,
        "valid_mask": result.valid_mask, "model_name": model_name, "model_revision": model_revision,
        "preprocessing": {"version": PREPROCESSING_VERSION, "sampling_method": SAMPLING_METHOD,
                          "target_frames": target_frames, "rgb": True, "roi_policy_version": ROI_POLICY_VERSION},
        "cache_fingerprint": cache_fingerprint, "detector_identity": result.detector_identity,
        "detector_weights_sha256": getattr(detector, "weights_sha256", None),
        "detection_parameters": {"confidence": getattr(detector, "confidence", None),
                                 "iou": getattr(detector, "iou", None), "imgsz": getattr(detector, "imgsz", None)},
        "sampled_frame_indices": result.sampled_indices, "selected_boxes": result.selected_boxes,
        "detection_confidences": result.detection_confidences,
        "valid_detection_count": result.valid_detection_count, "detection_coverage": result.detection_coverage,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, path)
