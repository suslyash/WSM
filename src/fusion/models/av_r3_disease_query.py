"""Availability-aware disease-query audio+video fusion model (R3)."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from chimera_ml.core.registry import MODELS
from chimera_ml.core.types import ModelOutput
from chimera_ml.models.base import BaseModel


class WSMAVR3DiseaseQueryModel(BaseModel):
    """Two independent disease queries with exact modality availability masks."""

    def __init__(
        self,
        audio_feature_dim: int = 768,
        video_feature_dim: int = 512,
        hidden_dim: int = 192,
        num_tasks: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if num_tasks != 2 or min(audio_feature_dim, video_feature_dim, hidden_dim) <= 0:
            raise ValueError("R3 requires two positive-dimensional disease tasks")
        self.audio_feature_dim = int(audio_feature_dim)
        self.video_feature_dim = int(video_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_tasks = int(num_tasks)
        self.audio_projection = nn.Sequential(
            nn.LayerNorm(self.audio_feature_dim),
            nn.Linear(self.audio_feature_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.video_projection = nn.Sequential(
            nn.LayerNorm(self.video_feature_dim),
            nn.Linear(self.video_feature_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.disease_queries = nn.Parameter(torch.empty(self.num_tasks, self.hidden_dim))
        nn.init.normal_(self.disease_queries, mean=0.0, std=0.02)
        self.audio_candidates = nn.ModuleList(
            [nn.LayerNorm(self.hidden_dim) for _ in range(self.num_tasks)]
        )
        self.video_candidates = nn.ModuleList(
            [nn.LayerNorm(self.hidden_dim) for _ in range(self.num_tasks)]
        )
        self.query_gate = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim // 4, 1),
        )
        self.task_fusion_norms = nn.ModuleList(
            [nn.LayerNorm(self.hidden_dim) for _ in range(self.num_tasks)]
        )
        self.main_heads = nn.ModuleList([self._head(dropout) for _ in range(self.num_tasks)])
        self.audio_aux_heads = nn.ModuleList([self._head(dropout) for _ in range(self.num_tasks)])
        self.video_aux_heads = nn.ModuleList([self._head(dropout) for _ in range(self.num_tasks)])

    def _head(self, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim // 2, 1),
        )

    def forward(self, batch: Any) -> ModelOutput:
        audio_cls = batch.inputs["audio_cls"]
        video = batch.inputs["video"]
        video_mask = batch.get_masks("video_mask").to(dtype=torch.bool, device=video.device)
        available = batch.get_masks("modality_available").to(dtype=torch.bool, device=video.device)
        if audio_cls.ndim != 2 or tuple(audio_cls.shape[1:]) != (self.audio_feature_dim,):
            raise ValueError(f"audio_cls must have shape [B,{self.audio_feature_dim}]")
        if video.ndim != 3 or video.shape[2] != self.video_feature_dim:
            raise ValueError(f"video must have shape [B,T,{self.video_feature_dim}]")
        if available.shape != (audio_cls.shape[0], 2):
            raise ValueError("modality_available must have shape [B,2]")
        if bool((~available.any(dim=1)).any()):
            raise ValueError("every sample must have at least one available modality")
        if bool((available[:, 1] & ~video_mask.any(dim=1)).any()):
            raise ValueError("available video samples need a valid video position")
        audio_available = available[:, 0:1].to(dtype=audio_cls.dtype, device=audio_cls.device)
        video_available = available[:, 1:2].to(dtype=video.dtype, device=video.device)
        audio_features = self.audio_projection(audio_cls * audio_available)
        video_weights = video_mask.unsqueeze(-1).to(dtype=video.dtype)
        video_raw = (video * video_weights).sum(dim=1) / video_weights.sum(dim=1).clamp_min(1)
        video_features = self.video_projection(video_raw * video_available)
        task_features = []
        task_weights = []
        audio_aux = []
        video_aux = []
        for task, query in enumerate(self.disease_queries):
            audio_candidate = self.audio_candidates[task](audio_features)
            video_candidate = self.video_candidates[task](video_features)
            query_batch = query.unsqueeze(0).expand(audio_features.shape[0], -1)
            gate_audio = self.query_gate(torch.cat([audio_candidate, query_batch], dim=1)).squeeze(1)
            gate_video = self.query_gate(torch.cat([video_candidate, query_batch], dim=1)).squeeze(1)
            gate_logits = torch.stack([gate_audio, gate_video], dim=1)
            gate_logits = gate_logits.masked_fill(~available, torch.finfo(gate_logits.dtype).min)
            weights = torch.softmax(gate_logits, dim=1)
            fused = self.task_fusion_norms[task](
                weights[:, 0:1] * audio_candidate + weights[:, 1:2] * video_candidate + query_batch
            )
            task_features.append(fused)
            task_weights.append(weights)
            audio_aux.append(self.audio_aux_heads[task](audio_features))
            video_aux.append(self.video_aux_heads[task](video_features))
        task_features_tensor = torch.stack(task_features, dim=1)
        preds = torch.cat([self.main_heads[t](task_features_tensor[:, t]) for t in range(self.num_tasks)], dim=1)
        return ModelOutput(
            preds=preds,
            aux={
                "task_features": task_features_tensor,
                "modality_weights": torch.stack(task_weights, dim=1),
                "audio_aux_logits": torch.cat(audio_aux, dim=1),
                "video_aux_logits": torch.cat(video_aux, dim=1),
                "audio_aux_valid": available[:, 0:1].expand(-1, self.num_tasks),
                "video_aux_valid": available[:, 1:2].expand(-1, self.num_tasks),
            },
        )


@MODELS.register("wsm_av_r3_disease_query_model")
def wsm_av_r3_disease_query_model(context: Any | None = None, **params: Any) -> WSMAVR3DiseaseQueryModel:
    params = dict(params)
    if context is not None:
        params.setdefault("audio_feature_dim", context.get("data.audio_feature_dim", 768))
        params.setdefault("video_feature_dim", context.get("data.video_feature_dim", 512))
        params.setdefault("num_tasks", context.get("data.num_tasks", 2))
    return WSMAVR3DiseaseQueryModel(**params)
