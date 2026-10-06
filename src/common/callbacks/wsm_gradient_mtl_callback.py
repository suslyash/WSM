"""Epoch logging for TASK-010A parameter-gradient comparator diagnostics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from chimera_ml.callbacks.base import BaseCallback
from chimera_ml.core.registry import CALLBACKS


@dataclass
class WSMGradientMTLCallback(BaseCallback):
    """Expose train-only gradient-combiner diagnostics at each DEV epoch."""

    def _loss(self, trainer: Any) -> Any:
        loss = getattr(trainer, "loss_fn", None)
        if loss is None or not hasattr(loss, "epoch_diagnostics") or not hasattr(loss, "reset_epoch_diagnostics"):
            raise TypeError("wsm_gradient_mtl_callback requires wsm_gradient_mtl_loss")
        return loss

    def on_fit_start(self, trainer: Any) -> None:
        loss = self._loss(trainer)
        model = getattr(trainer, "model", None)
        if model is None or not hasattr(loss, "bind_model"):
            raise TypeError("wsm_gradient_mtl_callback requires trainer.model and loss.bind_model")
        loss.bind_model(model)
        if loss.ownership_counts != {"shared": 14, "depression": 18, "parkinson": 18, "row": 1}:
            raise ValueError(f"unexpected R4 structural ownership: {loss.ownership_counts}")

    def on_epoch_start(self, trainer: Any, epoch: int) -> None:
        self._loss(trainer).reset_epoch_diagnostics()

    def on_epoch_end(self, trainer: Any, epoch: int, logs: dict[str, float]) -> None:
        loss = self._loss(trainer)
        diagnostics = loss.epoch_diagnostics()
        logs.update(diagnostics)
        weights = getattr(loss, "gradnorm_weights", None)
        if loss.method == "gradnorm" and weights is not None:
            logs["mtl/final_weight_depression"] = float(weights[0])
            logs["mtl/final_weight_parkinson"] = float(weights[1])
        logger = getattr(trainer, "mlflow_logger", None)
        if logger is not None and diagnostics:
            logger.log_metrics(diagnostics, step=epoch)


@CALLBACKS.register("wsm_gradient_mtl_callback")
def wsm_gradient_mtl_callback(context: Any | None = None, **params: Any) -> WSMGradientMTLCallback:
    del context, params
    return WSMGradientMTLCallback()
