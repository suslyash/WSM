from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from chimera_ml.callbacks._utils import resolve_splits
from chimera_ml.callbacks.base import BaseCallback
from chimera_ml.core.batch import Batch
from chimera_ml.core.registry import CALLBACKS
from chimera_ml.core.types import ModelOutput
from chimera_ml.metrics.confusion_matrix_metric import ConfusionMatrixMetric
from chimera_ml.training.cached_split_outputs import CachedSplitOutputs


@dataclass
class WSMSegmentMetricsCallback(BaseCallback):
    splits: list[str] = field(default_factory=lambda: ["auto"])
    task: str = "depression"
    task_names: list[str] = field(default_factory=lambda: ["depression", "parkinson"])
    class_names: list[str] = field(default_factory=lambda: ["healthy", "pathological"])
    artifact_path: str = "figures/wsm_segment_confusion_matrices"
    filename_template: str = "epoch_{epoch}.pdf"

    def on_fit_start(self, trainer: Any) -> None:
        trainer.config.collect_cache = True

    @torch.no_grad()
    def on_epoch_end(self, trainer: Any, epoch: int, logs: dict[str, float]) -> None:
        multitask_metrics = self._multitask_metrics(logs)
        logs.update(multitask_metrics)
        if trainer.mlflow_logger is not None and multitask_metrics:
            trainer.mlflow_logger.log_metrics(multitask_metrics, step=epoch)

        panels: list[dict[str, Any]] = []

        split_names = self._resolve_cached_splits(trainer)
        for split in split_names:
            cached = trainer.get_cached_split_outputs(split)
            if cached is None:
                continue

            panel = self._split_panel(split, cached, logs)
            if panel is not None:
                panels.append(panel)

        if trainer.mlflow_logger is not None and panels:
            self._log_image(trainer.mlflow_logger, panels, epoch)

        if multitask_metrics:
            shown = " | ".join(
                f"{key}={value:.4f}" for key, value in sorted(multitask_metrics.items()) if "mean_macro" in key or key.endswith("/mean_score")
            )
            self._info(trainer, f"[WSMSegmentMetricsCallback] {shown}")

    def _resolve_cached_splits(self, trainer: Any) -> list[str]:
        if "auto" in set(self.splits):
            return sorted(getattr(trainer, "cached_outputs", {}).keys())

        return [split for split, _ in resolve_splits(trainer, self.splits)]

    def _multitask_metrics(self, logs: dict[str, float]) -> dict[str, float]:
        out: dict[str, float] = {}
        prefixes = ("dev", "test_none", "test_soft", "test_hard")
        metric_names = ("num_samples", "loss", "macro_precision", "macro_recall", "macro_f1")
        for prefix in prefixes:
            values: dict[str, list[float]] = {
                "loss": [],
                "macro_precision": [],
                "macro_recall": [],
                "macro_f1": [],
                "score": [],
            }
            for task in self.task_names:
                task_metrics: dict[str, float] = {}
                for metric_name in metric_names:
                    value = logs.get(f"{prefix}_{task}/{metric_name}")
                    if value is None:
                        continue

                    value = float(value)
                    out[f"{prefix}/{task}/{metric_name}"] = value
                    task_metrics[metric_name] = value
                    if metric_name in values:
                        values[metric_name].append(value)

                if "macro_recall" in task_metrics and "macro_f1" in task_metrics:
                    score = 0.5 * (task_metrics["macro_recall"] + task_metrics["macro_f1"])
                    out[f"{prefix}/{task}/score"] = score
                    values["score"].append(score)

            for metric_name, items in values.items():
                if not items:
                    continue
                out[f"{prefix}/mean_{metric_name}"] = float(np.mean(items))

        return out

    def _split_panel(
        self,
        split: str,
        cached: CachedSplitOutputs,
        logs: dict[str, float],
    ) -> dict[str, Any] | None:
        logits = CachedSplitOutputs._concat_chunks(cached.preds)
        targets = CachedSplitOutputs._concat_chunks(cached.targets)
        if logits is None or targets is None:
            return None

        return self._panel(
            split,
            self._confusion_matrix(logits, targets),
            float(logs.get(f"{split}/macro_recall", float("nan"))),
            float(logs.get(f"{split}/macro_f1", float("nan"))),
        )

    def _confusion_matrix(self, logits: torch.Tensor, targets: torch.Tensor) -> np.ndarray:
        output = ModelOutput(preds=logits)
        batch = Batch(inputs={}, targets=targets, meta={})

        cm = ConfusionMatrixMetric()
        cm.reset()
        cm.update(output, batch)
        cm.compute()
        return cm.value()

    def _panel(self, name: str, cm: np.ndarray, uar: float, f1: float) -> dict[str, Any]:
        return {
            "name": name,
            "uar": uar,
            "f1": f1,
            "cm": cm,
        }

    def _log_image(self, logger: Any, panels: list[dict[str, Any]], epoch: int) -> None:
        import matplotlib.pyplot as plt

        cols = min(3, max(1, len(panels)))
        rows = int(np.ceil(len(panels) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4.8 * cols, 4.2 * rows), squeeze=False)
        for index, ax in enumerate(axes.flatten()):
            if index >= len(panels):
                ax.axis("off")
                continue
            self._draw_cm(ax, panels[index])

        fig.suptitle(f"WSM {self.task} segment confusion matrices, epoch {epoch}", fontsize=14)
        fig.tight_layout()
        logger.log_artifact_bytes(
            self._fig_to_pdf_bytes(fig),
            artifact_path=self.artifact_path,
            filename=self.filename_template.format(epoch=epoch),
        )
        plt.close(fig)

    def _draw_cm(self, ax: Any, panel: dict[str, Any]) -> None:
        cm = panel["cm"]
        row_sum = cm.sum(axis=1, keepdims=True)
        cm_pct = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=np.float64), where=row_sum != 0.0) * 100.0
        ax.imshow(cm_pct, cmap="Blues", vmin=0.0, vmax=100.0)
        for i in range(len(self.class_names)):
            for j in range(len(self.class_names)):
                color = "white" if cm_pct[i, j] > 50.0 else "black"
                ax.text(j, i, f"{int(cm[i, j])}\n{cm_pct[i, j]:.1f}%", ha="center", va="center", color=color)

        ax.set(
            xticks=np.arange(len(self.class_names)),
            yticks=np.arange(len(self.class_names)),
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            xlabel="Predicted",
            ylabel="True",
            title=f"{panel['name']}\nUAR={panel['uar']:.4f}, F1={panel['f1']:.4f}",
        )

    @staticmethod
    def _fig_to_pdf_bytes(fig: Any) -> bytes:
        from io import BytesIO

        buffer = BytesIO()
        fig.savefig(buffer, format="pdf", bbox_inches="tight")
        return buffer.getvalue()


@CALLBACKS.register("wsm_segment_metrics_callback")
def wsm_segment_metrics_callback(context: Any | None = None, **params: Any) -> WSMSegmentMetricsCallback:
    params = dict(params)
    if context is not None:
        params.setdefault("task", context.get("data.task", "depression"))
        params.setdefault("task_names", list(context.get("data.task_names", ["depression", "parkinson"])))
        params.setdefault("class_names", list(context.get("data.class_names", ["healthy", "pathological"])))

    return WSMSegmentMetricsCallback(**params)


@CALLBACKS.register("wsm_audio_metrics_callback")
def wsm_audio_metrics_callback(context: Any | None = None, **params: Any) -> WSMSegmentMetricsCallback:
    return wsm_segment_metrics_callback(context=context, **params)
