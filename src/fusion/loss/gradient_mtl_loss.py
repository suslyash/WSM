"""Faithful parameter-gradient MTL comparators for the DEV-only R4 extension."""
from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Iterable

import numpy as np
import torch
from scipy.optimize import minimize

from chimera_ml.core.registry import LOSSES
from fusion.loss.r4_ramps_balance_loss import WSMR4RampsBalanceLoss, _value

EPS = 1.0e-8


def _flat(parts: Iterable[torch.Tensor]) -> torch.Tensor:
    values = [part.detach().reshape(-1) for part in parts]
    if not values:
        return torch.empty(0)
    return torch.cat(values)


def _safe_cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
    if left.numel() == 0 or right.numel() == 0:
        return None
    norm = left.norm() * right.norm()
    if float(norm) <= EPS:
        return None
    return float(torch.clamp(torch.dot(left, right) / norm, -1.0, 1.0).item())


def pcgrad_two(grad_d: torch.Tensor, grad_p: torch.Tensor, eps: float = EPS) -> tuple[torch.Tensor, torch.Tensor]:
    """Two-task PCGrad projections, before the Equal-scale mean aggregation."""
    projected_d = grad_d.clone()
    projected_p = grad_p.clone()
    dot = torch.dot(grad_d, grad_p)
    if float(dot) < 0.0:
        projected_d = projected_d - dot * grad_p / (grad_p.square().sum() + eps)
        projected_p = projected_p - dot * grad_d / (grad_d.square().sum() + eps)
    return projected_d, projected_p


def cagrad_two(
    grad_d: torch.Tensor,
    grad_p: torch.Tensor,
    *,
    calpha: float = 0.5,
    rescale: int = 1,
    eps: float = EPS,
) -> tuple[torch.Tensor, np.ndarray, bool]:
    """Pinned CAGrad simplex solve, matching the official/LibMTL formula."""
    if rescale != 1:
        raise ValueError("TASK-010A freezes CAGrad rescale=1")
    grads = torch.stack([grad_d, grad_p])
    gram = (grads @ grads.T).detach().double().cpu()
    g0_norm = torch.sqrt(gram.mean() + eps)
    x_start = np.full(2, 0.5, dtype=np.float64)
    matrix = gram.numpy()
    constant = float(calpha * g0_norm.item() + eps)

    def objective(weights: np.ndarray) -> float:
        term = weights.reshape(1, -1) @ matrix @ x_start.reshape(-1, 1)
        norm_term = weights.reshape(1, -1) @ matrix @ weights.reshape(-1, 1)
        return float(term.sum() + constant * np.sqrt(norm_term.sum() + eps))

    result = minimize(
        objective,
        x_start,
        bounds=((0.0, 1.0), (0.0, 1.0)),
        constraints=({"type": "eq", "fun": lambda weights: 1.0 - float(weights.sum())},),
        method="SLSQP",
        options={"ftol": 1.0e-12, "maxiter": 100},
    )
    if not bool(result.success):
        raise RuntimeError(f"CAGrad simplex solve failed: {result.message}")
    weights = torch.as_tensor(result.x, device=grads.device, dtype=grads.dtype)
    gw = (grads * weights[:, None]).sum(0)
    lam = constant / (gw.norm() + eps)
    combined = (grads.mean(0) + lam.to(dtype=grads.dtype, device=grads.device) * gw) / (1.0 + calpha**2)
    return combined, result.x.copy(), bool(result.success)


def dbmtl_update(
    batch_d: torch.Tensor,
    batch_p: torch.Tensor,
    previous: torch.Tensor | None,
    *,
    beta: float = 0.9,
    beta_sigma: float = 0.0,
    step: int = 1,
    eps: float = EPS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Algorithm-1 two-task DB-MTL EMA and max-norm balancing."""
    if step <= 0:
        raise ValueError("DB-MTL step must be positive")
    batch = torch.stack([batch_d, batch_p])
    if previous is None:
        previous = torch.zeros_like(batch)
    decay = beta / (float(step) ** beta_sigma)
    ema = batch + decay * (previous - batch)
    norms = ema.norm(dim=1)
    alpha = norms.max() / (norms + eps)
    return (alpha[:, None] * ema).sum(0), ema


def _graph_parameters(losses: Iterable[torch.Tensor]) -> list[torch.Tensor]:
    """Collect trainable leaf parameters reachable from task-loss graphs."""
    found: dict[int, torch.Tensor] = {}
    # Keep the Python wrappers alive: PyTorch may otherwise reuse their ids while
    # traversing ``next_functions`` and incorrectly truncate the graph walk.
    visited: dict[int, Any] = {}
    pending = [loss.grad_fn for loss in losses if loss.grad_fn is not None]
    while pending:
        node = pending.pop()
        if node is None or id(node) in visited:
            continue
        visited[id(node)] = node
        variable = getattr(node, "variable", None)
        if isinstance(variable, torch.Tensor) and variable.is_leaf and variable.requires_grad:
            found[id(variable)] = variable
        for next_node, _ in getattr(node, "next_functions", ()):
            if next_node is not None:
                pending.append(next_node)
    return [found[key] for key in sorted(found)]


def _inject_parameter_gradients(
    reported_value: torch.Tensor,
    parameters: list[torch.Tensor],
    gradients: list[torch.Tensor | None],
) -> torch.Tensor:
    """Return a scalar whose backward writes the supplied parameter gradients.

    The trainer remains unchanged: AMP unscaling, clipping, and AdamW consume the
    resulting ``.grad`` values exactly where they do for ordinary losses.
    """
    if len(parameters) != len(gradients):
        raise ValueError("parameter/gradient length mismatch")

    class _ParameterGradientInjection(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, value: torch.Tensor, *params: torch.Tensor) -> torch.Tensor:
            ctx.gradients = gradients
            return value.detach()

        @staticmethod
        def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
            injected = [None if grad is None else grad_output * grad for grad in ctx.gradients]
            return (None, *injected)

    return _ParameterGradientInjection.apply(reported_value, *parameters)


class WSMGradientMTLLoss(WSMR4RampsBalanceLoss):
    """Frozen R4 task objectives with a source-grounded parameter-gradient combiner."""

    METHODS = {"pcgrad", "cagrad", "gradnorm", "dbmtl"}

    def __init__(
        self,
        method: str,
        *,
        pcgrad_seed: int = 0,
        calpha: float = 0.5,
        rescale: int = 1,
        gradnorm_alpha: float = 1.5,
        gradnorm_weight_lr: float = 0.025,
        db_beta: float = 0.9,
        db_beta_sigma: float = 0.0,
        **params: Any,
    ) -> None:
        if method not in self.METHODS:
            raise ValueError(f"method must be one of {sorted(self.METHODS)}")
        super().__init__(mode="equal", training_task="both", **params)
        if calpha != 0.5 or rescale != 1:
            raise ValueError("TASK-010A freezes CAGrad calpha=0.5 and rescale=1")
        if gradnorm_alpha != 1.5 or gradnorm_weight_lr != 0.025:
            raise ValueError("TASK-010A freezes GradNorm alpha=1.5 and weight lr=0.025")
        if db_beta != 0.9 or db_beta_sigma != 0.0:
            raise ValueError("TASK-010A freezes DB-MTL beta=0.9 and beta_sigma=0.0")
        self.method = method
        self.pcgrad_seed = int(pcgrad_seed)
        self._pcgrad_rng = random.Random(self.pcgrad_seed)
        self.calpha = float(calpha)
        self.rescale = int(rescale)
        self.gradnorm_alpha = float(gradnorm_alpha)
        self.gradnorm_weight_lr = float(gradnorm_weight_lr)
        self.db_beta = float(db_beta)
        self.db_beta_sigma = float(db_beta_sigma)
        self._gradnorm_weights: torch.nn.Parameter | None = None
        self._gradnorm_optimizer: torch.optim.Adam | None = None
        self._gradnorm_initial_losses: torch.Tensor | None = None
        self._db_buffer: torch.Tensor | None = None
        self._db_step = 0
        self.reset_epoch_diagnostics()

    def reset_epoch_diagnostics(self) -> None:
        self._epoch_stats: defaultdict[str, list[float]] = defaultdict(list)

    @property
    def gradnorm_weights(self) -> torch.Tensor:
        if self._gradnorm_weights is None:
            return torch.ones(2, dtype=torch.float32)
        return self._gradnorm_weights.detach()

    def _record(self, **values: float | None) -> None:
        for key, value in values.items():
            if value is not None and math.isfinite(float(value)):
                self._epoch_stats[key].append(float(value))

    def epoch_diagnostics(self) -> dict[str, float]:
        return {f"mtl/{key}": float(sum(values) / len(values)) for key, values in self._epoch_stats.items() if values}

    def _task_objectives(self, output: Any, batch: Any) -> tuple[list[torch.Tensor], list[bool]]:
        logits = output.preds
        if logits.ndim != 2 or tuple(logits.shape[1:]) != (2,):
            raise ValueError("R4 logits must have shape [B, 2]")
        targets = _value(batch, "targets").to(device=logits.device, dtype=logits.dtype)
        observed = _value(batch, "observed_mask").to(device=logits.device, dtype=torch.bool)
        accept = _value(batch, "pseudo_accept_mask").to(device=logits.device, dtype=torch.bool)
        pseudo = _value(batch, "pseudo_targets").to(device=logits.device, dtype=logits.dtype).detach()
        reliability = _value(batch, "pseudo_reliability").to(device=logits.device, dtype=logits.dtype).detach()
        if bool((accept & observed).any()):
            raise ValueError("pseudo acceptance overlaps observed truth")
        neutral = ~accept
        if bool(torch.isfinite(pseudo[neutral]).any()) or bool((reliability[neutral] != 0).any()):
            raise ValueError("rejected/observed pseudo fields must be NaN/zero")
        audio_aux = output.aux["audio_aux_logits"]
        video_aux = output.aux["video_aux_logits"]
        audio_valid = output.aux["audio_aux_valid"].to(device=logits.device, dtype=torch.bool)
        video_valid = output.aux["video_aux_valid"].to(device=logits.device, dtype=torch.bool)
        objectives, active = [], []
        for task in range(2):
            objective, task_active = self._components(
                logits, task, targets, observed, accept, pseudo, reliability,
                audio_aux, video_aux, audio_valid, video_valid,
            )
            objectives.append(objective)
            active.append(task_active)
        if not any(active):
            raise ValueError("R4 batch has neither observed nor accepted-pseudo supervision")
        return objectives, active

    def _ensure_gradnorm_state(self, reference: torch.Tensor) -> None:
        if self._gradnorm_weights is None:
            self._gradnorm_weights = torch.nn.Parameter(torch.ones(2, device=reference.device, dtype=reference.dtype))
            self._gradnorm_optimizer = torch.optim.Adam([self._gradnorm_weights], lr=self.gradnorm_weight_lr)

    def _gradnorm_combine(
        self,
        grad_d: list[torch.Tensor | None],
        grad_p: list[torch.Tensor | None],
        shared: list[int],
        objectives: list[torch.Tensor],
    ) -> list[torch.Tensor | None]:
        self._ensure_gradnorm_state(objectives[0])
        assert self._gradnorm_weights is not None and self._gradnorm_optimizer is not None
        weights_before = self._gradnorm_weights.detach().clone()
        shared_d = _flat(grad_d[index] for index in shared if grad_d[index] is not None)
        shared_p = _flat(grad_p[index] for index in shared if grad_p[index] is not None)
        raw_norms = torch.stack([shared_d.norm(), shared_p.norm()]).detach()
        losses = torch.stack([objective.detach() for objective in objectives])
        if self._gradnorm_initial_losses is None:
            self._gradnorm_initial_losses = losses.clone()
            target = torch.ones_like(raw_norms)
            auxiliary = 0.0
            rates = torch.ones_like(raw_norms)
        else:
            rates = losses / self._gradnorm_initial_losses.clamp_min(EPS)
            rates = rates / rates.mean().clamp_min(EPS)
            scaled_norms = self._gradnorm_weights * raw_norms
            average = scaled_norms.mean()
            target = (average * rates.pow(self.gradnorm_alpha)).detach()
            auxiliary_tensor = (scaled_norms - target).abs().sum()
            self._gradnorm_optimizer.zero_grad(set_to_none=True)
            auxiliary_tensor.backward()
            self._gradnorm_optimizer.step()
            with torch.no_grad():
                self._gradnorm_weights.clamp_(min=EPS)
                self._gradnorm_weights.mul_(2.0 / self._gradnorm_weights.sum().clamp_min(EPS))
            auxiliary = float(auxiliary_tensor.detach().item())
        self._record(
            weight_depression=float(weights_before[0].item()),
            weight_parkinson=float(weights_before[1].item()),
            raw_loss_depression=float(losses[0].item()),
            raw_loss_parkinson=float(losses[1].item()),
            rate_depression=float(rates[0].item()),
            rate_parkinson=float(rates[1].item()),
            norm_depression=float(raw_norms[0].item()),
            norm_parkinson=float(raw_norms[1].item()),
            target_depression=float(target[0].item()),
            target_parkinson=float(target[1].item()),
            auxiliary=auxiliary,
        )
        combined: list[torch.Tensor | None] = []
        for d_grad, p_grad in zip(grad_d, grad_p, strict=True):
            if d_grad is not None and p_grad is not None:
                combined.append(weights_before[0] * d_grad + weights_before[1] * p_grad)
            elif d_grad is not None:
                combined.append(weights_before[0] * d_grad)
            elif p_grad is not None:
                combined.append(weights_before[1] * p_grad)
            else:
                combined.append(None)
        return combined

    @staticmethod
    def _split(flattened: torch.Tensor, gradients: list[torch.Tensor | None], indices: list[int]) -> dict[int, torch.Tensor]:
        result: dict[int, torch.Tensor] = {}
        offset = 0
        for index in indices:
            reference = gradients[index]
            assert reference is not None
            width = reference.numel()
            result[index] = flattened[offset:offset + width].reshape_as(reference)
            offset += width
        if offset != flattened.numel():
            raise RuntimeError("gradient split did not consume its vector")
        return result

    def _combine_pair(
        self,
        grad_d: list[torch.Tensor | None],
        grad_p: list[torch.Tensor | None],
        shared: list[int],
        objectives: list[torch.Tensor],
    ) -> list[torch.Tensor | None]:
        if self.method == "gradnorm":
            return self._gradnorm_combine(grad_d, grad_p, shared, objectives)
        vector_d = _flat(grad_d[index] for index in shared if grad_d[index] is not None)
        vector_p = _flat(grad_p[index] for index in shared if grad_p[index] is not None)
        pre_cosine = _safe_cosine(vector_d, vector_p)
        conflict = pre_cosine is not None and pre_cosine < 0.0
        if self.method == "pcgrad":
            # The official shuffle is deterministic under the run seed. With two
            # tasks there is exactly one opponent, so it cannot alter the result.
            self._pcgrad_rng.shuffle([0, 1])
            transformed_d, transformed_p = pcgrad_two(vector_d, vector_p)
            shared_vector = 0.5 * (transformed_d + transformed_p)
            self._record(
                pre_cosine=pre_cosine,
                post_cosine=_safe_cosine(transformed_d, transformed_p),
                conflict=float(conflict),
                pre_norm_depression=float(vector_d.norm().item()),
                pre_norm_parkinson=float(vector_p.norm().item()),
                post_norm_depression=float(transformed_d.norm().item()),
                post_norm_parkinson=float(transformed_p.norm().item()),
            )
        elif self.method == "cagrad":
            shared_vector, weights, converged = cagrad_two(
                vector_d, vector_p, calpha=self.calpha, rescale=self.rescale,
            )
            mean_vector = 0.5 * (vector_d + vector_p)
            self._record(
                pre_cosine=pre_cosine,
                conflict=float(conflict),
                combined_norm=float(shared_vector.norm().item()),
                mean_norm=float(mean_vector.norm().item()),
                adjustment_norm=float((shared_vector - mean_vector).norm().item()),
                solver_converged=float(converged),
                simplex_weight_depression=float(weights[0]),
                simplex_weight_parkinson=float(weights[1]),
            )
        elif self.method == "dbmtl":
            transformed_d = torch.autograd.grad(
                torch.log(objectives[0] + EPS), [], allow_unused=True
            )
            raise AssertionError("DB-MTL transformed gradients are computed before _combine_pair")
        else:
            raise RuntimeError(f"unexpected method {self.method}")
        split = self._split(shared_vector, grad_d, shared)
        combined: list[torch.Tensor | None] = []
        for index, (d_grad, p_grad) in enumerate(zip(grad_d, grad_p, strict=True)):
            if index in split:
                combined.append(split[index])
            elif d_grad is not None:
                combined.append(d_grad)
            elif p_grad is not None:
                combined.append(p_grad)
            else:
                combined.append(None)
        return combined

    def _dbmtl_combine(
        self,
        parameters: list[torch.Tensor],
        objectives: list[torch.Tensor],
        shared: list[int],
    ) -> list[torch.Tensor | None]:
        transformed = [torch.log(objective + EPS) for objective in objectives]
        grad_d = list(torch.autograd.grad(transformed[0], parameters, retain_graph=True, allow_unused=True))
        grad_p = list(torch.autograd.grad(transformed[1], parameters, retain_graph=True, allow_unused=True))
        vector_d = _flat(grad_d[index] for index in shared if grad_d[index] is not None)
        vector_p = _flat(grad_p[index] for index in shared if grad_p[index] is not None)
        self._db_step += 1
        shared_vector, self._db_buffer = dbmtl_update(
            vector_d, vector_p, self._db_buffer, beta=self.db_beta,
            beta_sigma=self.db_beta_sigma, step=self._db_step,
        )
        assert self._db_buffer is not None
        ema_d, ema_p = self._db_buffer
        norms = torch.stack([ema_d.norm(), ema_p.norm()])
        normalized = norms.max() / (norms + EPS)
        self._record(
            pre_cosine=_safe_cosine(vector_d, vector_p),
            post_cosine=_safe_cosine(ema_d, ema_p),
            conflict=float((_safe_cosine(vector_d, vector_p) or 0.0) < 0.0),
            ema_norm_depression=float(norms[0].item()),
            ema_norm_parkinson=float(norms[1].item()),
            normalized_norm_depression=float((normalized[0] * norms[0]).item()),
            normalized_norm_parkinson=float((normalized[1] * norms[1]).item()),
        )
        split = self._split(shared_vector, grad_d, shared)
        combined: list[torch.Tensor | None] = []
        for index, (d_grad, p_grad) in enumerate(zip(grad_d, grad_p, strict=True)):
            if index in split:
                combined.append(split[index])
            elif d_grad is not None:
                combined.append(d_grad)
            elif p_grad is not None:
                combined.append(p_grad)
            else:
                combined.append(None)
        return combined

    def __call__(self, output: Any, batch: Any) -> torch.Tensor:
        objectives, active = self._task_objectives(output, batch)
        if active == [True, False]:
            return objectives[0]
        if active == [False, True]:
            return objectives[1]
        reported = 0.5 * (objectives[0] + objectives[1])
        if not output.preds.requires_grad or not getattr(output.preds, "requires_grad", False):
            return reported
        parameters = _graph_parameters(objectives)
        if not parameters:
            raise RuntimeError("gradient MTL could not find trainable parameters in task objectives")
        if self.method == "dbmtl":
            raw_d = list(torch.autograd.grad(objectives[0], parameters, retain_graph=True, allow_unused=True))
            raw_p = list(torch.autograd.grad(objectives[1], parameters, retain_graph=True, allow_unused=True))
            shared = [index for index, pair in enumerate(zip(raw_d, raw_p, strict=True)) if pair[0] is not None and pair[1] is not None]
            combined = self._dbmtl_combine(parameters, objectives, shared)
        else:
            grad_d = list(torch.autograd.grad(objectives[0], parameters, retain_graph=True, allow_unused=True))
            grad_p = list(torch.autograd.grad(objectives[1], parameters, retain_graph=True, allow_unused=True))
            shared = [index for index, pair in enumerate(zip(grad_d, grad_p, strict=True)) if pair[0] is not None and pair[1] is not None]
            combined = self._combine_pair(grad_d, grad_p, shared, objectives)
        d_only = sum(d is not None and p is None for d, p in zip(raw_d if self.method == "dbmtl" else grad_d, raw_p if self.method == "dbmtl" else grad_p, strict=True))
        p_only = sum(d is None and p is not None for d, p in zip(raw_d if self.method == "dbmtl" else grad_d, raw_p if self.method == "dbmtl" else grad_p, strict=True))
        neither = sum(d is None and p is None for d, p in zip(raw_d if self.method == "dbmtl" else grad_d, raw_p if self.method == "dbmtl" else grad_p, strict=True))
        self._record(parameters_depression_only=float(d_only), parameters_parkinson_only=float(p_only), parameters_both=float(len(shared)), parameters_neither=float(neither))
        return _inject_parameter_gradients(reported, parameters, combined)


@LOSSES.register("wsm_gradient_mtl_loss")
def wsm_gradient_mtl_loss(context: Any | None = None, **params: Any) -> WSMGradientMTLLoss:
    del context
    return WSMGradientMTLLoss(**params)
