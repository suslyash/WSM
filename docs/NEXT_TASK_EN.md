# TASK-002N: Implement and Register the Prototype-Aware V2 Video Model Contract

## Role

You are the implementing Codex. Execute only this task, update docs/PROGRESS_EN.md, commit on the required task branch, push to origin, then stop. Follow AGENTS.md.

Required branch:

    codex/task-002n

Do not create a V2 training config, do not run training, do not change preprocessing/cache/DataModule semantics, and do not modify src/audio.

## Goal

Implement the second and final Stage 2 video family:

    V2 = accepted V1 temporal video encoder
         + task-specific class prototypes
         + classwise prototype/MLP gating

The model must preserve the exact project output contract:

    logits [B,2] ordered [depression, parkinson]

Unknown labels remain handled only by the existing masked sparse loss.

This task is model-contract implementation and synthetic verification only.

## Manager-Fixed Minimal V2 Contract

For each task t in {depression, parkinson}:

1. reuse the V1 video encoder through pooled representation h in R^H;
2. maintain two learned class prototypes:
   - p_t,0 for negative class;
   - p_t,1 for positive class;
3. L2-normalize h and both prototypes;
4. compute cosine similarities:
   - s_t,0 = cos(h, p_t,0)
   - s_t,1 = cos(h, p_t,1)
5. define the prototype logit:
   - proto_logit_t = prototype_scale * (s_t,1 - s_t,0)
6. compute a V1-style task MLP logit:
   - mlp_logit_t
7. compute a learned scalar gate:
   - gate_t = sigmoid(gate_mlp_t(h))
8. final task logit:
   - logit_t = (1 - gate_t) * mlp_logit_t + gate_t * proto_logit_t

Stack task logits into [B,2].

This is the fixed initial V2 design. Do not add attention over prototypes, extra task tokens, cross-task fusion, pseudo-labeling, or a contrastive loss in this task.

## Required Reading

1. AGENTS.md
2. docs/PROJECT_REQUIREMENTS.md Sections 2, 3, 7, 8, 11, 13, 14
3. Stage 2 in docs/PLAN.md
4. docs/PROGRESS_EN.md through MANAGER-DECISION-014
5. docs/NEXT_TASK_EN.md
6. src/video/models/depart_v1.py
7. src/common/loss/wsm_masked_sparse_loss.py
8. src/chimera_plugin.py

## Allowed Tracked Files

- src/video/models/depart_v2.py
- src/video/models/__init__.py
- src/chimera_plugin.py
- docs/PROGRESS_EN.md

No other tracked file may be modified.

## Registry Key

Register:

    wsm_video_depart_v2_model

Do not alter the V1 registry key or implementation.

## Fixed Constructor Defaults

Use:

    video_feature_dim=512
    hidden_dim=192
    num_layers=2
    num_heads=4
    ff_mult=4
    dropout=0.2
    sequence_steps=60
    num_tasks=2
    prototype_scale=10.0
    gate_hidden_dim=64

num_tasks must remain exactly 2.

## Encoder Contract

The temporal encoder path must remain architecturally equivalent to V1:

- input LayerNorm;
- Linear to hidden_dim;
- GELU;
- Dropout;
- learned positional embeddings;
- TransformerEncoder;
- output LayerNorm;
- masked mean pooling.

You MAY factor/reuse code only if it does not modify depart_v1.py in this task.

Simplest acceptable implementation: duplicate the small V1 encoder structure into depart_v2.py.

Do not refactor V1.

## Prototype Parameters

Store learned prototypes as a parameter with shape:

    [2,2,H]

ordered:

    task 0 = depression
    task 1 = parkinson
    class 0 = negative
    class 1 = positive

Initialize prototypes with a small normal distribution, e.g. std=0.02.

During forward:

- normalize prototypes along H;
- normalize pooled features along H;
- cosine similarities shape must be [B,2,2].

Prototype parameters must receive gradients through the existing masked sparse BCE because final logits depend on prototype evidence.

## MLP Branch

For each task use an independent scalar MLP head equivalent in capacity to V1:

    LayerNorm(H)
    Dropout
    Linear(H,H)
    GELU
    Dropout
    Linear(H,1)

Return stacked MLP logits [B,2].

## Gate Branch

For each task use an independent scalar gate MLP:

    LayerNorm(H)
    Linear(H, gate_hidden_dim)
    GELU
    Dropout
    Linear(gate_hidden_dim,1)
    Sigmoid

Gate output shape:

    [B,2]

Every gate value must lie strictly within [0,1].

Do not condition gate inputs on:

- task_id metadata;
- corpus;
- labels;
- observed_mask;
- split;
- Test protocol identity.

Gate may use only the pooled video representation.

## Final Output Contract

Return ModelOutput with:

    preds: final logits [B,2]

and aux containing at least:

    features                 -> pooled [B,H]
    task_logits              -> depression/parkinson final logits
    mlp_logits               -> [B,2]
    prototype_logits         -> [B,2]
    prototype_similarities   -> [B,2,2]
    prototype_gates          -> [B,2]
    normalized_prototypes    -> [2,2,H]

Do not apply sigmoid to final logits.

## Contrastive-Ablation Preparation

Do NOT add contrastive loss now.

The aux contract above must be sufficient for a later controlled prototype contrastive loss/ablation to access:

- normalized pooled representation;
- normalized task/class prototypes;
- classwise similarities.

If needed, also expose:

    normalized_features -> [B,H]

Do not use labels inside forward.

## Mask/Shape Semantics

Preserve V1 validation semantics:

- video [B,T,512];
- video_mask [B,T];
- 1 <= T <= 60;
- every sample must have at least one valid frame;
- masked temporal positions excluded from attention/pooling;
- changing masked temporal feature values must not change logits in eval mode.

## Chimera Factory

Context-aware factory must:

- read data.video_feature_dim if available;
- enforce data.num_tasks == 2 if present;
- default video_feature_dim=512;
- default num_tasks=2.

Update src/chimera_plugin.py with explicit import:

    video.models.depart_v2

Required project import failures must not be hidden as optional warnings.

## Acceptance Criteria

- MODELS contains wsm_video_depart_v2_model;
- V1 registry remains intact;
- plugin imports V2 with no project warning;
- input [B,60,512] + bool mask works;
- output.preds [B,2];
- no task_id selection;
- no 3-class softmax;
- prototype tensor [2,2,H];
- prototype similarities [B,2,2];
- MLP logits [B,2];
- prototype logits [B,2];
- gates [B,2] in [0,1];
- final logits exactly equal the documented convex gate blend;
- mask invariance passes;
- all-invalid sample raises ValueError;
- wsm_masked_sparse_loss accepts output;
- forward/loss/backward produces finite scalar loss;
- trainable prototype parameters receive finite nonzero gradients on a synthetic batch with observed labels for both tasks;
- at least one gate parameter receives finite gradient;
- masked NaN labels remain unsupervised;
- no training config;
- no training run;
- no Test metric computation;
- src/audio unchanged;
- python compilation passes;
- registry smoke passes;
- git diff --check passes;
- branch codex/task-002n committed and pushed;
- main/master untouched;
- tracked diff contains only the four allowed paths.

## Exact Verification Commands

Compile:

    python3 -m py_compile \
      src/video/models/depart_v2.py \
      src/video/models/__init__.py \
      src/chimera_plugin.py

Registry smoke:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python - <<'PY'
    import warnings
    from chimera_ml.core.registry import MODELS
    import chimera_plugin

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        chimera_plugin.register()

    warnings_v2 = [
        str(x.message)
        for x in caught
        if "Failed to import" in str(x.message)
        and "video.models.depart_v2" in str(x.message)
    ]
    assert not warnings_v2, warnings_v2
    assert "wsm_video_depart_v1_model" in MODELS.keys()
    assert "wsm_video_depart_v2_model" in MODELS.keys()
    print("V2 registry smoke passed")
    PY

Synthetic V2 smoke:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python - <<'PY'
    import math
    import torch

    from common.loss.wsm_masked_sparse_loss import WSMMaskedSparseLoss
    from video.models.depart_v2 import WSMVideoDepartV2Model

    class FakeBatch:
        def __init__(self, video, video_mask, targets=None, observed=None):
            self.inputs = {"video": video}
            self._video_mask = video_mask
            self.targets = targets
            self._observed = observed

        def get_masks(self, name):
            if name == "video_mask":
                return self._video_mask
            if name == "observed_mask":
                return self._observed
            raise KeyError(name)

    torch.manual_seed(11)

    model = WSMVideoDepartV2Model(
        video_feature_dim=512,
        hidden_dim=64,
        num_layers=1,
        num_heads=4,
        ff_mult=2,
        dropout=0.0,
        sequence_steps=60,
        prototype_scale=10.0,
        gate_hidden_dim=16,
    )

    video = torch.randn(4, 60, 512)
    mask = torch.ones(4, 60, dtype=torch.bool)
    mask[0, 49:] = False
    mask[1, 37:] = False

    model.eval()
    batch = FakeBatch(video, mask)
    out = model(batch)

    assert tuple(out.preds.shape) == (4,2)
    assert tuple(out.aux["features"].shape) == (4,64)
    assert tuple(out.aux["mlp_logits"].shape) == (4,2)
    assert tuple(out.aux["prototype_logits"].shape) == (4,2)
    assert tuple(out.aux["prototype_similarities"].shape) == (4,2,2)
    assert tuple(out.aux["prototype_gates"].shape) == (4,2)
    assert tuple(out.aux["normalized_prototypes"].shape) == (2,2,64)

    gates = out.aux["prototype_gates"]
    assert torch.all(gates >= 0) and torch.all(gates <= 1)

    expected = (
        (1.0 - gates) * out.aux["mlp_logits"]
        + gates * out.aux["prototype_logits"]
    )
    assert torch.allclose(out.preds, expected, atol=1e-6, rtol=1e-6)

    changed = video.clone()
    changed[~mask] = 1e6
    changed_out = model(FakeBatch(changed, mask))
    assert torch.allclose(out.preds, changed_out.preds, atol=1e-5, rtol=1e-5)

    bad = mask.clone()
    bad[0] = False
    try:
        model(FakeBatch(video, bad))
    except ValueError:
        pass
    else:
        raise AssertionError("all-invalid sample must fail")

    targets = torch.tensor([
        [1.0, float("nan")],
        [0.0, float("nan")],
        [float("nan"), 1.0],
        [float("nan"), 0.0],
    ])
    observed = torch.tensor([
        [True, False],
        [True, False],
        [False, True],
        [False, True],
    ])

    model.train()
    train_batch = FakeBatch(video, mask, targets, observed)
    train_out = model(train_batch)
    loss = WSMMaskedSparseLoss()(train_out, train_batch)

    assert loss.ndim == 0
    assert math.isfinite(float(loss.detach()))
    loss.backward()

    assert model.class_prototypes.grad is not None
    assert torch.isfinite(model.class_prototypes.grad).all()
    assert float(model.class_prototypes.grad.abs().sum()) > 0

    gate_grads = [
        p.grad
        for name, p in model.named_parameters()
        if "gate" in name and p.grad is not None
    ]
    assert gate_grads
    assert all(torch.isfinite(g).all() for g in gate_grads)

    print("V2 prototype/gate forward-loss-backward smoke passed", float(loss.detach()))
    PY

Then:

    git diff --check
    git diff -- src/audio
    git status --short

Before commit inspect only:

    git diff -- \
      src/video/models/depart_v2.py \
      src/video/models/__init__.py \
      src/chimera_plugin.py \
      docs/PROGRESS_EN.md

After commit/push:

    git status --short
    git rev-parse --abbrev-ref HEAD
    git rev-parse HEAD
    git diff --stat origin/main...HEAD
    git diff --name-only origin/main...HEAD

## Required PROGRESS_EN Update

Record:

- branch;
- implementation commit SHA;
- push result;
- registry key;
- exact prototype tensor/order;
- prototype logit equation;
- gate equation;
- final blend equation;
- fixed defaults;
- aux contract;
- mask-invariance result;
- forward/loss/backward result;
- prototype-gradient result;
- gate-gradient result;
- confirmation no contrastive loss yet;
- confirmation no config/training/Test metrics;
- src/audio unchanged;
- Stage 2 remains in V2 implementation;
- recommended next atomic step only.

## Required Handoff

Respond in English using exactly:

1. Outcome
2. Changed files
3. Verification
4. Plan status
5. Blockers and risks
6. Next atomic step

Explicitly include:

- branch codex/task-002n;
- implementation commit SHA;
- pushed-to-origin status;
- main/master untouched;
- registry key;
- output/aux shapes;
- prototype gradient result;
- gate gradient result;
- no contrastive loss;
- no training/Test metrics;
- src/audio unchanged.

Stop after this task.
