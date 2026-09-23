# TASK-002F: Implement and Register the V1 DEPART-Like Temporal Video Model Contract

## Role

You are the implementing Codex. Execute only this task, update docs/PROGRESS_EN.md, commit on the required task branch, push to origin, then stop. Follow AGENTS.md.

Required branch:

    codex/task-002f

Do not run full-dataset feature extraction, do not train a model, do not create a training config, do not implement V2/prototypes, and do not inspect Test predictions or Test metrics.

## Goal

Implement the Stage 2 V1 video classifier contract on top of the accepted cached-feature shape:

    [B, 60, 512] frozen CLIP body-ROI features
      -> learned projection
      -> temporal Transformer
      -> masked temporal pooling
      -> two independent binary logits [B, 2]

The two output columns are ordered:

    [depression, parkinson]

The model must directly produce both disease logits for every sample. It must not use task_id selection and must not reduce the problem to a 3-class softmax.

This task is model integration and smoke verification only. No training run is authorized.

## Required Reading

1. AGENTS.md
2. docs/PROJECT_REQUIREMENTS.md Sections 2-10 and 13-14
3. Stage 2 in docs/PLAN.md
4. docs/PROGRESS_EN.md through TASK-002E
5. docs/NEXT_TASK_EN.md
6. src/video/features/clip_video_features.py
7. src/common/loss/wsm_masked_sparse_loss.py
8. src/audio/models/audio_mamba_segment.py for Chimera model conventions only
9. src/chimera_plugin.py

## Allowed Tracked Files

- src/video/models/__init__.py
- src/video/models/depart_v1.py
- src/chimera_plugin.py
- docs/PROGRESS_EN.md

Creating src/video/models is allowed.

No other tracked file may be modified.

## Fixed V1 Model Contract

Registry key:

    wsm_video_depart_v1_model

Inputs:

- batch.inputs["video"]: float tensor [B, T, D]
- batch.get_masks("video_mask"): bool-compatible tensor [B, T]

Expected accepted cache contract:

- T = 60 for the current V1 cache;
- D = 512 for pinned CLIP ViT-B/32 features.

Outputs:

- ModelOutput.preds: logits [B, 2], ordered [depression, parkinson]
- ModelOutput.aux["features"]: pooled video representation [B, H]
- ModelOutput.aux["task_logits"]:
  - "depression": logits[:, 0]
  - "parkinson": logits[:, 1]

Do not apply sigmoid inside the model; the registered masked BCE loss expects logits.

## Architecture Requirements

Implement src/video/models/depart_v1.py using only video-local/PyTorch components.

Do not import fusion modules.

Required structure:

1. input validation;
2. feature projection:
   - LayerNorm(video_feature_dim)
   - Linear(video_feature_dim, hidden_dim)
   - GELU
   - Dropout
3. learned positional embeddings for up to sequence_steps=60;
4. TransformerEncoder with batch_first=true and norm_first=true;
5. post-encoder LayerNorm;
6. masked mean pooling across valid temporal positions only;
7. two independent scalar binary heads, one per disease.

Recommended defaults for this project contract:

    video_feature_dim=512
    hidden_dim=192
    num_layers=2
    num_heads=4
    ff_mult=4
    dropout=0.2
    sequence_steps=60
    num_tasks=2

These are implementation defaults for the initial V1 contract, not a hyperparameter sweep.

Each scalar head may be:

    LayerNorm(H)
    Dropout
    Linear(H, H)
    GELU
    Dropout
    Linear(H, 1)

Stack the two scalar outputs into [B,2].

## Mask and Validation Semantics

The model must validate:

- video is rank 3 [B,T,D];
- mask is rank 2 [B,T];
- batch/time dimensions match;
- feature dimension equals configured video_feature_dim;
- T <= sequence_steps;
- every sample has at least one valid frame.

Masked temporal positions:

- must be excluded from Transformer attention via key-padding mask;
- must be excluded from temporal pooling;
- must not influence logits.

Do not silently turn an all-invalid sample into a valid sample. Raise a clear ValueError.

The model must work when some but not all of the 60 temporal positions are invalid.

## Chimera Registration

Register:

    wsm_video_depart_v1_model

Use BaseModel and return ModelOutput.

Context-aware factory behavior:

- if context exists, allow video_feature_dim from data.video_feature_dim;
- default to 512 if no context value is present;
- num_tasks must be 2 for this project contract.

Update src/chimera_plugin.py with an explicit import of the model registration module.

A project-module import failure must not be hidden as an optional warning.

## No DataModule or Config Yet

Do not add:

- cache dataset/datamodule;
- YAML training config;
- callbacks;
- optimizer;
- metrics;
- training/inference scripts.

Those are later atomic tasks after this model contract is accepted.

## Acceptance Criteria

- MODELS registry contains wsm_video_depart_v1_model;
- plugin import has no project-module warning for the new model;
- model accepts [B,60,512] + bool [B,60] mask;
- output.preds shape is [B,2];
- outputs are two independent logits, not softmax probabilities;
- no task_id input/selection exists;
- masked positions do not affect outputs in eval-mode invariance smoke;
- partial masks work;
- all-invalid sample raises ValueError;
- ModelOutput.aux contains pooled features and separate depression/parkinson logits;
- masked sparse loss accepts model output and [B,2] partial targets;
- forward/loss/backward produces finite scalar loss and finite model gradients;
- unknown targets remain masked NaNs and are not supervised;
- no cache extraction;
- no training/model selection;
- no Test rows/predictions/metrics;
- src/audio unchanged;
- python compilation passes;
- registry smoke passes;
- forward/loss/backward smoke passes;
- git diff --check passes;
- branch codex/task-002f committed and pushed;
- main/master untouched;
- tracked diff contains only the four allowed paths.

## Exact Verification Commands

Run from repository root.

Compile:

    python3 -m py_compile       src/video/models/__init__.py       src/video/models/depart_v1.py       src/chimera_plugin.py

Registry smoke:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python - <<'PY'
    import warnings

    from chimera_ml.core.registry import MODELS
    import chimera_plugin

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        chimera_plugin.register()

    project_warnings = [
        str(item.message)
        for item in caught
        if "Failed to import" in str(item.message)
        and "video.models.depart_v1" in str(item.message)
    ]
    assert not project_warnings, project_warnings
    assert "wsm_video_depart_v1_model" in MODELS.keys()
    print("V1 video model registry smoke passed")
    PY

Forward/loss/backward and masking smoke:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python - <<'PY'
    import math
    import torch

    from common.loss.wsm_masked_sparse_loss import WSMMaskedSparseLoss
    from video.models.depart_v1 import WSMVideoDepartV1Model

    class FakeBatch:
        def __init__(self, video, video_mask):
            self.inputs = {"video": video}
            self._video_mask = video_mask

        def get_masks(self, name):
            assert name == "video_mask"
            return self._video_mask

    torch.manual_seed(7)

    model = WSMVideoDepartV1Model(
        video_feature_dim=512,
        hidden_dim=64,
        num_layers=1,
        num_heads=4,
        ff_mult=2,
        dropout=0.0,
        sequence_steps=60,
    )
    model.eval()

    video = torch.randn(4, 60, 512)
    mask = torch.ones(4, 60, dtype=torch.bool)
    mask[0, 55:] = False
    mask[1, 40:] = False

    batch = FakeBatch(video, mask)
    output = model(batch)

    assert tuple(output.preds.shape) == (4, 2)
    assert torch.isfinite(output.preds).all()
    assert tuple(output.aux["features"].shape) == (4, 64)
    assert tuple(output.aux["task_logits"]["depression"].shape) == (4,)
    assert tuple(output.aux["task_logits"]["parkinson"].shape) == (4,)

    # Masked temporal positions must not affect logits.
    changed = video.clone()
    changed[~mask] = 1e6
    output_changed = model(FakeBatch(changed, mask))
    assert torch.allclose(output.preds, output_changed.preds, atol=1e-5, rtol=1e-5)

    try:
        bad_mask = mask.clone()
        bad_mask[0] = False
        model(FakeBatch(video, bad_mask))
    except ValueError:
        pass
    else:
        raise AssertionError("all-invalid video sample must fail")

    model.train()
    output = model(batch)

    targets = torch.tensor(
        [
            [1.0, float("nan")],
            [0.0, float("nan")],
            [float("nan"), 1.0],
            [float("nan"), 0.0],
        ],
        dtype=torch.float32,
    )
    observed = torch.tensor(
        [
            [True, False],
            [True, False],
            [False, True],
            [False, True],
        ],
        dtype=torch.bool,
    )

    loss_fn = WSMMaskedSparseLoss()
    loss = loss_fn.compute_from_tensors(output.preds, targets, observed)

    assert loss.ndim == 0
    assert math.isfinite(float(loss.detach()))

    loss.backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)

    print("V1 video forward/loss/backward smoke passed", float(loss.detach()))
    PY

Then:

    git diff --check
    git diff -- src/audio
    git status --short

Before commit inspect only:

    git diff --       src/video/models/__init__.py       src/video/models/depart_v1.py       src/chimera_plugin.py       docs/PROGRESS_EN.md

After commit/push:

    git status --short
    git rev-parse --abbrev-ref HEAD
    git rev-parse HEAD
    git diff --stat origin/main...HEAD
    git diff --name-only origin/main...HEAD

## Required PROGRESS_EN Update

Append TASK-002F facts without erasing prior evidence.

Record:

- branch;
- implementation commit SHA;
- push result;
- registry key;
- exact V1 architecture/defaults;
- explicit confirmation that outputs are [depression, parkinson] logits [B,2];
- confirmation no task_id selection or 3-class softmax exists;
- partial-mask and all-invalid behavior;
- masking invariance smoke result;
- forward/loss/backward result and shapes;
- exact commands/results;
- no cache extraction;
- zero Test rows/metrics;
- no training/model selection;
- src/audio unchanged;
- Stage 2 remains partial;
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

- branch codex/task-002f;
- implementation commit SHA;
- pushed-to-origin status;
- main/master untouched;
- diff summary;
- registry key;
- model output shape;
- forward/loss/backward result;
- mask invariance result;
- no cache extraction;
- no Test rows/metrics;
- no training;
- src/audio unchanged.

Stop after this task.
