# TASK-004D: Implement and Register the F1 Availability-Aware Shared-Representation Sparse A+V MTL Model

## Role

You are the implementing Codex. Execute only this task, update docs/PROGRESS_EN.md, commit on the required task branch, push to origin, then stop. Follow AGENTS.md.

Required branch:

    codex/task-004d

Do not create a training config.
Do not run a real training experiment.
Do not modify the accepted A+V DataModule.
Do not modify src/audio or src/video.
Do not start text/description work.
Do not add pseudo-labeling.

## Goal

Implement the second Stage 4 fusion baseline:

    F1 = availability-aware shared A+V representation
         + sparse two-head multitask classification

F1 must consume the accepted TASK-004A A+V Batch contract and return exactly two independent logits:

    [depression, parkinson]

The purpose of F1 is to isolate shared-representation gain relative to F0 while keeping the same pooled frozen modality inputs and sparse observed-label supervision.

## Fixed F1 Architecture

Inputs:

- audio_cls: [B,768]
- video: [B,Tv,512]
- video_mask: [B,Tv]
- modality_available: [B,2] ordered [audio,video]

Audio path:

    audio_input = zero unavailable rows
    a = AudioProjection(audio_input)
    a = zero unavailable rows after projection

Video path:

    v_raw = masked_mean(video, video_mask)
    v_raw = zero unavailable rows
    v = VideoProjection(v_raw)
    v = zero unavailable rows after projection

Shared fusion:

    joined = concat(a,v)                 # [B,2H]
    shared = SharedFusion(joined)        # [B,H]

Task heads:

    depression_logit = DepressionHead(shared)
    parkinson_logit  = ParkinsonHead(shared)

Final output:

    preds = stack([depression_logit, parkinson_logit]) -> [B,2]

There is NO late-fusion gate in F1.

## Availability Semantics

modality_available is required:

    [B,2] bool ordered [audio,video]

Rules:

- every sample must have at least one available modality;
- if audio is unavailable, its effective representation must be exactly zero before shared fusion;
- if video is unavailable, its effective representation must be exactly zero before shared fusion;
- changing an unavailable modality's finite raw values must not change final logits in eval mode;
- an available video sample must have at least one valid video temporal position;
- an unavailable video sample MAY have an all-false video_mask;
- neither modality available MUST raise ValueError.

Do not add an availability embedding or availability-bit feature in this baseline. Availability acts only as hard masking.

## Important Baseline Boundary

F1 MUST NOT contain:

- F0 late-fusion gates;
- per-modality disease logits;
- task_id/task_ids;
- task embeddings/tokens;
- corpus/split/Test-protocol IDs;
- temporal cross-attention;
- synchrony blocks;
- TACME/relation banks;
- pseudo-labeling;
- prototype logic;
- contrastive loss;
- text/description.

F1 is only:

    pooled modality features
    -> modality projections
    -> hard availability masking
    -> one shared fusion MLP
    -> two independent disease heads

## Required Reading

1. AGENTS.md
2. docs/PROJECT_REQUIREMENTS.md Sections 2-9, 11, 13-14
3. docs/PLAN.md Stage 4
4. docs/PROGRESS_EN.md through MANAGER-DECISION-019
5. docs/NEXT_TASK_EN.md
6. src/fusion/data/wsm_av_fusion_datamodule.py
7. src/fusion/models/av_f0_gated_late.py
8. src/common/loss/wsm_masked_sparse_loss.py
9. src/chimera_plugin.py

## Allowed Tracked Files

- src/fusion/models/av_f1_shared_mtl.py
- src/fusion/models/__init__.py
- src/chimera_plugin.py
- docs/PROGRESS_EN.md

No other tracked file may be modified.

## Registry Key

Register:

    wsm_av_f1_shared_mtl_model

Do not alter existing model registry keys.

## Fixed Constructor Defaults

Use:

    audio_feature_dim=768
    video_feature_dim=512
    hidden_dim=192
    fusion_hidden_dim=192
    dropout=0.2
    num_tasks=2

Require num_tasks == 2.

## Projection Blocks

Audio projection:

    LayerNorm(audio_feature_dim)
    Linear(audio_feature_dim, hidden_dim)
    GELU
    Dropout(dropout)

Video projection:

    LayerNorm(video_feature_dim)
    Linear(video_feature_dim, hidden_dim)
    GELU
    Dropout(dropout)

These match the F0 representation width/capacity before fusion.

## Video Pooling

Use masked mean over the cached video temporal features.

For available video rows:

- at least one video_mask position must be true.

For unavailable video rows:

- all-false video_mask is allowed;
- the effective pooled video representation must be zero.

Masked/padded values must not affect pooled output.

## Shared Fusion Trunk

Use exactly:

    LayerNorm(2 * hidden_dim)
    Linear(2 * hidden_dim, fusion_hidden_dim)
    GELU
    Dropout(dropout)
    Linear(fusion_hidden_dim, hidden_dim)
    GELU
    Dropout(dropout)

Output:

    shared_features [B,hidden_dim]

No residual branch is required.

## Disease Heads

Use two independent heads, one per disease:

    LayerNorm(hidden_dim)
    Linear(hidden_dim, hidden_dim)
    GELU
    Dropout(dropout)
    Linear(hidden_dim,1)

Return stacked logits [B,2].

Do not apply sigmoid to final logits.

## ModelOutput Contract

Return:

    preds -> [B,2]

aux must contain at least:

    features_audio           -> [B,H]
    features_video           -> [B,H]
    features_shared          -> [B,H]
    effective_audio_features -> [B,H]
    effective_video_features -> [B,H]
    task_logits = {
        "depression": preds[:,0],
        "parkinson": preds[:,1],
    }

No per-modality logits are required in F1.

The shared representation MUST be suitable for later gradient norm/cosine diagnostics without changing the forward contract.

## Context-Aware Factory

Factory must:

- read data.audio_feature_dim when available;
- read data.video_feature_dim when available;
- enforce data.num_tasks == 2 when present;
- default to 768/512/2 otherwise.

Update src/chimera_plugin.py with an explicit required import:

    fusion.models.av_f1_shared_mtl

Do not hide a required project import failure as an optional warning.

## Acceptance Criteria

Registry/model:

- MODELS contains wsm_av_f1_shared_mtl_model;
- plugin imports it without project warning;
- output.preds is [B,2];
- no task_id/task_ids input;
- no 3-class softmax;
- no late-fusion gate;
- no per-modality disease logits;
- no pseudo-labeling;
- no text/description;
- no TACME/cross-attention/relation bank.

Availability/masks:

- effective unavailable modality representation is exactly zero;
- both-available rows use both projected representations;
- audio-only rows zero the video representation;
- video-only rows zero the audio representation;
- neither-available rows raise ValueError;
- changing unavailable audio values does not change logits;
- changing unavailable video values does not change logits;
- changing padded/masked video positions does not change logits;
- available video with all-false mask raises ValueError;
- unavailable video with all-false mask is accepted.

Sparse loss/backward:

- wsm_masked_sparse_loss accepts output;
- masked NaN labels remain unsupervised;
- synthetic forward/loss/backward produces finite scalar loss;
- audio projection receives finite nonzero gradient on observed rows with audio available;
- video projection receives finite nonzero gradient on observed rows with video available;
- shared fusion trunk receives finite nonzero gradient;
- both disease heads receive finite nonzero gradients when their task has observed labels.

Real DataModule smoke:

- build WSMAVFusionDataModule;
- get one real train batch;
- run F1 forward + wsm_masked_sparse_loss + backward;
- output [B,2];
- finite loss/gradients;
- no task_id/task_ids input.

Safety:

- no training config;
- no real training run;
- no Test metrics;
- no DataModule changes;
- no F0 changes;
- src/audio unchanged;
- src/video unchanged;
- text/description remains deferred;
- python compilation passes;
- registry smoke passes;
- git diff --check passes;
- branch codex/task-004d committed and pushed;
- main/master untouched;
- tracked diff contains only the four allowed paths.

## Exact Verification Commands

Compile:

    python3 -m py_compile \
      src/fusion/models/av_f1_shared_mtl.py \
      src/fusion/models/__init__.py \
      src/chimera_plugin.py

Registry smoke:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python - <<'PY'
    import warnings
    from chimera_ml.core.registry import MODELS
    import chimera_plugin

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        chimera_plugin.register()

    failures = [
        str(x.message)
        for x in caught
        if "Failed to import" in str(x.message)
        and "fusion.models.av_f1_shared_mtl" in str(x.message)
    ]
    assert not failures, failures
    assert "wsm_av_f0_gated_late_model" in MODELS.keys()
    assert "wsm_av_f1_shared_mtl_model" in MODELS.keys()
    print("F1 registry smoke passed")
    PY

Synthetic contract smoke:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python - <<'PY'
    import math
    import torch

    from chimera_ml.core.batch import Batch
    from common.loss.wsm_masked_sparse_loss import WSMMaskedSparseLoss
    from fusion.models.av_f1_shared_mtl import WSMAVSharedMTLF1Model

    torch.manual_seed(23)

    model = WSMAVSharedMTLF1Model(
        audio_feature_dim=768,
        video_feature_dim=512,
        hidden_dim=64,
        fusion_hidden_dim=64,
        dropout=0.0,
    )

    audio = torch.randn(4,768)
    video = torch.randn(4,7,512)
    video_mask = torch.tensor([
        [1,1,1,1,1,0,0],
        [0,0,0,0,0,0,0],
        [1,1,1,0,0,0,0],
        [1,1,1,1,0,0,0],
    ], dtype=torch.bool)

    available = torch.tensor([
        [1,1],
        [1,0],
        [0,1],
        [1,1],
    ], dtype=torch.bool)

    targets = torch.tensor([
        [1.0, float("nan")],
        [0.0, float("nan")],
        [float("nan"), 1.0],
        [float("nan"), 0.0],
    ])
    observed = torch.tensor([
        [1,0],
        [1,0],
        [0,1],
        [0,1],
    ], dtype=torch.bool)

    def make_batch(a, v, vm, av):
        return Batch(
            inputs={"audio_cls": a, "video": v},
            targets=targets,
            masks={
                "video_mask": vm,
                "observed_mask": observed,
                "modality_available": av,
            },
            meta={},
        )

    batch = make_batch(audio, video, video_mask, available)

    model.eval()
    out = model(batch)

    assert tuple(out.preds.shape) == (4,2)
    assert tuple(out.aux["features_audio"].shape) == (4,64)
    assert tuple(out.aux["features_video"].shape) == (4,64)
    assert tuple(out.aux["features_shared"].shape) == (4,64)
    assert tuple(out.aux["effective_audio_features"].shape) == (4,64)
    assert tuple(out.aux["effective_video_features"].shape) == (4,64)

    # Hard availability zeroing.
    assert torch.equal(
        out.aux["effective_video_features"][1],
        torch.zeros_like(out.aux["effective_video_features"][1]),
    )
    assert torch.equal(
        out.aux["effective_audio_features"][2],
        torch.zeros_like(out.aux["effective_audio_features"][2]),
    )

    # Masked video padding invariance.
    video_pad_changed = video.clone()
    video_pad_changed[video_mask == 0] = 1e6
    out_pad = model(make_batch(audio, video_pad_changed, video_mask, available))
    assert torch.allclose(out.preds, out_pad.preds, atol=1e-5, rtol=1e-5)

    # Unavailable audio invariance.
    audio_changed = audio.clone()
    audio_changed[2] = 1e6
    out_audio = model(make_batch(audio_changed, video, video_mask, available))
    assert torch.allclose(out.preds[2], out_audio.preds[2], atol=1e-5, rtol=1e-5)

    # Unavailable video invariance; all-false mask is accepted for row 1.
    video_changed = video.clone()
    video_changed[1] = 1e6
    out_video = model(make_batch(audio, video_changed, video_mask, available))
    assert torch.allclose(out.preds[1], out_video.preds[1], atol=1e-5, rtol=1e-5)

    # Available video with no valid frames must fail.
    bad_mask = video_mask.clone()
    bad_mask[0] = False
    try:
        model(make_batch(audio, video, bad_mask, available))
    except ValueError:
        pass
    else:
        raise AssertionError("available video with all-false mask must fail")

    # Neither modality available must fail.
    bad_available = available.clone()
    bad_available[0] = False
    try:
        model(make_batch(audio, video, video_mask, bad_available))
    except ValueError:
        pass
    else:
        raise AssertionError("neither-available sample must fail")

    model.train()
    train_out = model(batch)
    loss = WSMMaskedSparseLoss()(train_out, batch)
    assert loss.ndim == 0 and math.isfinite(float(loss.detach()))
    loss.backward()

    grads = {
        name: p.grad
        for name,p in model.named_parameters()
        if p.requires_grad and p.grad is not None
    }

    assert any("audio_projection" in n and float(g.abs().sum()) > 0 for n,g in grads.items())
    assert any("video_projection" in n and float(g.abs().sum()) > 0 for n,g in grads.items())
    assert any("shared_fusion" in n and float(g.abs().sum()) > 0 for n,g in grads.items())
    assert any("task_heads.0" in n and float(g.abs().sum()) > 0 for n,g in grads.items())
    assert any("task_heads.1" in n and float(g.abs().sum()) > 0 for n,g in grads.items())
    assert all(torch.isfinite(g).all() for g in grads.values())

    print("F1 synthetic forward/loss/backward smoke passed", float(loss.detach()))
    PY

Real A+V DataModule smoke:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python - <<'PY'
    import math
    import torch

    from common.loss.wsm_masked_sparse_loss import WSMMaskedSparseLoss
    from fusion.data.wsm_av_fusion_datamodule import WSMAVFusionDataModule
    from fusion.models.av_f1_shared_mtl import WSMAVSharedMTLF1Model

    dm = WSMAVFusionDataModule(
        data_root="/media/maxim/Databases/WSM_NEW",
        audio_feature_cache_root="/media/maxim/Databases/WSM_NEW/features",
        video_cache_root="/media/maxim/Programs/Features/WSM/video_depart_v1_fullframe_fallback/cache",
        batch_size=4,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )

    batch = next(iter(dm.train_dataloader()))
    assert "task_id" not in batch.inputs and "task_ids" not in batch.inputs

    model = WSMAVSharedMTLF1Model(
        audio_feature_dim=dm.audio_feature_dim,
        video_feature_dim=512,
        hidden_dim=64,
        fusion_hidden_dim=64,
        dropout=0.0,
    )

    out = model(batch)
    loss = WSMMaskedSparseLoss()(out, batch)

    assert tuple(out.preds.shape) == (4,2)
    assert math.isfinite(float(loss.detach()))
    loss.backward()

    assert all(
        p.grad is None or torch.isfinite(p.grad).all()
        for p in model.parameters()
    )

    print("F1 real DataModule smoke passed", float(loss.detach()))
    PY

Then:

    git diff --check
    git diff -- src/audio
    git diff -- src/video
    git diff -- src/fusion/data/wsm_av_fusion_datamodule.py
    git diff -- src/fusion/models/av_f0_gated_late.py
    git status --short

Before commit inspect only:

    git diff -- \
      src/fusion/models/av_f1_shared_mtl.py \
      src/fusion/models/__init__.py \
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
- exact F1 architecture/equations;
- fixed constructor defaults;
- availability zeroing semantics;
- aux output shapes;
- synthetic mask/unavailable-modality invariance results;
- synthetic masked loss/backward result;
- real DataModule forward/loss/backward result;
- shared-trunk and both-head gradient checks;
- no task_id verification;
- confirmation no config/training/Test metrics;
- confirmation F0 and DataModule unchanged;
- confirmation text/description deferred;
- src/audio unchanged;
- src/video unchanged;
- Stage 4 status;
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

- branch codex/task-004d;
- implementation commit SHA;
- pushed-to-origin status;
- main/master untouched;
- registry key;
- output/aux shapes;
- availability-zeroing checks;
- synthetic and real DataModule loss/backward results;
- shared-trunk/both-head gradient result;
- no training/Test metrics;
- text/description deferred;
- src/audio unchanged;
- src/video unchanged.

Stop after this task.
