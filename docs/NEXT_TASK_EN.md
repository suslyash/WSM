# TASK-004F: Implement and Register the F2 Availability-Aware Task-Specific Directed A+V Relation-Bank Model

## Role

You are the implementing Codex. Execute only this task, update docs/PROGRESS_EN.md, commit on the required task branch, push to origin, then stop. Follow AGENTS.md.

Required branch:

    codex/task-004f

Do not create a training config.
Do not run a real training experiment.
Do not modify the accepted A+V DataModule.
Do not modify F0 or F1.
Do not modify src/audio or src/video.
Do not add pseudo-labeling, flow matching, PAGB, or text/description.

## Goal

Implement the third Stage 4 fusion baseline:

    F2 = F1 shared representation
         + shared directed A<->V relation bank
         + task-specific expert gating
         + two independent sparse disease heads

F2 must consume the accepted TASK-004A A+V Batch contract and return exactly:

    [depression, parkinson]

The purpose is to isolate task-aware directed fusion relative to the already measured F1 baseline.

## Manager-Fixed F2 Contract

F2 is a minimal pooled-feature TACME-like adaptation.

It MUST retain the same raw inputs and base representation path as F1:

- audio_cls [B,768]
- masked-mean video [B,512]
- modality-specific projection to H
- hard availability zeroing
- F1-style shared_fusion trunk over concat(audio,video)

Then add exactly two SHARED directed relation experts:

    expert 0 = audio_to_video
    expert 1 = video_to_audio

The relation expert bank is shared by both disease tasks.

Each disease has its own learned gate over this same two-expert bank.

No external task_id/task_ids are allowed.

## Required Reading

1. AGENTS.md
2. docs/PROJECT_REQUIREMENTS.md Sections 2-9, 11, 13-14
3. docs/PLAN.md Stage 4
4. docs/PROGRESS_EN.md through MANAGER-DECISION-021
5. docs/NEXT_TASK_EN.md
6. src/fusion/data/wsm_av_fusion_datamodule.py
7. src/fusion/models/av_f1_shared_mtl.py
8. src/fusion/models/av_f0_gated_late.py
9. src/common/loss/wsm_masked_sparse_loss.py
10. src/chimera_plugin.py

## Allowed Tracked Files

- src/fusion/models/av_f2_task_aware_directed.py
- src/fusion/models/__init__.py
- src/chimera_plugin.py
- docs/PROGRESS_EN.md

No other tracked file may be modified.

## Registry Key

Register:

    wsm_av_f2_task_aware_directed_model

Do not alter existing model registry keys.

## Fixed Constructor Defaults

Use:

    audio_feature_dim=768
    video_feature_dim=512
    hidden_dim=192
    fusion_hidden_dim=192
    relation_hidden_dim=192
    dropout=0.2
    num_tasks=2

Require num_tasks == 2.

## 1. Base F1 Path — Preserve Exactly

Use the same effective-input semantics as F1.

Audio:

    audio_input = zero unavailable audio rows
    a = AudioProjection(audio_input)
    a_eff = zero unavailable audio rows after projection

Video:

    v_raw = masked_mean(video, video_mask)
    v_raw = zero unavailable video rows
    v = VideoProjection(v_raw)
    v_eff = zero unavailable video rows after projection

Shared F1 base:

    base_shared = SharedFusion(concat(a_eff, v_eff))

Projection blocks and SharedFusion MUST match F1 architecture/capacity exactly.

Do not import/reuse F1 by calling its forward; implement the F2 module self-contained so its aux contract is explicit.

## 2. Shared Directed Relation Expert Bank

Implement a reusable local module such as DirectedRelationExpert.

For ordered query q and context c, both [B,H]:

    relation_input = concat(q,c)

Use:

    LayerNorm(2H)
    Linear(2H, relation_hidden_dim)
    GELU
    Dropout
    Linear(relation_hidden_dim, H)
    Dropout

Then add a query residual and normalize:

    expert(q,c) = LayerNorm(q + RelationMLP(concat(q,c)))

Create exactly two independently parameterized experts:

    audio_to_video = expert(a_eff, v_eff)
    video_to_audio = expert(v_eff, a_eff)

The direction is encoded by:

- ordered query/context arguments;
- separate expert parameters.

Do not create task-specific expert banks.

## 3. Relation Validity

modality_available:

    [B,2] bool ordered [audio,video]

A directed cross-modal relation requires BOTH modalities.

Therefore for each sample:

- both modalities available:
    relation_valid = [true,true]
- audio-only:
    relation_valid = [false,false]
- video-only:
    relation_valid = [false,false]
- neither:
    raise ValueError

For invalid relation rows:

- effective relation expert outputs exposed to downstream task fusion MUST be exactly zero;
- task relation weights MUST be exactly zero;
- final task representation falls back to the F1 base_shared path.

Changing raw values of an unavailable modality must not change final logits.

## 4. Task-Specific Expert Gates

For each task t in:

    depression
    parkinson

use a separate gate module shared across the two expert positions:

    Linear(H,H)
    LayerNorm(H)
    GELU
    Dropout(dropout)
    Linear(H,1)

Apply the task gate independently to each expert output:

    scores_t = [gate_t(E_A2V), gate_t(E_V2A)]  -> [B,2]

For rows where both relations are valid:

    weights_t = softmax(scores_t, dim=expert)

For rows with no valid cross-modal relation:

    weights_t = [0,0]

Required task expert-weight tensor:

    [B,2,2]

dimensions:

    batch, task, expert

task order:

    [depression, parkinson]

expert order:

    [audio_to_video, video_to_audio]

For both-available rows, weights sum exactly to 1 for each task.

## 5. Task-Specific Relation Fusion

For each task:

    relation_t =
        w_t,A2V * E_A2V
        + w_t,V2A * E_V2A

Then:

    task_feature_t =
        LayerNorm(base_shared + relation_t)

Use an independent LayerNorm(H) per task.

No learnable residual scalar is added in this baseline.

For single-modality rows relation_t is exactly zero, so task_feature_t is simply the normalized F1 base representation.

## 6. Disease Heads

Use two independent heads with the same capacity as F1:

    LayerNorm(H)
    Linear(H,H)
    GELU
    Dropout(dropout)
    Linear(H,1)

Return:

    preds [B,2]

ordered:

    [depression, parkinson]

Do not apply sigmoid to final logits.

## 7. ModelOutput Contract

Return ModelOutput with:

    preds -> [B,2]

aux must contain at least:

    features_audio              -> [B,H]
    features_video              -> [B,H]
    effective_audio_features    -> [B,H]
    effective_video_features    -> [B,H]
    features_shared             -> [B,H]
    relation_experts            -> [B,2,H]
    relation_valid              -> [B,2] bool
    task_expert_weights         -> [B,2,2]
    task_relation_features      -> [B,2,H]
    task_features               -> [B,2,H]
    task_logits = {
        "depression": preds[:,0],
        "parkinson": preds[:,1],
    }

The aux contract must make later expert-weight and gradient diagnostics possible without changing forward.

## 8. Important F2 Boundary

F2 MUST NOT contain:

- external task_id/task_ids input;
- task/corpus/split/Test protocol metadata input;
- task-specific expert banks;
- pseudo-labeling;
- teacher/student logic;
- flow matching;
- PAGB;
- learned loss balancing;
- auxiliary relation loss;
- contrastive loss;
- prototype logic;
- temporal cross-attention over raw sequences;
- new temporal encoders;
- text/description;
- semantic label embeddings.

F2 uses ONLY:

    existing pooled frozen A+V features
    + F1 base path
    + shared ordered relation experts
    + task-specific expert gates
    + observed sparse BCE

## 9. Availability and Mask Semantics

Preserve F1 semantics:

- every sample needs at least one available modality;
- available video requires >=1 true video_mask position;
- unavailable video may have all-false video_mask;
- masked video values do not affect output;
- unavailable audio/video raw values do not affect output;
- neither modality available raises ValueError.

For single-modality rows:

- relation_valid is all false;
- relation_experts after validity masking are exact zero;
- task_expert_weights are exact zero;
- task_relation_features are exact zero.

## 10. Context-Aware Factory

Factory must:

- read data.audio_feature_dim when available;
- read data.video_feature_dim when available;
- enforce data.num_tasks == 2 when present;
- default to 768/512/2 otherwise.

Update src/chimera_plugin.py with explicit required import:

    fusion.models.av_f2_task_aware_directed

Do not hide a project import failure as an optional warning.

## Acceptance Criteria

Registry/model:

- MODELS contains wsm_av_f2_task_aware_directed_model;
- F0/F1 registry keys remain intact;
- plugin imports F2 without project warning;
- preds [B,2];
- no task_id/task_ids;
- no 3-class softmax;
- no pseudo-labeling;
- no flow matching/PAGB;
- no text/description.

Architecture:

- F1-equivalent modality projections and shared_fusion exist;
- exactly two shared directed relation experts exist;
- experts are independently parameterized;
- there is NOT one expert bank per task;
- task-specific gates are independent;
- relation_experts [B,2,H];
- task_expert_weights [B,2,2];
- task_relation_features [B,2,H];
- task_features [B,2,H].

Gate/availability semantics:

- both-available weights are finite, in [0,1], and sum to 1 per task;
- single-modality rows have exactly zero relation weights;
- single-modality rows have exactly zero relation features;
- relation_valid correctly marks both experts valid only when A+V are both available;
- neither available raises ValueError.

Directionality:

- swapping the ordered expert inputs while keeping expert parameters fixed must not be treated as the same operation;
- verify audio_to_video and video_to_audio are separate modules/parameter sets;
- on generic unequal synthetic a/v features, their outputs should not be identically equal.

Invariance:

- masked video padding changes do not change logits;
- unavailable audio changes do not change logits;
- unavailable video changes do not change logits.

Sparse loss/backward:

- wsm_masked_sparse_loss accepts output;
- masked NaN labels remain unsupervised;
- finite scalar loss;
- both modality projections get finite nonzero gradients;
- shared_fusion gets finite nonzero gradient;
- BOTH directed experts get finite nonzero gradients on a both-available synthetic batch with observed labels;
- both task gates get finite nonzero gradients;
- both task heads get finite nonzero gradients.

Real DataModule smoke:

- build WSMAVFusionDataModule;
- use a real train batch;
- F2 forward + masked sparse loss + backward;
- preds [B,2];
- finite gradients;
- task_expert_weights finite and normalized because current canonical rows are both-modal;
- no task_id input.

Safety:

- no training config;
- no real training run;
- no Test metrics;
- no DataModule changes;
- no F0/F1 changes;
- src/audio unchanged;
- src/video unchanged;
- text/description remains deferred;
- python compilation passes;
- registry smoke passes;
- git diff --check passes;
- branch codex/task-004f committed and pushed;
- main/master untouched;
- tracked diff contains only the four allowed paths.

## Exact Verification Commands

Compile:

    python3 -m py_compile \
      src/fusion/models/av_f2_task_aware_directed.py \
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
        and "fusion.models.av_f2_task_aware_directed" in str(x.message)
    ]
    assert not failures, failures
    assert "wsm_av_f0_gated_late_model" in MODELS.keys()
    assert "wsm_av_f1_shared_mtl_model" in MODELS.keys()
    assert "wsm_av_f2_task_aware_directed_model" in MODELS.keys()
    print("F2 registry smoke passed")
    PY

Synthetic F2 contract smoke:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python - <<'PY'
    import math
    import torch

    from chimera_ml.core.batch import Batch
    from common.loss.wsm_masked_sparse_loss import WSMMaskedSparseLoss
    from fusion.models.av_f2_task_aware_directed import WSMAVTaskAwareDirectedF2Model

    torch.manual_seed(31)

    model = WSMAVTaskAwareDirectedF2Model(
        audio_feature_dim=768,
        video_feature_dim=512,
        hidden_dim=64,
        fusion_hidden_dim=64,
        relation_hidden_dim=64,
        dropout=0.0,
    )

    audio = torch.randn(6,768)
    video = torch.randn(6,7,512)
    video_mask = torch.tensor([
        [1,1,1,1,1,0,0],
        [1,1,1,1,1,1,1],
        [1,1,1,0,0,0,0],
        [0,0,0,0,0,0,0],
        [1,1,1,1,0,0,0],
        [1,1,1,1,1,0,0],
    ], dtype=torch.bool)

    available = torch.tensor([
        [1,1],
        [1,1],
        [0,1],
        [1,0],
        [1,1],
        [1,1],
    ], dtype=torch.bool)

    targets = torch.tensor([
        [1.0, float("nan")],
        [0.0, float("nan")],
        [float("nan"), 1.0],
        [float("nan"), 0.0],
        [1.0, float("nan")],
        [float("nan"), 1.0],
    ])
    observed = ~torch.isnan(targets)

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

    assert tuple(out.preds.shape) == (6,2)
    assert tuple(out.aux["features_audio"].shape) == (6,64)
    assert tuple(out.aux["features_video"].shape) == (6,64)
    assert tuple(out.aux["features_shared"].shape) == (6,64)
    assert tuple(out.aux["relation_experts"].shape) == (6,2,64)
    assert tuple(out.aux["relation_valid"].shape) == (6,2)
    assert tuple(out.aux["task_expert_weights"].shape) == (6,2,2)
    assert tuple(out.aux["task_relation_features"].shape) == (6,2,64)
    assert tuple(out.aux["task_features"].shape) == (6,2,64)

    both = available.all(dim=1)
    single = ~both

    weights = out.aux["task_expert_weights"]
    assert torch.isfinite(weights).all()
    assert torch.all(weights >= 0) and torch.all(weights <= 1)
    assert torch.allclose(
        weights[both].sum(dim=-1),
        torch.ones_like(weights[both, :, 0]),
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.equal(weights[single], torch.zeros_like(weights[single]))

    rel = out.aux["task_relation_features"]
    assert torch.equal(rel[single], torch.zeros_like(rel[single]))

    valid = out.aux["relation_valid"]
    assert bool(valid[both].all())
    assert not bool(valid[single].any())

    experts = out.aux["relation_experts"]
    assert not torch.allclose(experts[0,0], experts[0,1])

    # Separate expert parameter sets.
    a2v_ids = {id(p) for p in model.audio_to_video_expert.parameters()}
    v2a_ids = {id(p) for p in model.video_to_audio_expert.parameters()}
    assert a2v_ids and v2a_ids and a2v_ids.isdisjoint(v2a_ids)

    # Masked padding invariance.
    video_pad_changed = video.clone()
    video_pad_changed[~video_mask] = 1e6
    out_pad = model(make_batch(audio, video_pad_changed, video_mask, available))
    assert torch.allclose(out.preds, out_pad.preds, atol=1e-5, rtol=1e-5)

    # Unavailable audio invariance row 2.
    audio_changed = audio.clone()
    audio_changed[2] = 1e6
    out_audio = model(make_batch(audio_changed, video, video_mask, available))
    assert torch.allclose(out.preds[2], out_audio.preds[2], atol=1e-5, rtol=1e-5)

    # Unavailable video invariance row 3.
    video_changed = video.clone()
    video_changed[3] = 1e6
    out_video = model(make_batch(audio, video_changed, video_mask, available))
    assert torch.allclose(out.preds[3], out_video.preds[3], atol=1e-5, rtol=1e-5)

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
    assert all(torch.isfinite(g).all() for g in grads.values())
    assert any("audio_projection" in n and float(g.abs().sum()) > 0 for n,g in grads.items())
    assert any("video_projection" in n and float(g.abs().sum()) > 0 for n,g in grads.items())
    assert any("shared_fusion" in n and float(g.abs().sum()) > 0 for n,g in grads.items())
    assert any("audio_to_video_expert" in n and float(g.abs().sum()) > 0 for n,g in grads.items())
    assert any("video_to_audio_expert" in n and float(g.abs().sum()) > 0 for n,g in grads.items())
    assert any("task_gates.0" in n and float(g.abs().sum()) > 0 for n,g in grads.items())
    assert any("task_gates.1" in n and float(g.abs().sum()) > 0 for n,g in grads.items())
    assert any("task_heads.0" in n and float(g.abs().sum()) > 0 for n,g in grads.items())
    assert any("task_heads.1" in n and float(g.abs().sum()) > 0 for n,g in grads.items())

    print("F2 synthetic forward/loss/backward smoke passed", float(loss.detach()))
    PY

Real A+V DataModule smoke:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python - <<'PY'
    import math
    import torch

    from common.loss.wsm_masked_sparse_loss import WSMMaskedSparseLoss
    from fusion.data.wsm_av_fusion_datamodule import WSMAVFusionDataModule
    from fusion.models.av_f2_task_aware_directed import WSMAVTaskAwareDirectedF2Model

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

    model = WSMAVTaskAwareDirectedF2Model(
        audio_feature_dim=dm.audio_feature_dim,
        video_feature_dim=512,
        hidden_dim=64,
        fusion_hidden_dim=64,
        relation_hidden_dim=64,
        dropout=0.0,
    )

    out = model(batch)
    loss = WSMMaskedSparseLoss()(out, batch)
    assert tuple(out.preds.shape) == (4,2)
    assert math.isfinite(float(loss.detach()))
    assert torch.isfinite(out.aux["task_expert_weights"]).all()
    assert torch.allclose(
        out.aux["task_expert_weights"].sum(dim=-1),
        torch.ones_like(out.aux["task_expert_weights"][...,0]),
        atol=1e-6,
        rtol=1e-6,
    )
    loss.backward()
    assert all(
        p.grad is None or torch.isfinite(p.grad).all()
        for p in model.parameters()
    )

    print("F2 real DataModule smoke passed", float(loss.detach()))
    PY

Then:

    git diff --check
    git diff -- src/audio
    git diff -- src/video
    git diff -- src/fusion/data/wsm_av_fusion_datamodule.py
    git diff -- src/fusion/models/av_f0_gated_late.py
    git diff -- src/fusion/models/av_f1_shared_mtl.py
    git status --short

Before commit inspect only:

    git diff -- \
      src/fusion/models/av_f2_task_aware_directed.py \
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
- exact F2 relation/expert/gating equations;
- fixed constructor defaults;
- confirmation base modality/shared path matches F1;
- expert parameter-sharing structure;
- expert order and task order;
- aux output shapes;
- availability/relation-validity semantics;
- both/single modality gate checks;
- directionality and separate-parameter checks;
- synthetic masked loss/backward result;
- gradient checks for projections/shared trunk/both experts/both gates/both heads;
- real DataModule forward/loss/backward result;
- no task_id verification;
- confirmation no config/training/Test metrics;
- confirmation no pseudo-labeling/flow matching/PAGB;
- confirmation F0/F1/DataModule unchanged;
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

- branch codex/task-004f;
- implementation commit SHA;
- pushed-to-origin status;
- main/master untouched;
- registry key;
- output/aux shapes;
- relation expert order;
- task gate-weight semantics;
- directionality check;
- availability fallback checks;
- synthetic and real DataModule loss/backward results;
- both-expert/both-gate gradient result;
- no training/Test metrics;
- no pseudo-labeling/flow matching/PAGB;
- text/description deferred;
- src/audio unchanged;
- src/video unchanged.

Stop after this task.
