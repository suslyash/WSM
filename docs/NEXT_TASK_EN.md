# TASK-004B: Implement and Register the F0 Availability-Aware Gated A+V Late-Fusion Model

## Role

You are the implementing Codex. Execute only this task, update docs/PROGRESS_EN.md, commit on the required task branch, push to origin, then stop. Follow AGENTS.md.

Required branch:

    codex/task-004b

Do not create a training config.
Do not run a real training experiment.
Do not modify the accepted A+V DataModule contract.
Do not modify src/audio or src/video.
Do not start text/description work.

## Goal

Implement the first Stage 4 fusion baseline:

    F0 = availability-aware task-wise gated late fusion

F0 must consume the accepted TASK-004A A+V Batch contract and return exactly two independent logits:

    [depression, parkinson]

This is intentionally a simple lower-bound fusion model.

## Fixed F0 Architecture

Inputs:

- audio_cls: [B,768]
- video: [B,Tv,512]
- video_mask: [B,Tv]
- modality_available: [B,2] ordered [audio,video]

Audio representation:

    a = AudioProjection(audio_cls)

Video representation:

    v_raw = masked_mean(video, video_mask)
    v = VideoProjection(v_raw)

Use the same hidden width for both projected representations.

For each task t in {depression, parkinson}:

    audio_logit_t = AudioHead_t(a)
    video_logit_t = VideoHead_t(v)

Compute one scalar gate from both modality representations:

    raw_gate_t = sigmoid(Gate_t(concat(a,v)))

Interpret raw_gate_t as the audio weight before availability correction.

Availability-aware normalized weights:

- if both audio and video are available:
      w_audio = raw_gate
      w_video = 1 - raw_gate
- if only audio is available:
      w_audio = 1
      w_video = 0
- if only video is available:
      w_audio = 0
      w_video = 1
- if neither modality is available:
      raise ValueError

Final task logit:

    logit_t = w_audio_t * audio_logit_t
            + w_video_t * video_logit_t

Stack task logits into:

    [B,2]

## Important Baseline Boundary

F0 MUST NOT contain:

- temporal cross-attention;
- synchrony blocks;
- relation banks;
- TACME-style directed experts;
- a shared multimodal fusion trunk after concatenation;
- task embeddings;
- task_id/task_ids input;
- corpus/split/Test-protocol IDs;
- pseudo-labeling;
- contrastive loss;
- text/description;
- prototype logic.

F0 is only pooled frozen features + modality-specific heads + task-specific scalar late-fusion gates.

## Required Reading

1. AGENTS.md
2. docs/PROJECT_REQUIREMENTS.md Sections 2-9, 11, 13-14
3. docs/PLAN.md Stage 4
4. docs/PROGRESS_EN.md through MANAGER-DECISION-017
5. docs/NEXT_TASK_EN.md
6. src/fusion/data/wsm_av_fusion_datamodule.py
7. src/common/loss/wsm_masked_sparse_loss.py
8. src/video/models/depart_v1.py for two-head output style only
9. src/chimera_plugin.py

## Allowed Tracked Files

- src/fusion/models/av_f0_gated_late.py
- src/fusion/models/__init__.py
- src/chimera_plugin.py
- docs/PROGRESS_EN.md

No other tracked file may be modified.

## Registry Key

Register:

    wsm_av_f0_gated_late_model

Do not alter existing model registry keys.

## Fixed Constructor Defaults

Use:

    audio_feature_dim=768
    video_feature_dim=512
    hidden_dim=192
    gate_hidden_dim=64
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

No temporal encoder in F0.

## Unimodal Task Heads

For each task and each modality, use an independent scalar head:

    LayerNorm(hidden_dim)
    Dropout(dropout)
    Linear(hidden_dim, hidden_dim)
    GELU
    Dropout(dropout)
    Linear(hidden_dim, 1)

Required collections:

- 2 audio task heads;
- 2 video task heads.

## Gate Heads

For each task:

    LayerNorm(2 * hidden_dim)
    Linear(2 * hidden_dim, gate_hidden_dim)
    GELU
    Dropout(dropout)
    Linear(gate_hidden_dim, 1)
    Sigmoid

Gate input is only:

    concat(audio_representation, video_representation)

Do not use labels, observed_mask, task identity metadata, corpus, split, or evaluation protocol.

## Mask Semantics

Video:

- video [B,T,512];
- video_mask [B,T] bool;
- every sample marked video-available must have at least one valid video position;
- masked positions must not affect pooled video representation.

Audio:

- F0 uses audio_cls, not audio temporal tokens;
- audio_cls must be finite [B,768].

modality_available:

    [B,2] bool, ordered [audio,video]

Validation:

- at least one modality must be available for every sample;
- if audio_available=false, audio branch may still be numerically computed but its final weight MUST be exactly 0;
- if video_available=false, video branch may still be numerically computed but its final weight MUST be exactly 0;
- changing an unavailable modality's values must not change final logits in eval mode.

## ModelOutput Contract

Return:

    preds -> [B,2]

aux must contain at least:

    features_audio      -> [B,H]
    features_video      -> [B,H]
    audio_logits        -> [B,2]
    video_logits        -> [B,2]
    raw_audio_gates     -> [B,2]
    fusion_weights      -> [B,2,2]

fusion_weights last dimension order:

    [audio, video]

Also expose:

    task_logits = {
        "depression": preds[:,0],
        "parkinson": preds[:,1],
    }

No sigmoid on final logits.

## Context-Aware Factory

Factory must:

- read data.audio_feature_dim when available;
- read data.video_feature_dim when available;
- enforce data.num_tasks == 2 when present;
- default to 768/512/2 otherwise.

Update src/chimera_plugin.py with an explicit required import:

    fusion.models.av_f0_gated_late

Do not hide a project import failure as an optional warning.

## Acceptance Criteria

Registry/model:

- MODELS contains wsm_av_f0_gated_late_model;
- plugin imports it without project warning;
- F0 output is [B,2];
- no task_id/task_ids input;
- no 3-class softmax;
- no text/description;
- no pseudo-labeling;
- no temporal fusion/cross-attention/TACME block.

Weights/gates:

- raw_audio_gates [B,2], each in [0,1];
- fusion_weights [B,2,2];
- fusion_weights sum to 1 over modality dimension;
- both-available rows use [g,1-g];
- audio-only rows use exactly [1,0];
- video-only rows use exactly [0,1];
- neither-available rows raise ValueError;
- final logits exactly equal weighted audio/video logits.

Mask invariance:

- changing padded/masked video values does not change output;
- changing audio values on audio-unavailable rows does not change output;
- changing video values on video-unavailable rows does not change output.

Sparse loss:

- wsm_masked_sparse_loss accepts F0 output;
- masked NaN targets remain unsupervised;
- synthetic forward/loss/backward gives finite scalar loss;
- audio projection/head parameters receive finite nonzero gradient when observed rows use audio;
- video projection/head parameters receive finite nonzero gradient when observed rows use video;
- at least one gate parameter receives finite gradient on both-available rows.

Real DataModule smoke:

- build WSMAVFusionDataModule;
- get one real train batch;
- run F0 forward + masked sparse loss + backward;
- finite loss and gradients;
- batch does not contain task_id input.

Safety:

- no training config;
- no real training run;
- no Test metrics;
- no DataModule changes;
- src/audio unchanged;
- src/video unchanged;
- python compilation passes;
- registry smoke passes;
- git diff --check passes;
- branch codex/task-004b committed and pushed;
- main/master untouched;
- tracked diff contains only the four allowed paths.

## Exact Verification Commands

Compile:

    python3 -m py_compile \
      src/fusion/models/av_f0_gated_late.py \
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
        and "fusion.models.av_f0_gated_late" in str(x.message)
    ]
    assert not failures, failures
    assert "wsm_av_f0_gated_late_model" in MODELS.keys()
    print("F0 registry smoke passed")
    PY

Synthetic contract smoke:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python - <<'PY'
    import math
    import torch

    from chimera_ml.core.batch import Batch
    from common.loss.wsm_masked_sparse_loss import WSMMaskedSparseLoss
    from fusion.models.av_f0_gated_late import WSMAVGatedLateFusionF0Model

    torch.manual_seed(17)

    model = WSMAVGatedLateFusionF0Model(
        audio_feature_dim=768,
        video_feature_dim=512,
        hidden_dim=64,
        gate_hidden_dim=16,
        dropout=0.0,
    )

    audio = torch.randn(4,768)
    video = torch.randn(4,7,512)
    video_mask = torch.tensor([
        [1,1,1,1,1,0,0],
        [1,1,1,1,1,1,1],
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

    batch = Batch(
        inputs={
            "audio_cls": audio,
            "video": video,
        },
        targets=targets,
        masks={
            "video_mask": video_mask,
            "observed_mask": observed,
            "modality_available": available,
        },
        meta={},
    )

    model.eval()
    out = model(batch)

    assert tuple(out.preds.shape) == (4,2)
    assert tuple(out.aux["audio_logits"].shape) == (4,2)
    assert tuple(out.aux["video_logits"].shape) == (4,2)
    assert tuple(out.aux["raw_audio_gates"].shape) == (4,2)
    assert tuple(out.aux["fusion_weights"].shape) == (4,2,2)

    weights = out.aux["fusion_weights"]
    assert torch.allclose(weights.sum(dim=-1), torch.ones_like(weights[...,0]))
    assert torch.equal(weights[1], torch.tensor([[1.0,0.0],[1.0,0.0]]))
    assert torch.equal(weights[2], torch.tensor([[0.0,1.0],[0.0,1.0]]))

    expected = (
        weights[...,0] * out.aux["audio_logits"]
        + weights[...,1] * out.aux["video_logits"]
    )
    assert torch.allclose(out.preds, expected, atol=1e-6, rtol=1e-6)

    # Masked video padding invariance.
    changed = video.clone()
    changed[~video_mask] = 1e6
    changed_batch = Batch(
        inputs={"audio_cls": audio, "video": changed},
        targets=targets,
        masks={
            "video_mask": video_mask,
            "observed_mask": observed,
            "modality_available": available,
        },
        meta={},
    )
    changed_out = model(changed_batch)
    assert torch.allclose(out.preds, changed_out.preds, atol=1e-5, rtol=1e-5)

    # Unavailable audio invariance for row 2.
    audio_changed = audio.clone()
    audio_changed[2] = 1e6
    audio_out = model(Batch(
        inputs={"audio_cls": audio_changed, "video": video},
        targets=targets,
        masks={"video_mask": video_mask, "observed_mask": observed, "modality_available": available},
        meta={},
    ))
    assert torch.allclose(out.preds[2], audio_out.preds[2], atol=1e-5, rtol=1e-5)

    # Unavailable video invariance for row 1.
    video_changed = video.clone()
    video_changed[1] = 1e6
    video_out = model(Batch(
        inputs={"audio_cls": audio, "video": video_changed},
        targets=targets,
        masks={"video_mask": video_mask, "observed_mask": observed, "modality_available": available},
        meta={},
    ))
    assert torch.allclose(out.preds[1], video_out.preds[1], atol=1e-5, rtol=1e-5)

    bad_available = available.clone()
    bad_available[0] = False
    try:
        model(Batch(
            inputs={"audio_cls": audio, "video": video},
            targets=targets,
            masks={"video_mask": video_mask, "observed_mask": observed, "modality_available": bad_available},
            meta={},
        ))
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
    assert any("gate" in n and float(g.abs().sum()) > 0 for n,g in grads.items())
    assert all(torch.isfinite(g).all() for g in grads.values())

    print("F0 synthetic forward/loss/backward smoke passed", float(loss.detach()))
    PY

Real A+V DataModule smoke:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python - <<'PY'
    import math
    import torch

    from common.loss.wsm_masked_sparse_loss import WSMMaskedSparseLoss
    from fusion.data.wsm_av_fusion_datamodule import WSMAVFusionDataModule
    from fusion.models.av_f0_gated_late import WSMAVGatedLateFusionF0Model

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

    model = WSMAVGatedLateFusionF0Model(
        audio_feature_dim=dm.audio_feature_dim,
        video_feature_dim=512,
        hidden_dim=64,
        gate_hidden_dim=16,
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

    print("F0 real DataModule smoke passed", float(loss.detach()))
    PY

Then:

    git diff --check
    git diff -- src/audio
    git diff -- src/video
    git diff -- src/fusion/data/wsm_av_fusion_datamodule.py
    git status --short

Before commit inspect only:

    git diff -- \
      src/fusion/models/av_f0_gated_late.py \
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
- fixed F0 equation/architecture;
- fixed constructor defaults;
- availability-weight semantics;
- aux output shapes;
- synthetic mask/modality invariance results;
- synthetic masked loss/backward result;
- real DataModule forward/loss/backward result;
- no task_id verification;
- confirmation no config/training/Test metrics;
- confirmation DataModule unchanged;
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

- branch codex/task-004b;
- implementation commit SHA;
- pushed-to-origin status;
- main/master untouched;
- registry key;
- output/aux shapes;
- availability-aware gate checks;
- synthetic and real DataModule loss/backward results;
- no training/Test metrics;
- text/description deferred;
- src/audio unchanged;
- src/video unchanged.

Stop after this task.
