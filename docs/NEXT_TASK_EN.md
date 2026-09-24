# TASK-005A4-C2: Make CLIP Text-Output Unwrapping Tensor-Safe and Reproduce the Accepted Semantic Cache

## Task Identifier and Title

TASK-005A4-C2 — Make CLIP text-output unwrapping tensor-safe and reproduce the already accepted TASK-005A4-C1 semantic audit/cache.

This is one narrow corrective task only.

Required branch:

    codex/task-005a4

The manager explicitly authorizes reuse of this existing branch for exactly one additional corrective implementation commit. Preserve all existing manager and Codex commits. Do not reset, rebase away, overwrite, or force-push branch history.

## Goal

Fix one committed reproducibility bug in `normalize_prompt_bank()`.

The current compatibility path does:

    getattr(x, "text_embeds", None) or getattr(x, "pooler_output", None)

When `text_embeds` is a multi-element `torch.Tensor`, Python evaluates the Tensor's truth value for `or` and raises:

    RuntimeError: Boolean value of Tensor with more than one value is ambiguous

Replace this with explicit, tensor-safe field selection.

Then rerun the exact already-approved semantic audit completely offline and verify that the previously accepted OOF/full-DEV rules and TRAIN cache result reproduce.

Do not redesign any research method and do not start TASK-005B.

The owner explicitly authorized the already completed provisioning/download of the exact `openai/clip-vit-base-patch32`, revision `main`; do not download or update it again in this task.

## Required Reading

Read in this order before editing:

1. AGENTS.md
2. docs/README.md
3. docs/PROJECT_REQUIREMENTS.md
4. docs/PLAN.md, especially Stage 5
5. docs/PROGRESS_EN.md through MANAGER-REVIEW-035
6. docs/NEXT_TASK_EN.md
7. src/fusion/loss/ramps_r2_semantic.py
8. scripts/common/prepare_ramps_r2_semantic_targets.py

Use only the active English documents listed in docs/README.md. Do not load archived Russian variants.

## Allowed Files

Codex may modify only:

- src/fusion/loss/ramps_r2_semantic.py
- docs/PROGRESS_EN.md

Do not modify the production script unless the current committed script itself prevents the unchanged rerun for a reason directly caused by this correction. If that occurs, stop and report instead of broadening scope.

The manager has already updated docs/NEXT_TASK_EN.md and docs/PROGRESS_EN.md on this branch. Preserve those changes.

## Forbidden Actions

- Do not modify src/audio.
- Do not modify src/video.
- Do not modify src/fusion/models.
- Do not modify src/fusion/data.
- Do not modify src/chimera_plugin.py.
- Do not modify scripts/common/prepare_ramps_r2_semantic_targets.py under normal execution of this task.
- Do not modify PROJECT_REQUIREMENTS.md or PLAN.md.
- Do not download, update, or substitute any CLIP/VLM model.
- Do not change CLIP model identity or revision.
- Do not change the fixed prompt bank.
- Do not change semantic margin/calibration.
- Do not change OOF assignment.
- Do not change S1/S2/S3.
- Do not change any selected depression rule or frozen Parkinson rule.
- Do not change precision_target=0.90.
- Do not change min_support=10.
- Do not change folds=5.
- Do not use Test rows or Test metrics.
- Do not re-extract raw frames.
- Do not start student training.
- Do not start TASK-005B.
- Do not start R3/R4, Stage 6, Stage 7, or general Text/Description Stage 3.
- Do not make missing-label correctness or comorbidity claims.

## Implementation Requirements

### 1. Tensor-safe CLIP output unwrapping

Update only `normalize_prompt_bank()` as needed so that it safely supports:

1. a direct `torch.Tensor`;
2. an object whose `text_embeds` attribute is a multi-row `torch.Tensor`;
3. an object whose `text_embeds` is absent/None but whose `pooler_output` is a `torch.Tensor`.

Required behavior:

- never use Python truth-value evaluation on a Tensor;
- prefer `text_embeds` when it is a Tensor;
- otherwise fall back to `pooler_output` only when that is a Tensor;
- raise a clear `TypeError` when neither field provides a Tensor;
- preserve the existing normalization contract exactly:
  - L2-normalize each row;
  - average rows;
  - L2-normalize the mean;
- do not change dtype semantics unnecessarily;
- output must be finite and unit norm within numerical tolerance.

### 2. Regression smoke

Add no new test file. Use an exact inline smoke command that constructs a small fake output object with a multi-element `text_embeds` Tensor and proves:

- no ambiguous-bool Tensor error occurs;
- direct Tensor and `text_embeds` object paths produce identical output;
- `pooler_output` fallback works;
- returned embedding is finite;
- returned embedding has unit norm.

### 3. Offline production reproduction

The exact model is already provisioned. Run the production audit with:

    HF_HUB_OFFLINE=1

The script already uses `local_files_only=True`; both controls must remain in effect.

Use `--overwrite` only to rewrite the already existing TASK-005A4 artifact directory.

The rerun must not access Test rows/metrics or raw video frames.

### 4. Required reproduction target

The rerun must reproduce the accepted method state without reselection or redesign.

Expected deterministic evidence from TASK-005A4-C1:

Depression OOF selected S1:
- positive tau_conf = 0.61
- positive precision/support approximately 0.9132653 / 196
- negative tau_conf = 0.77
- negative precision/support approximately 0.9259259 / 27

Depression full-DEV:
- positive remains enabled
- positive precision/support approximately 0.9020619 / 194
- negative is disabled
- negative measured precision/support approximately 0.8333333 / 30

Frozen Parkinson Family A:
- positive tau_conf = 0.77, precision/support = 1.0 / 21
- negative tau_conf = 0.50, precision/support approximately 0.9585492 / 193

TRAIN cache:
- depression accepted = 376 / 2665
- Parkinson accepted = 1801 / 3660
- total accepted missing = 2177
- depression accepted positive/negative = 376 / 0
- Parkinson accepted positive/negative = 212 / 1589

Small floating-point metric differences are acceptable only if they do not alter any selected/confirmed rule, support count, accepted count, or gate state. Record exact reproduced values.

### 5. Cache invariants

After the rerun, independently inspect the published `train_missing_targets.pt` and assert:

- 6325 rows;
- 6325 unique segment IDs;
- task dimension = 2;
- zero pseudo acceptance on observed entries;
- pseudo targets are NaN on all observed/rejected entries;
- pseudo reliability is zero on all observed/rejected entries;
- pseudo_class is -1 on all observed/rejected entries;
- accepted pseudo targets are finite and within [0,1];
- every accepted pseudo target equals the stored calibrated strong-audio probability exactly;
- accepted reliability is finite and within [0,1];
- pseudo_class values are only -1, 0, 1;
- accepted depression count = 376;
- accepted Parkinson count = 1801.

Do not inspect missing-label correctness because no dual-annotated truth is available.

## Acceptance Criteria

TASK-005A4-C2 passes only if:

- branch remains `codex/task-005a4`;
- only `src/fusion/loss/ramps_r2_semantic.py` and `docs/PROGRESS_EN.md` change after the manager task commit;
- `normalize_prompt_bank()` never truth-tests a Tensor;
- direct Tensor, `text_embeds`, and `pooler_output` smoke paths all pass;
- py_compile passes;
- exact CLIP model/revision load fully offline;
- no new download occurs;
- fixed prompt bank remains unchanged;
- exact policy remains precision_target=0.90, min_support=10, folds=5;
- depression OOF/full-DEV rule state reproduces;
- frozen Parkinson rules reproduce;
- TRAIN accepted counts reproduce exactly;
- all cache invariants pass;
- no Test rows/metrics are used;
- no raw-frame extraction occurs;
- no student training/TASK-005B/R3/R4/Stage 6/Stage 7/general Text/Description work occurs;
- src/audio and src/video remain unchanged;
- git diff --check passes;
- docs/PROGRESS_EN.md records the exact rerun evidence, corrective commit SHA, and push result;
- one corrective commit is pushed to origin;
- main/master remains untouched by Codex.

## Exact Verification Commands

Run from the repository root.

### 1. Compile

    python3 -m py_compile       src/fusion/loss/ramps_r2_semantic.py       scripts/common/prepare_ramps_r2_semantic_targets.py

### 2. Tensor-safe regression smoke

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python - <<'PY'
    import torch
    from fusion.loss.ramps_r2_semantic import normalize_prompt_bank

    torch.manual_seed(42)
    x = torch.randn(3, 512, dtype=torch.float64)

    class TextEmbedsOutput:
        def __init__(self, value):
            self.text_embeds = value
            self.pooler_output = None

    class PoolerOutput:
        def __init__(self, value):
            self.text_embeds = None
            self.pooler_output = value

    direct = normalize_prompt_bank(x)
    via_text = normalize_prompt_bank(TextEmbedsOutput(x))
    via_pooler = normalize_prompt_bank(PoolerOutput(x))

    assert torch.equal(direct, via_text)
    assert torch.equal(direct, via_pooler)
    assert torch.isfinite(direct).all()
    assert abs(float(direct.norm()) - 1.0) < 1e-12
    print("tensor-safe CLIP output regression: ok")
    PY

### 3. Offline CLIP load

    HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python - <<'PY'
    from transformers import CLIPModel, CLIPProcessor

    name = "openai/clip-vit-base-patch32"
    revision = "main"
    processor = CLIPProcessor.from_pretrained(name, revision=revision, local_files_only=True)
    model = CLIPModel.from_pretrained(name, revision=revision, local_files_only=True)
    assert model.config.projection_dim == 512
    print("offline CLIP load: ok")
    PY

### 4. Exact production rerun

    HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python       scripts/common/prepare_ramps_r2_semantic_targets.py       --data-root /media/maxim/Databases/WSM_NEW       --audio-feature-cache-root /media/maxim/Databases/WSM_NEW/features       --video-cache-root /media/maxim/Programs/Features/WSM/video_depart_v1_fullframe_fallback/cache       --audio-checkpoint logs/wsm_audio_segment_wavlm_base_l9_pool4/multitask_audio_mamba_2026-08-19_14-12_audio_mamba_segment_model_wsm_audio_models-e0ce-006_ea8c8d77/checkpoints/epoch=4_dev_mean_score=0.7878.pt       --video-checkpoint logs/wsm_mm_pd_dep_v1/depart_v2_prototype_gated_2026-09-24_14-39_wsm_video_depart_v2_model_0a7294f4/checkpoints/epoch=5_dev_mean_score=0.7066.pt       --clip-model openai/clip-vit-base-patch32       --clip-revision main       --output-root /media/maxim/Programs/Features/WSM/ramps_r2_semantic_v1       --precision-target 0.90       --min-support 10       --folds 5       --batch-size 32       --num-workers 4       --device cuda       --overwrite

### 5. Independent cache invariant check

    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python - <<'PY'
    from pathlib import Path
    import torch

    path = Path("/media/maxim/Programs/Features/WSM/ramps_r2_semantic_v1/train_missing_targets.pt")
    assert path.is_file()
    a = torch.load(path, map_location="cpu", weights_only=False)

    ids = a["segment_ids"]
    observed = a["observed_mask"].bool()
    accept = a["pseudo_accept_mask"].bool()
    targets = a["pseudo_targets"]
    reliability = a["pseudo_reliability"]
    pseudo_class = a["pseudo_class"]
    audio_probs = a["calibrated_audio_probs"]

    assert len(ids) == 6325
    assert len(set(ids)) == 6325
    assert tuple(observed.shape) == (6325, 2)
    assert tuple(accept.shape) == (6325, 2)

    rejected_or_observed = ~accept
    assert not bool((accept & observed).any())
    assert bool(torch.isnan(targets[rejected_or_observed]).all())
    assert bool((reliability[rejected_or_observed] == 0).all())
    assert bool((pseudo_class[rejected_or_observed] == -1).all())

    accepted_targets = targets[accept]
    accepted_reliability = reliability[accept]
    assert bool(torch.isfinite(accepted_targets).all())
    assert bool(((accepted_targets >= 0) & (accepted_targets <= 1)).all())
    assert torch.equal(accepted_targets, audio_probs[accept])
    assert bool(torch.isfinite(accepted_reliability).all())
    assert bool(((accepted_reliability >= 0) & (accepted_reliability <= 1)).all())
    assert set(torch.unique(pseudo_class).tolist()).issubset({-1, 0, 1})

    assert int(accept[:, 0].sum()) == 376
    assert int(accept[:, 1].sum()) == 1801
    assert int((accept[:, 0] & (pseudo_class[:, 0] == 1)).sum()) == 376
    assert int((accept[:, 0] & (pseudo_class[:, 0] == 0)).sum()) == 0
    assert int((accept[:, 1] & (pseudo_class[:, 1] == 1)).sum()) == 212
    assert int((accept[:, 1] & (pseudo_class[:, 1] == 0)).sum()) == 1589

    print("published semantic cache invariants: ok")
    PY

### 6. Scope and repository checks

    git diff --check
    git diff origin/main -- src/audio
    git diff origin/main -- src/video
    git diff origin/main -- src/fusion/models
    git diff origin/main -- src/fusion/data
    git status --short
    git diff --stat origin/main...HEAD
    git log -4 --oneline --decorate

## Required Handoff

Respond in English using exactly:

1. Outcome
2. Changed files
3. Verification
4. Plan status
5. Blockers and risks
6. Next atomic step

Explicitly include:

- branch `codex/task-005a4`;
- corrective commit SHA;
- pushed-to-origin status;
- main/master untouched;
- manager commits preserved;
- exact code change for tensor-safe output unwrapping;
- tensor-object regression smoke result;
- offline CLIP load result;
- confirmation that no download/update occurred in this task;
- audio/video historical DEV reproduction;
- depression OOF selected rules and support/precision;
- depression full-DEV confirmation;
- frozen Parkinson rule reproduction;
- exact TRAIN accepted counts and class counts;
- independent cache invariant check result;
- unchanged precision_target=0.90/min_support=10/folds=5;
- no Test use;
- no raw-frame extraction;
- no Candidate B/fusion teacher;
- no student training;
- no missing-label correctness/comorbidity claim;
- TASK-005B/R3/R4 not started;
- general Text/Description still deferred;
- src/audio unchanged;
- src/video unchanged;
- Stage 5 status.

If all acceptance criteria pass, recommended next atomic step is manager review for TASK-005B.

Stop after TASK-005A4-C2.
