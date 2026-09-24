# TASK-005A4-C1: Complete the Fixed CLIP Semantic Audit Success Path Without Changing the Method

## Task Identifier and Title

TASK-005A4-C1 — Complete the already-authorized TASK-005A4 success path and fixed-contract enforcement.

This is a corrective task only. The manager explicitly authorizes reuse of the existing branch:

    codex/task-005a4

Do not create another branch. Do not reset, overwrite, rebase away, or force-push existing TASK-005A4 work. Add one corrective implementation commit after the manager commits now present on this branch.

## Goal

Make the existing TASK-005A4 implementation faithful to its original contract without changing the research method.

The local exact CLIP dependency may still be unavailable. This corrective task does not provision, download, or substitute it. Instead, make the code structurally complete so that:

1. the current local-CLIP-unavailable path remains a valid blocked outcome with the required audit artifacts and no TRAIN cache; and
2. if the exact local CLIP processor/model becomes available, the unchanged semantic audit can continue through OOF/full-DEV confirmation and, only if the two-head gate passes, build and validate the required two-head TRAIN cache.

Do not start TASK-005B in this task.

## Required Reading

Read in this order before editing:

1. AGENTS.md
2. docs/README.md
3. docs/PROJECT_REQUIREMENTS.md
4. docs/PLAN.md, especially Stage 5
5. docs/PROGRESS_EN.md through MANAGER-REVIEW-034
6. docs/NEXT_TASK_EN.md
7. src/fusion/loss/ramps_r1_teacher.py
8. src/fusion/loss/ramps_r2_reliability.py
9. src/fusion/loss/ramps_r2_semantic.py
10. scripts/common/prepare_ramps_r2_semantic_targets.py
11. src/fusion/models/frozen_audio_temporal_adapter.py
12. src/fusion/data/wsm_av_fusion_datamodule.py
13. src/video/models/depart_v2.py

Use only the active English documentation listed in docs/README.md. Do not load archived Russian variants.

## Allowed Files

Codex may modify only:

- src/fusion/loss/ramps_r2_semantic.py
- src/fusion/loss/__init__.py
- scripts/common/prepare_ramps_r2_semantic_targets.py
- docs/PROGRESS_EN.md

The manager has already modified docs/NEXT_TASK_EN.md and docs/PROGRESS_EN.md on this branch. Preserve those manager changes.

## Forbidden Actions

- Do not modify src/audio.
- Do not modify src/video.
- Do not modify src/fusion/models or src/fusion/data.
- Do not modify checkpoints, accepted caches, manifests, configs, PROJECT_REQUIREMENTS.md, or PLAN.md.
- Do not download any CLIP/VLM model or processor.
- Do not substitute another model, revision, tokenizer, prompt bank, or image representation.
- Do not add a fourth semantic reliability family.
- Do not change any authorized prompt.
- Do not lower or raise the fixed final reliability policy: precision_target must be exactly 0.90 and min_support exactly 10.
- Do not change folds from exactly 5.
- Do not reselect Parkinson rules.
- Do not use Candidate B or any Stage-4 fusion model as a teacher.
- Do not iterate Test rows or inspect Test metrics.
- Do not call val_dataloader().
- Do not read or re-extract raw video frames.
- Do not train a student.
- Do not start TASK-005B, R3, R4, Stage 6, Stage 7, or deferred general Text/Description work.
- Do not claim missing-label correctness or comorbidity recovery.

## Implementation Requirements

### 1. Complete the success path

Remove the sentinel/unconditional failure that currently prevents a successful deployment path.

If and only if the frozen depression semantic OOF rule survives frozen full-DEV confirmation, then:

- infer all 6325 canonical TRAIN rows exactly once;
- use the full-DEV fitted audio/video temperatures and semantic calibrator;
- apply only the confirmed depression rule(s);
- for depression Family S2, use candidate-side OOD percentile/reference with no target leakage;
- apply only the frozen Parkinson TASK-005A3 Family-A rules:
  - positive tau_conf = 0.77;
  - negative tau_conf = 0.50;
- never semantically reselect Parkinson.

Build and atomically publish:

    /media/maxim/Programs/Features/WSM/ramps_r2_semantic_v1/train_missing_targets.pt

only after every required cache invariant passes.

The cache must contain every field required by the original TASK-005A4 contract:

- task_names
- segment_ids
- observed_mask
- observed_targets
- raw_audio_logits
- calibrated_audio_probs
- raw_video_logits
- calibrated_video_probs
- semantic_margin_depression
- calibrated_semantic_prob_depression
- audio_task_features
- pseudo_accept_mask
- pseudo_targets
- pseudo_reliability
- pseudo_class
- depression_selected_rules
- parkinson_frozen_rules
- audio_temperatures
- video_temperatures
- semantic_calibrator
- semantic_prompt_bank
- semantic_prompt_bank_sha256
- clip_model_name
- clip_model_revision
- audio_checkpoint_path
- audio_checkpoint_sha256
- video_checkpoint_path
- video_checkpoint_sha256

Required cache invariants:

- rows = 6325 and tasks = 2;
- unique canonical segment IDs;
- observed targets/masks preserved exactly;
- no pseudo acceptance on observed entries;
- observed/rejected pseudo_target is NaN;
- observed/rejected pseudo_reliability is 0;
- observed/rejected pseudo_class is -1;
- accepted pseudo_target is finite in [0,1];
- every accepted pseudo_target equals the calibrated strong-audio probability for that task exactly;
- accepted reliability is finite in [0,1];
- pseudo_class is only -1, 0, or 1;
- accepted depression count > 0;
- accepted Parkinson count > 0.

Observed truth always wins.

### 2. Correct detached semantic reliability

Fix the detached semantic reliability utility so that it is explicitly candidate-side aware.

For depression:

- S1 reliability:
  min(audio_conf_c, semantic_conf_c)
- S2 reliability:
  min(audio_conf_c, semantic_conf_c) * (1 - uncertainty) * (1 - ood_percentile_c)
- S3 reliability:
  min(audio_conf_c, video_conf_c, semantic_conf_c) * (1 - uncertainty)

The utility must use:

- ood_percentile_positive when candidate side c=1;
- ood_percentile_negative when candidate side c=0.

It must not read a nonexistent generic `ood_percentile` field and must not use a held-out or missing target to choose a side.

Parkinson frozen Family-A reliability remains:

    min(audio_conf_c, video_conf_c)

All reliability tensors must be detached and clamped to [0,1].

### 3. Enforce the fixed contract exactly

The production script must reject any deviation from:

    clip_model = openai/clip-vit-base-patch32
    clip_revision = main
    precision_target = 0.90
    min_support = 10
    folds = 5

Do not merely reject values below the threshold; require exact equality for the fixed task policy.

The audio/video checkpoint SHA and video epoch-5 strict-load requirements remain unchanged.

### 4. Add deterministic initialization

Add the required deterministic seed initialization before model/data inference.

Use a single fixed seed of 42 for Python and Torch, including CUDA when available. Do not add a new dependency solely for seeding.

Do not change the deterministic hash-based fold assignment.

### 5. Preserve the blocked local-CLIP path

If the exact local processor/model still cannot load with local_files_only=True:

- do not access the network;
- do not substitute another model;
- write semantic_prompt_bank.json;
- write semantic_search.json;
- write audit.json;
- set blocked=true;
- set train_inference_ran=false;
- set train_missing_targets_published=false;
- keep depression/neutral embedding hashes null;
- do not create train_missing_targets.pt;
- exit with a clear RuntimeError.

### 6. Preserve all semantic and evaluation invariants

Keep exactly the original fixed prompts, semantic margin, positive-monotonic calibration, five-fold OOF contract, side-conditional OOD logic, S1/S2/S3 definitions, frozen full-DEV confirmation, and frozen Parkinson rule reproduction.

Do not use Test for model, threshold, epoch, modality, reliability, or ablation selection.

## Acceptance Criteria

TASK-005A4-C1 passes only if all of the following are true:

- tracked diff remains inside the four allowed implementation paths plus the pre-existing manager edits;
- src/audio and src/video remain unchanged;
- the sentinel success-path RuntimeError is gone;
- the code contains a complete TRAIN cache construction/validation/publication path reachable only after the two-head DEV gate;
- S2 reliability uses the correct candidate-side OOD percentile;
- reliability is detached, finite, bounded, and target-leakage-free;
- fixed CLIP identity/revision, precision_target=0.90, min_support=10, and folds=5 are enforced exactly;
- deterministic seed 42 initialization is present;
- fixed prompts and exactly S1/S2/S3 remain unchanged;
- Parkinson frozen R2 rules remain unchanged and are not reselected;
- no Test rows or metrics are used;
- no raw-frame extraction is introduced;
- no student training, TASK-005B, R3/R4, or general Stage-3 Text/Description work starts;
- py_compile passes;
- deterministic synthetic checks pass;
- git diff --check passes;
- the exact production rerun below is attempted.

Production outcome handling:

- If the exact local CLIP dependency is still unavailable, the corrective task may finish as environment-blocked only if the three blocked artifacts are rewritten correctly, no TRAIN cache exists, and the code review/synthetic checks prove the complete pass path is implemented.
- If the exact local CLIP dependency is available, run the unchanged semantic audit through DEV. If depression remains undeployable, publish only the audit artifacts and no TRAIN cache. If depression becomes deployable, build and validate the full two-head TRAIN cache exactly as specified.

Update docs/PROGRESS_EN.md with exact commands, results, branch, corrective commit SHA, push result, all deviations, and the remaining blocker/status.

## Exact Verification Commands

Run from the repository root on branch `codex/task-005a4`.

1. Compile:

    python3 -m py_compile \
      src/fusion/loss/ramps_r2_semantic.py \
      src/fusion/loss/__init__.py \
      scripts/common/prepare_ramps_r2_semantic_targets.py

2. Candidate-side reliability smoke:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python - <<'PY'
    import torch
    from fusion.loss.ramps_r2_semantic import semantic_reliability

    records = {
        "audio_prob": torch.tensor([0.90, 0.10], dtype=torch.float64),
        "video_prob": torch.tensor([0.80, 0.20], dtype=torch.float64),
        "semantic_prob": torch.tensor([0.85, 0.15], dtype=torch.float64),
        "audio_entropy": torch.tensor([0.20, 0.20], dtype=torch.float64),
        "video_entropy": torch.tensor([0.20, 0.20], dtype=torch.float64),
        "semantic_entropy": torch.tensor([0.20, 0.20], dtype=torch.float64),
        "ood_percentile_positive": torch.tensor([0.10, 0.90], dtype=torch.float64),
        "ood_percentile_negative": torch.tensor([0.90, 0.10], dtype=torch.float64),
    }
    rule = {"enabled": True, "family": "S2", "tau_conf": 0.50, "tau_entropy": 0.75, "tau_ood": 0.95}
    mask = torch.tensor([True, True])

    rp = semantic_reliability(records, rule, True, mask)
    rn = semantic_reliability(records, rule, False, mask)

    assert rp.shape == rn.shape == torch.Size([2])
    assert torch.isfinite(rp).all() and torch.isfinite(rn).all()
    assert ((0 <= rp) & (rp <= 1)).all()
    assert ((0 <= rn) & (rn <= 1)).all()
    assert not rp.requires_grad and not rn.requires_grad
    assert rp[0] > rp[1], (rp, rn)
    assert rn[1] > rn[0], (rp, rn)
    print("candidate-side semantic reliability smoke: ok")
    PY

3. Verify the old sentinel is absent:

    ! grep -n "semantic TRAIN writer not reached" scripts/common/prepare_ramps_r2_semantic_targets.py

4. Verify fixed prompt/model strings remain present:

    grep -n "openai/clip-vit-base-patch32" scripts/common/prepare_ramps_r2_semantic_targets.py
    grep -n "a video of a person with depression" scripts/common/prepare_ramps_r2_semantic_targets.py
    grep -n "a person with depressive symptoms" scripts/common/prepare_ramps_r2_semantic_targets.py
    grep -n "a person showing signs of depression" scripts/common/prepare_ramps_r2_semantic_targets.py

5. Production rerun.

The manager authorizes `--overwrite` for this corrective rerun only because TASK-005A4 already populated the fixed output directory:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python \
      scripts/common/prepare_ramps_r2_semantic_targets.py \
      --data-root /media/maxim/Databases/WSM_NEW \
      --audio-feature-cache-root /media/maxim/Databases/WSM_NEW/features \
      --video-cache-root /media/maxim/Programs/Features/WSM/video_depart_v1_fullframe_fallback/cache \
      --audio-checkpoint logs/wsm_audio_segment_wavlm_base_l9_pool4/multitask_audio_mamba_2026-08-19_14-12_audio_mamba_segment_model_wsm_audio_models-e0ce-006_ea8c8d77/checkpoints/epoch=4_dev_mean_score=0.7878.pt \
      --video-checkpoint logs/wsm_mm_pd_dep_v1/depart_v2_prototype_gated_2026-09-24_14-39_wsm_video_depart_v2_model_0a7294f4/checkpoints/epoch=5_dev_mean_score=0.7066.pt \
      --clip-model openai/clip-vit-base-patch32 \
      --clip-revision main \
      --output-root /media/maxim/Programs/Features/WSM/ramps_r2_semantic_v1 \
      --precision-target 0.90 \
      --min-support 10 \
      --folds 5 \
      --batch-size 32 \
      --num-workers 4 \
      --device cuda \
      --overwrite

6. Final repository checks:

    git diff --check
    git diff origin/main -- src/audio
    git diff origin/main -- src/video
    git diff origin/main -- src/fusion/models
    git diff origin/main -- src/fusion/data
    git status --short
    git diff --stat origin/main...HEAD
    git log -3 --oneline --decorate

## Required Handoff

Respond in English with exactly:

1. Outcome
2. Changed files
3. Verification
4. Plan status
5. Blockers and risks
6. Next atomic step

Explicitly report:

- branch `codex/task-005a4`;
- corrective commit SHA;
- pushed-to-origin status;
- main/master untouched;
- whether manager commits were preserved;
- whether the success-path sentinel was removed;
- whether full TRAIN cache construction/validation is now implemented;
- exact fixed-contract enforcement;
- deterministic seed evidence;
- candidate-side S2 reliability verification;
- audio/video checkpoint verification reached by the production rerun;
- exact CLIP model/revision and whether local-only load succeeded;
- if CLIP loaded: historical audio/video DEV reproduction, frozen Parkinson rule reproduction, depression OOF/full-DEV results, deployable state, and cache publication status;
- if CLIP did not load: exact exception, required blocked artifacts, and confirmation that train_missing_targets.pt is absent;
- unchanged precision_target=0.90 and min_support=10;
- no Test use;
- no raw-frame extraction;
- no Candidate B/fusion teacher;
- no student training;
- no missing-label correctness/comorbidity claim;
- R3/R4 not started;
- general Text/Description still deferred;
- src/audio unchanged;
- src/video unchanged;
- Stage 5 status.

Stop after TASK-005A4-C1. Do not start TASK-005B.
