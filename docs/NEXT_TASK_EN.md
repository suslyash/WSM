# TASK-005A3: Recover Two-Head RAMPS Reliability with Independent Video Agreement, Uncertainty, and OOD Filtering

## Role

You are the implementing Codex. Execute only this offline RAMPS-R2 reliability task, update docs/PROGRESS_EN.md, commit on the required task branch, push to origin, then stop. Follow AGENTS.md.

Required branch:

    codex/task-005a3

Do not train a student.
Do not modify src/audio or src/video.
Do not modify any accepted model or DataModule.
Do not use Candidate B or any Stage-4 fusion model as a teacher.
Do not lower the final precision target below 0.90.
Do not lower min_support below 10.
Do not iterate/evaluate any Test protocol.
Do not start R3/R4.
Do not start text/description work.

## Goal

TASK-005A2 established a genuine R1 blocker:

- Parkinson missing-head pseudo-targets can be accepted at the fixed reliability standard.
- Depression cannot accept any pseudo-target under strong-audio confidence alone at:
      precision_target = 0.90
      min_support = 10

Do NOT relax that standard.

Instead, test the next PLAN-authorized RAMPS reliability layer using independent evidence:

1. frozen strong temporal audio = primary teacher and pseudo-target source;
2. independently selected Stage-2 video V2 = agreement evidence only;
3. predictive entropy/disagreement = uncertainty evidence;
4. class-conditional frozen-audio task-feature distance = OOD evidence;
5. deterministic 5-fold out-of-fold DEV reliability selection;
6. publish a TRAIN two-head missing-target cache only if both diseases recover an eligible rule at the unchanged reliability standard.

The video model is NOT the pseudo-target source.

For accepted TRAIN missing entries:

    pseudo_target = calibrated strong-audio probability

Video/uncertainty/OOD signals only decide acceptance/reliability.

## Required Reading

1. AGENTS.md
2. docs/PROJECT_REQUIREMENTS.md Sections 2, 5, 6, 8, 9, 10, 11, 12, 13, 14
3. docs/PLAN.md Stage 5
4. docs/PROGRESS_EN.md through MANAGER-DECISION-032
5. docs/NEXT_TASK_EN.md
6. docs/SOTA_REVIEW_EN.md Sections 5, 7, 8, 9, 10, 11, 13
7. src/fusion/loss/ramps_r1_teacher.py
8. src/fusion/models/frozen_audio_temporal_adapter.py
9. src/video/models/depart_v2.py
10. configs/wsm_mm_pd_dep_v1/video/01_depart_v2_prototype.yaml
11. src/fusion/data/wsm_av_fusion_datamodule.py
12. src/common/callbacks/wsm_segment_callback.py

## Allowed Tracked Files

- src/fusion/loss/ramps_r2_reliability.py
- src/fusion/loss/__init__.py
- scripts/common/prepare_ramps_r2_av_targets.py
- docs/PROGRESS_EN.md

No other tracked file may be modified.

No training config is authorized.

## Fixed Teachers

### Primary strong-audio teacher

Checkpoint:

    logs/wsm_audio_segment_wavlm_base_l9_pool4/multitask_audio_mamba_2026-08-19_14-12_audio_mamba_segment_model_wsm_audio_models-e0ce-006_ea8c8d77/checkpoints/epoch=4_dev_mean_score=0.7878.pt

Required SHA256:

    0873c7cb5e32d415cdd301058949f5dd140c16b874cc5230d027a4ee33e3daf2

Adapter:

    FrozenAudioTemporalAdapter

Historical DEV reference:

    depression Score = 0.7479183895
    Parkinson Score = 0.8277353635
    Mean_Score = 0.7878268765

The strong-audio calibrated probability remains the ONLY pseudo-target value.

### Independent video agreement teacher

Use the already Stage-2-selected V2 checkpoint:

    logs/wsm_mm_pd_dep_v1/depart_v2_prototype_gated_2026-09-24_14-39_wsm_video_depart_v2_model_0a7294f4/checkpoints/epoch=5_dev_mean_score=0.7066.pt

Model contract:

    WSMVideoDepartV2Model
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

Strict-load the checkpoint model_state_dict.

Require the checkpoint payload to identify epoch 5.

Compute and record its SHA256.

Before using it in reliability selection, reproduce canonical DEV within absolute tolerance 0.0005 for:

    depression Score = 0.620101
    Parkinson Score = 0.793043
    Mean_Score = 0.706572

If strict load or DEV reproduction fails:

    STOP BLOCKED.

Do not substitute V1, Candidate B, or another checkpoint.

Both teachers must be:

- eval-only;
- requires_grad=false;
- run under inference_mode/no_grad;
- absent from any optimizer.

No optimizer is needed anywhere in TASK-005A3.

## Fixed Data

Use:

    WSMAVFusionDataModule

Roots:

    data_root = /media/maxim/Databases/WSM_NEW
    audio_feature_cache_root = /media/maxim/Databases/WSM_NEW/features
    video_cache_root = /media/maxim/Programs/Features/WSM/video_depart_v1_fullframe_fallback/cache

Expected counts:

    train = 6325
    dev = 933
    test_none = 1364
    test_soft = 1208
    test_hard = 1014
    joined_total = 8622
    missing_audio = 0
    missing_video = 0

Only:

    train_dataset
    val_dataset

may be iterated.

MUST NOT call:

    val_dataloader()

MUST NOT iterate or infer:

    test_none
    test_soft
    test_hard

## Fixed Output Root

Write runtime artifacts outside Git to:

    /media/maxim/Programs/Features/WSM/ramps_r2_av_reliability_v1

Required always:

    reliability_search.json
    audit.json

Write only if the two-head gate passes:

    train_missing_targets.pt

Do not add runtime artifacts to git.

The production script must refuse overwrite of a non-empty output root unless explicit --overwrite is supplied.

## 1. Reusable R2 Reliability Utilities

Implement:

    src/fusion/loss/ramps_r2_reliability.py

Reuse R1 temperature/calibration utilities rather than duplicating them where appropriate.

Implement deterministic utilities for:

- normalized binary entropy;
- deterministic stratified 5-fold assignment;
- class confidence;
- audio/video same-class agreement;
- normalized audio task-feature class-centroid cosine distance;
- empirical OOD percentile against a reference distance distribution;
- bounded reliability-rule search;
- rule application;
- detached reliability weights.

No sklearn/scipy/new dependency.

## 2. Deterministic Stratified Five-Fold DEV Contract

Perform reliability-family selection separately for each disease.

Use ONLY rows where that disease is observed.

Create exactly 5 folds, stratified by binary label.

Deterministic assignment:

1. within each class, sort rows by:
       sha256(f"{task_name}:{segment_id}")
2. assign round-robin fold indices 0..4.

Require:

- each observed row appears in exactly one held-out fold;
- no held-out row is used to fit its own audio temperature;
- no held-out row is used to fit its own video temperature;
- no held-out row is used to fit its own class centroid/OOD reference distribution;
- both labels are present in every fitting partition.

If a fitting partition lacks either class:

    STOP BLOCKED.

## 3. Out-of-Fold Teacher Calibration

For each task and each fold:

Using the other four folds only:

- fit audio temperature with fit_binary_temperature;
- fit video temperature with fit_binary_temperature;
- compute normalized frozen-audio task-feature centroid for class 0;
- compute normalized frozen-audio task-feature centroid for class 1;
- compute fit-partition cosine-distance reference distribution to each row's true-class centroid.

On the held-out fold compute:

    p_audio
    p_video
    audio class confidence
    video class confidence
    joint_confidence
    normalized audio entropy
    normalized video entropy
    uncertainty = max(audio_entropy, video_entropy)
    class-conditional audio OOD percentile

Binary entropy:

    H(p) = -(p*ln(p) + (1-p)*ln(1-p)) / ln(2)

Use numerical clamping only to avoid log(0).

Class side:

    positive if p >= 0.5
    negative if p < 0.5

A row can enter a side-specific acceptance rule only when BOTH teachers predict the same side.

For side c:

    audio_conf = p_audio      if c=1 else 1-p_audio
    video_conf = p_video      if c=1 else 1-p_video
    joint_confidence = min(audio_conf, video_conf)

Audio OOD distance:

- L2-normalize frozen audio task feature;
- L2-normalize class centroid;
- distance = 1 - cosine_similarity(feature, centroid).

OOD percentile:

- compare held-out distance to the fitting partition's distance distribution for the candidate class side;
- percentile = fraction of reference distances <= held-out distance;
- lower percentile = more in-distribution.

Store one pooled out-of-fold record per observed DEV row.

## 4. Exactly Three Predeclared Reliability Families

The search is limited to exactly these families.

The FINAL empirical precision target remains:

    0.90

Minimum support remains:

    10

### Family A — Agreement + Joint Confidence

Require same-class audio/video agreement.

Accept side c when:

    joint_confidence >= tau_conf

Grid:

    tau_conf = 0.50, 0.51, ..., 0.99

### Family B — Agreement + Joint Confidence + Low Entropy

Require Family A plus:

    uncertainty <= tau_entropy

Fixed entropy cutoffs:

    0.25
    0.50
    0.75

Search the cross-product of:

    tau_conf grid
    tau_entropy set

### Family C — Agreement + Joint Confidence + Low Entropy + In-Distribution Audio Feature

Require Family B plus:

    ood_percentile <= tau_ood

Fixed OOD percentile cutoffs:

    0.80
    0.90
    0.95

Search:

    tau_conf grid
    tau_entropy set
    tau_ood set

Do NOT add a fourth family.

Do NOT use a semantic/VLM feature in this task.

## 5. OOF Family Selection Policy

For each:

    task in {depression, parkinson}
    side in {positive, negative}

evaluate all allowed rule configurations on the pooled OOF DEV records.

For each configuration record:

- family;
- parameters;
- support;
- correct;
- empirical precision;
- class coverage = correct / number of true DEV examples of that class.

Eligible iff:

    support >= 10
    precision >= 0.90

Select the eligible configuration using deterministic ordering:

1. maximum support;
2. then maximum precision;
3. then lower complexity:
       Family A before B before C;
4. then lower tau_conf;
5. for B/C, higher tau_entropy;
6. for C, higher tau_ood.

If no rule is eligible for a side:

    side enabled = false

A disease is OOF-recoverable if at least ONE side is enabled.

Do not require both positive and negative sides to be enabled.

Record full coverage curves/search tables in reliability_search.json.

## 6. Full-DEV Confirmation of the OOF-Selected Rule

After OOF selection, fit deployment statistics on ALL observed DEV rows for each disease:

- one full-DEV audio temperature;
- one full-DEV video temperature;
- class 0/1 audio task-feature centroids;
- full-DEV class-specific distance reference distributions.

Apply the EXACT OOF-selected family/hyperparameters without redesigning or changing thresholds.

For every enabled side require again on full observed DEV:

    support >= 10
    precision >= 0.90

If an OOF-selected side fails full-DEV confirmation:

    disable that side

Do NOT refit another rule after seeing the full-DEV failure.

A disease is deployable only if at least one side remains enabled after full-DEV confirmation.

The overall R2 two-head gate passes only if BOTH diseases are deployable.

## 7. TRAIN Missing-Head Target Construction

Only if the overall two-head gate passes:

Infer all 6325 canonical TRAIN rows exactly once with both frozen teachers.

Use the full-DEV deployment temperatures/centroids/reference distributions.

For each missing task entry:

1. determine audio/video probabilities;
2. determine agreed side;
3. compute joint confidence, uncertainty, and audio OOD percentile;
4. apply that task/side's confirmed selected rule.

If accepted:

    pseudo_target = calibrated_audio_probability

NOT video probability.
NOT an average.
NOT a hard 0/1 label.

Pseudo class:

    1 for accepted positive
    0 for accepted negative

Reliability weight:

Family A:

    reliability = joint_confidence

Family B:

    reliability = joint_confidence * (1 - uncertainty)

Family C:

    reliability = joint_confidence * (1 - uncertainty) * (1 - ood_percentile)

Clamp only numerically to [0,1].

Reliability is detached/offline.

For observed entries:

    pseudo_accept_mask = false
    pseudo_target = NaN
    pseudo_reliability = 0
    pseudo_class = -1

Observed truth always wins.

For rejected missing entries use the same null contract.

## 8. train_missing_targets.pt Contract

If the two-head gate passes, save at least:

    version = "ramps-r2-av-reliability-v1"
    task_names = ["depression", "parkinson"]
    segment_ids
    observed_mask
    observed_targets

    raw_audio_logits
    calibrated_audio_probs
    raw_video_logits
    calibrated_video_probs

    audio_task_features

    pseudo_accept_mask
    pseudo_targets
    pseudo_reliability
    pseudo_class

    selected_rules
    audio_temperatures
    video_temperatures

    audio_checkpoint_path
    audio_checkpoint_sha256
    video_checkpoint_path
    video_checkpoint_sha256

Requirements:

- rows=6325, tasks=2;
- unique canonical segment IDs;
- observed mask/targets unchanged;
- no pseudo values on observed entries;
- accepted target finite [0,1];
- accepted target equals calibrated AUDIO probability for that entry;
- reliability finite [0,1];
- reliability zero on observed/rejected entries;
- pseudo_class in {-1,0,1};
- at least one accepted missing entry for depression;
- at least one accepted missing entry for Parkinson.

If the overall two-head gate fails:

    DO NOT write train_missing_targets.pt.

## 9. reliability_search.json Contract

Always write, even if R2 remains blocked.

Include:

- artifact version;
- generation UTC;
- exact audio/video checkpoint paths and SHA256;
- teacher/model class names;
- data/cache roots;
- canonical counts;
- no-Test statement;
- 5-fold construction rule;
- per-fold class counts;
- per-fold fitted audio/video temperatures;
- full-DEV fitted audio/video temperatures;
- historical teacher DEV reproduction metrics;
- all three family definitions;
- full OOF candidate audit table per task/side;
- selected OOF rule per task/side or disabled;
- full-DEV confirmation per selected side;
- whether each disease is deployable;
- whether overall two-head gate passes.

Do not include Test values.

## 10. audit.json Contract

Always write.

If blocked, audit the failed reliability selection.

If passed, additionally audit TRAIN target construction.

Include per task:

- missing TRAIN count;
- accepted total;
- rejected total;
- coverage;
- accepted positive/negative counts;
- selected family/rule per side;
- OOF precision/support per enabled side;
- full-DEV precision/support per enabled side;
- accepted calibrated-audio probability stats;
- reliability stats;
- audio/video agreement rate among missing rows.

Overall:

    train_rows = 6325
    observed_overwrite_violations = 0
    duplicate_segment_ids = 0
    nonfinite_accepted_targets = 0
    pseudo_values_on_observed_entries = 0
    pseudo_acceptance_on_observed_entries = 0

Do NOT claim missing-label accuracy/correctness/comorbidity recovery.

## 11. Production Script

Implement:

    scripts/common/prepare_ramps_r2_av_targets.py

Required sequence:

1. deterministic seeds;
2. verify audio checkpoint SHA;
3. compute video checkpoint SHA;
4. strict-load exact video V2 checkpoint and verify epoch=5;
5. build accepted A+V DataModule;
6. verify canonical counts;
7. build/freeze/eval both teachers;
8. infer canonical DEV only, collecting:
   - segment IDs;
   - targets/observed masks;
   - audio logits/task features;
   - video logits;
9. reproduce historical DEV metrics for BOTH teachers;
10. construct deterministic stratified 5-fold OOF records;
11. evaluate exactly Families A/B/C;
12. freeze selected OOF rules;
13. run full-DEV confirmation with no fallback redesign;
14. write reliability_search.json;
15. if two-head gate fails:
       write audit.json;
       exit with RuntimeError explaining which task/side failed;
       do not infer TRAIN;
       do not write target cache;
16. if two-head gate passes:
       infer canonical TRAIN only;
       construct missing-only AUDIO soft targets using R2 acceptance;
       validate invariants;
       write train_missing_targets.pt and audit.json.

Do not call dm.val_dataloader().

Do not infer Test.

## Exact Production Command

Run:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python \
      scripts/common/prepare_ramps_r2_av_targets.py \
      --data-root /media/maxim/Databases/WSM_NEW \
      --audio-feature-cache-root /media/maxim/Databases/WSM_NEW/features \
      --video-cache-root /media/maxim/Programs/Features/WSM/video_depart_v1_fullframe_fallback/cache \
      --audio-checkpoint logs/wsm_audio_segment_wavlm_base_l9_pool4/multitask_audio_mamba_2026-08-19_14-12_audio_mamba_segment_model_wsm_audio_models-e0ce-006_ea8c8d77/checkpoints/epoch=4_dev_mean_score=0.7878.pt \
      --video-checkpoint logs/wsm_mm_pd_dep_v1/depart_v2_prototype_gated_2026-09-24_14-39_wsm_video_depart_v2_model_0a7294f4/checkpoints/epoch=5_dev_mean_score=0.7066.pt \
      --output-root /media/maxim/Programs/Features/WSM/ramps_r2_av_reliability_v1 \
      --precision-target 0.90 \
      --min-support 10 \
      --folds 5 \
      --batch-size 32 \
      --num-workers 4 \
      --device cuda

If CUDA unavailable:

    STOP BLOCKED.

No CPU fallback.

## Required Verification

Compile:

    python3 -m py_compile \
      src/fusion/loss/ramps_r2_reliability.py \
      src/fusion/loss/__init__.py \
      scripts/common/prepare_ramps_r2_av_targets.py

Synthetic/unit checks must cover:

- entropy finite and [0,1];
- deterministic stratified folds;
- no held-out leakage into temperature/centroid fitting;
- OOD percentile in [0,1];
- Family A selection;
- Family B selection;
- Family C selection;
- disabled side behavior;
- deterministic tie-breaking;
- observed truth cannot become pseudo;
- pseudo target uses audio probability exactly;
- reliability in [0,1].

Then run the exact production command.

Afterward verify:

- checkpoint/model DEV reproductions;
- no teacher gradients;
- no optimizer instantiated;
- no Test iteration or metric;
- artifact contracts;
- if blocked, no train_missing_targets.pt exists;
- if passed, all TRAIN cache invariants pass.

Finally:

    git diff --check
    git diff -- src/audio
    git diff -- src/video
    git diff -- src/fusion/models
    git diff -- src/fusion/data
    git status --short

Before commit inspect only:

    git diff -- \
      src/fusion/loss/ramps_r2_reliability.py \
      src/fusion/loss/__init__.py \
      scripts/common/prepare_ramps_r2_av_targets.py \
      docs/PROGRESS_EN.md

## Acceptance Criteria

TASK-005A3 passes as an R2 target-preparation step only if:

- exact audio checkpoint SHA matches;
- exact video V2 checkpoint strict-loads and epoch=5;
- video checkpoint SHA is recorded;
- audio DEV reproduction passes;
- video V2 DEV reproduction passes;
- both teachers frozen/eval/no-grad;
- no optimizer exists;
- no Candidate B/fusion teacher;
- deterministic 5-fold OOF protocol passes;
- exactly three reliability families are searched;
- final precision target remains 0.90;
- min_support remains 10;
- no Test row/metric is used;
- selected rules are frozen from OOF evidence before full-DEV confirmation;
- no fallback redesign after full-DEV failure;
- both diseases retain at least one enabled side after OOF + full-DEV reliability gates;
- TRAIN inference occurs only after that two-head DEV gate passes;
- accepted TRAIN targets come only from calibrated audio probabilities;
- observed truth is never overwritten;
- at least one accepted missing target exists for each disease;
- no missing-label correctness claim;
- no student training;
- no R3/R4;
- text/description deferred;
- src/audio unchanged;
- src/video unchanged;
- accepted models/DataModule unchanged;
- git diff --check passes;
- tracked diff contains only the four allowed paths;
- branch codex/task-005a3 committed and pushed;
- main/master untouched.

If the reliability search is blocked because one disease remains undeployable, the task is a valid BLOCKED evidence task if:

- reliability_search.json and audit.json are complete;
- no target cache is published;
- no thresholds are relaxed;
- no Test is used;
- no student training occurs.

## Required PROGRESS_EN Update

Append TASK-005A3 evidence.

Record:

- branch;
- implementation/evidence commit SHA;
- push result;
- audio checkpoint path/SHA;
- video checkpoint path/SHA;
- strict video load/epoch proof;
- CUDA/GPU;
- canonical counts;
- audio DEV reproduction;
- video V2 DEV reproduction;
- 5-fold construction/audit;
- audio/video full-DEV temperatures;
- OOF selected rule per task/side;
- OOF support/precision/coverage;
- full-DEV confirmation support/precision;
- disease deployable/blocked states;
- output artifact paths;
- if passed:
  - per-task TRAIN accepted/rejected/coverage;
  - accepted pos/neg counts;
  - target/reliability stats;
  - observed-overwrite audit;
- if blocked:
  - explicit no target-cache publication;
- explicit unchanged 0.90/min_support=10 statement;
- explicit no-Test statement;
- explicit no Candidate B/fusion teacher;
- explicit no missing-label correctness claim;
- explicit no student training;
- R3/R4 not started;
- text/description deferred;
- src/audio unchanged;
- src/video unchanged;
- Stage 5 status;
- recommended next atomic task only.

If TASK-005A3 passes the two-head gate:

    recommended next = TASK-005B wire the accepted R2 offline cache into an observed+pseudo sparse loss/data contract and prove direct missing-head gradients without full training.

If TASK-005A3 remains blocked:

    recommended next = manager review for a semantic/VLM evidence step; do not lower the reliability target automatically.

## Required Handoff

Respond in English using exactly:

1. Outcome
2. Changed files
3. Verification
4. Plan status
5. Blockers and risks
6. Next atomic step

Explicitly include:

- branch codex/task-005a3;
- implementation/evidence commit SHA;
- pushed-to-origin status;
- main/master untouched;
- audio checkpoint SHA verification;
- video checkpoint SHA and DEV reproduction;
- OOF selected rules for depression/Parkinson sides;
- OOF and full-DEV precision/support;
- whether each disease is deployable;
- whether a valid two-head TRAIN cache was published;
- if published: accepted/rejected/coverage and pos/neg counts per disease;
- artifact paths;
- unchanged precision_target=0.90/min_support=10;
- no Test use;
- no Candidate B/fusion teacher;
- no student training;
- no missing-label correctness claim;
- R3/R4 not started;
- text/description deferred;
- src/audio unchanged;
- src/video unchanged.

Stop after this task.
