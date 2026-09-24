# TASK-004K: Run a Bounded Three-Model Audio-First Temporal A+V Search

## Role

You are the implementing Codex. Execute only this bounded research sprint, update docs/PROGRESS_EN.md, commit on the required task branch, push to origin, then stop. Follow AGENTS.md.

Required branch:

    codex/task-004k

This task explicitly supersedes the unexecuted TASK-004I4.

This is a manager-authorized exception to the previous one-model-at-a-time Stage 4 workflow: you may design, implement, and train exactly THREE bounded candidate fusion models in this one task.

Do not add a fourth candidate.
Do not unfreeze or modify src/audio.
Do not modify src/video.
Do not modify the accepted A+V DataModule.
Do not modify the historical audio checkpoint.
Do not start RAMPS, text, or description work.
Do not use Test metrics to design, rank, revise, or select candidates.

## Goal

Find an audio-first temporal A+V fusion candidate that can beat the exact frozen temporal-audio DEV reference while preserving task balance.

Frozen strong-audio reference:

    depression Score = 0.7479183895
    Parkinson Score = 0.8277353635
    Mean_Score = 0.7878268765

The audio system is the dominant anchor.

Every candidate MUST:

    final_logits = frozen_audio_base_logits + learned_video_correction

and MUST initialize with:

    learned_video_correction == 0 exactly

so that before training:

    final_logits == frozen_audio_base_logits exactly

for every valid row.

The video path is allowed to correct the strong audio model, but it is not allowed to replace the frozen audio base.

## Research Freedom Granted

You have freedom to choose exact internal details within the three required candidate families below.

You may choose:

- exact hidden layout inside the stated parameter cap;
- attention head count when valid;
- whether candidate C uses temporal-video attention, directed relations, or both;
- exact task-specific gate MLP structure;
- exact residual hidden structure;
- whether candidate B/C reuse small helper modules across tasks;
- implementation organization across the three allowed model files.

However:

- the three candidate architectures MUST be fully specified in a fixed SEARCH MANIFEST before the first training run;
- once the first metric-bearing training run starts, you MUST NOT redesign any candidate architecture based on DEV or Test results;
- all three candidates use the same fixed training protocol except one documented batch-size OOM fallback described below.

## Required Reading

1. AGENTS.md
2. docs/PROJECT_REQUIREMENTS.md Sections 2-10, 13-14
3. docs/PLAN.md Stage 4 and Promotion Rule
4. docs/PROGRESS_EN.md through MANAGER-DECISION-030
5. docs/NEXT_TASK_EN.md
6. configs/wsm_mm_pd_dep_v1/fusion/03_f1_temporal_audio_residual.yaml
7. src/fusion/models/frozen_audio_temporal_adapter.py
8. src/fusion/models/av_f1_temporal_audio_residual.py
9. src/fusion/models/av_f2_task_aware_directed.py
10. src/common/optimizers.py
11. src/fusion/data/wsm_av_fusion_datamodule.py
12. src/common/callbacks/wsm_segment_callback.py
13. src/common/loss/wsm_masked_sparse_loss.py
14. src/chimera_plugin.py

## Allowed Tracked Files

- src/fusion/models/av_audio_first_zero_residual.py
- src/fusion/models/av_audio_query_temporal_video.py
- src/fusion/models/av_audio_confidence_gated.py
- src/chimera_plugin.py
- configs/wsm_mm_pd_dep_v1/fusion/04_audio_first_zero_residual.yaml
- configs/wsm_mm_pd_dep_v1/fusion/05_audio_query_temporal_video.yaml
- configs/wsm_mm_pd_dep_v1/fusion/06_audio_confidence_gated.yaml
- docs/PROGRESS_EN.md

No other tracked file may be modified.

Existing accepted models, especially:

    src/fusion/models/av_f1_temporal_audio_residual.py
    src/fusion/models/av_f2_task_aware_directed.py

must remain unchanged.

## Fixed Historical Audio Anchor

Exact checkpoint:

    logs/wsm_audio_segment_wavlm_base_l9_pool4/multitask_audio_mamba_2026-08-19_14-12_audio_mamba_segment_model_wsm_audio_models-e0ce-006_ea8c8d77/checkpoints/epoch=4_dev_mean_score=0.7878.pt

Required SHA256:

    0873c7cb5e32d415cdd301058949f5dd140c16b874cc5230d027a4ee33e3daf2

All three candidates MUST use:

    FrozenAudioTemporalAdapter

or a thin fusion-side wrapper around it.

The historical audio model must remain:

- fully frozen;
- eval-only even under parent model.train();
- absent from optimizer groups;
- internally task-conditioned only by its fixed disease indices;
- unchanged on disk and in src/audio.

No candidate may use pooled audio_cls as its audio anchor.

## Mandatory Candidate Families

You must implement and train exactly these THREE candidate families.

### Candidate A — Zero-Initialized Audio-Anchored Residual

Registry key:

    wsm_av_audio_first_zero_residual_model

File:

    src/fusion/models/av_audio_first_zero_residual.py

This is the simplest no-harm control.

Required idea:

    audio_base_logits
    +
    zero-initialized task residual from
        frozen audio task feature
        + video representation

You may improve the hidden MLP organization relative to TASK-004I3, but:

- final correction layer for each task MUST be zero-initialized;
- initial predictions MUST equal audio_base_logits exactly;
- video may be masked-mean or learned pooled;
- trainable parameter count excluding frozen audio MUST be <= 500,000.

### Candidate B — Audio-Query Temporal-Video Attention Residual

Registry key:

    wsm_av_audio_query_temporal_video_model

File:

    src/fusion/models/av_audio_query_temporal_video.py

This candidate MUST use the temporal video sequence, not only masked-mean video.

Required idea:

- frozen temporal-audio task feature is the query;
- projected temporal video frames are key/value;
- attention is task-specific by query even if projection/attention parameters are shared;
- attended video context produces a task residual;
- final task correction layer is zero-initialized;
- initial predictions equal audio_base_logits exactly.

Use the existing video mask.

Do not add a new video encoder family or modify cached features.

You may choose the exact attention block and head count, but:

- hidden width must remain <= 256;
- attention heads must divide the hidden width;
- trainable parameter count excluding frozen audio MUST be <= 1,000,000.

### Candidate C — Audio-Confidence-Gated Video Correction

Registry key:

    wsm_av_audio_confidence_gated_model

File:

    src/fusion/models/av_audio_confidence_gated.py

This candidate explicitly gives more authority to the stronger audio model.

Required idea:

    final_logit_t =
        audio_base_logit_t
        + gate_t * residual_t

where gate_t is learned per task and conditioned on the frozen audio evidence.

The gate MUST use at least:

- frozen audio task feature;
- magnitude or equivalent confidence information derived from the frozen audio base logit.

It MAY also use:

- temporal-video attention context;
- pooled video context;
- one F2-like directed relation feature.

The gate must remain bounded:

    0 <= gate_t <= 1

Use sigmoid or another explicit bounded construction.

The residual output layer MUST be zero-initialized, so initial predictions still equal the frozen audio base exactly.

The intended behavior is:

- when audio is already confident, the model may learn to suppress video correction;
- when audio is uncertain, it may allow stronger video correction.

Do not hard-code a heuristic confidence threshold.

Trainable parameter count excluding frozen audio MUST be <= 1,000,000.

## Global Candidate Invariants

All three candidates MUST:

- output preds [B,2] ordered [depression, parkinson];
- use independent binary logits, no sigmoid in forward;
- consume no external task_id/task_ids;
- use observed sparse labels only;
- use wsm_masked_sparse_loss;
- require audio availability;
- support video availability masks;
- fall back exactly to frozen audio base when video is unavailable;
- initialize with residual correction exactly zero for video-available rows too;
- keep frozen audio parameter grads None;
- remain compatible with wsm_trainable_adamw_optimizer.

At construction, exact tensor identity is required:

    torch.equal(output.preds, output.aux["audio_base_logits"])

for every candidate.

## SEARCH MANIFEST — Freeze Designs Before Training

Before the FIRST full training run:

1. implement all three candidate models;
2. compile/register/smoke all three;
3. create all three fixed configs;
4. append a SEARCH MANIFEST section to docs/PROGRESS_EN.md.

The SEARCH MANIFEST must record, for A/B/C:

- registry key;
- exact forward equation;
- exact video representation/attention path;
- gate equation where applicable;
- hidden dimensions;
- attention heads if applicable;
- trainable scalar parameter count;
- confirmation residual output is zero-initialized;
- exact config path;
- run order.

The manifest must explicitly state:

    "Candidate architectures are frozen before the first metric-bearing training run."

After the first full training run begins:

- do NOT change any candidate model source;
- do NOT change any candidate config;
- do NOT add a fourth candidate;
- do NOT use candidate A results to redesign B or C;
- do NOT use Test results for any reason.

If a source/config correctness bug prevents a candidate from running after the manifest is frozen, stop the task as partially blocked and report the exact candidate. Do not redesign around its metrics.

## Fixed Config/Training Protocol

Create:

    configs/wsm_mm_pd_dep_v1/fusion/04_audio_first_zero_residual.yaml
    configs/wsm_mm_pd_dep_v1/fusion/05_audio_query_temporal_video.yaml
    configs/wsm_mm_pd_dep_v1/fusion/06_audio_confidence_gated.yaml

All three configs use exactly:

    seed: 42
    experiment_name: wsm_mm_pd_dep_v1

Data:

    name: wsm_av_fusion_datamodule

Data params:

    data_root: /media/maxim/Databases/WSM_NEW
    audio_feature_cache_root: /media/maxim/Databases/WSM_NEW/features
    video_cache_root: /media/maxim/Programs/Features/WSM/video_depart_v1_fullframe_fallback/cache
    batch_size: 8
    num_workers: 4
    pin_memory: true
    persistent_workers: true
    shuffle_train: true
    drop_last_train: false

Candidate-specific run names:

    A: av_audio_first_zero_residual
    B: av_audio_query_temporal_video
    C: av_audio_confidence_gated

Each model must receive the exact frozen audio checkpoint path.

Loss:

    name: wsm_masked_sparse_loss

Optimizer:

    name: wsm_trainable_adamw_optimizer
    lr: 0.0001
    weight_decay: 0.01

Training:

    epochs: 30
    device: cuda
    mixed_precision: true
    grad_clip_norm: 0.5
    log_every_steps: 25
    collect_cache: true

Metrics:

    []

Required instrumentation for every config:

- wsm_segment_metrics_callback with splits=[auto]
- checkpoint_callback
- snapshot_callback
- early_stopping_callback
- wsm_summary_callback
- console_file_logger
- mlflow_logger

Checkpoint:

    monitor: dev/mean_score
    mode: max
    save_top_k: 2
    save_last: true

Early stopping:

    monitor: dev/mean_score
    mode: max
    patience: 6
    min_delta: 0.0005

MLflow:

    tracking_uri: sqlite:///logs/mlflow.db

No scheduler.
No pseudo-labeling.
No task loss reweighting.
No audio fine-tuning.
No threshold tuning.

### OOM exception

Default batch size is 8 for all candidates.

Only if Candidate B or C is CUDA-OOM specifically because of temporal attention:

- you may retry that candidate once with batch_size=4;
- record the full OOM traceback and the batch-size fallback;
- do not alter any other hyperparameter;
- do not reduce sequence content or architecture after seeing metrics.

No other training hyperparameter tuning is allowed.

## Pre-Run Gate for Every Candidate

Before any full run, each candidate must pass:

- registry construction;
- exact checkpoint SHA;
- frozen audio eval/no-grad;
- optimizer_ids == trainable_ids;
- optimizer/frozen intersection empty;
- trainable parameter cap;
- real production batch forward/loss/backward;
- no external task IDs;
- finite gradients after wake-up.

Because every candidate starts with a zero final correction layer:

1. fresh-model initial forward must have:
       residual/correction == 0 exactly
       preds == audio_base_logits exactly
2. first backward may have zero upstream gradients;
3. perform one bounded in-memory optimizer step for smoke only;
4. on second backward require finite nonzero gradients through every intended trainable path;
5. discard this smoke model and use a fresh model for the full run.

Do not save smoke weights.

## Exact Frozen-Audio DEV Initialization Gate

Before each candidate training run, use a FRESH candidate model and canonical DEV only.

Require final initialized predictions to reproduce:

    depression Score = 0.7479183895
    Parkinson Score = 0.8277353635
    Mean_Score = 0.7878268765

with absolute tolerance <= 0.0005.

Additionally require:

    preds == audio_base_logits

exactly on every DEV batch before training.

No Test loader may be iterated for this gate.

## Full Training Runs

After SEARCH MANIFEST is frozen and all three candidates pass pre-run gates, run exactly three metric-bearing experiments in the declared order.

For each candidate:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train \
      --config-path <candidate-config>

Do not interrupt a healthy run.

Do not edit later candidates after seeing earlier candidate metrics.

## Mandatory Metrics Every Epoch

For every completed epoch of every candidate record:

- train loss;

DEV:
- depression UAR/MF1/Score;
- Parkinson UAR/MF1/Score;
- Mean_Score;

TEST_NONE:
- depression UAR/MF1/Score;
- Parkinson UAR/MF1/Score;
- Mean_Score;

TEST_SOFT:
- depression UAR/MF1/Score;
- Parkinson UAR/MF1/Score;
- Mean_Score;

TEST_HARD:
- depression UAR/MF1/Score;
- Parkinson UAR/MF1/Score;
- Mean_Score.

All must be finite.

Best epoch per candidate:

    argmax(dev/mean_score)

Training loss and Test metrics are NOT selectors.

## Candidate Ranking

After all three runs complete, rank candidates ONLY by each candidate's best:

    DEV/Mean_Score

Do not rank by Test.

Frozen audio comparator:

    Mean_Score = 0.7878268765
    depression Score = 0.7479183895
    Parkinson Score = 0.8277353635

For each candidate report:

    candidate DEV Mean - frozen audio DEV Mean
    candidate depression Score - frozen audio depression Score
    candidate Parkinson Score - frozen audio Parkinson Score

Define a "safe single-seed screen winner" only if BOTH are true:

1. candidate DEV Mean_Score > 0.7878268765
2. neither task Score is more than 0.010 below the frozen-audio task Score

This is only a screening nomination, not final promotion.

If multiple candidates satisfy this, nominate exactly one with the highest DEV Mean_Score.

If no candidate satisfies it:

- nominate none;
- preserve all three negative results;
- do NOT add a fourth candidate in this task.

## Selected-Checkpoint Same-Row Audits

For every candidate's DEV-selected checkpoint, run the same-row canonical DEV audit:

Collect:

    final_logits
    audio_base_logits
    candidate correction/residual
    targets
    observed_mask

Report:

- frozen-base task UAR/MF1/Score and Mean;
- final task UAR/MF1/Score and Mean;
- final-minus-base task Score deltas and Mean delta;
- residual/correction mean/std/mean_abs/min/max per task;
- corrected errors;
- introduced errors;
- sign flips.

For Candidate C also report:

- gate mean/std/min/max per task;
- gate mean on base-correct rows;
- gate mean on base-wrong rows;
- correlation between gate and absolute frozen audio base logit, if straightforward to compute without source changes.

Diagnostics are descriptive only.

## Run Provenance

For every candidate record:

- config path;
- registry key;
- implementation SHA;
- GPU;
- trainable parameter count;
- actual batch size;
- epochs completed;
- best epoch;
- best DEV metrics;
- same-epoch Test metrics;
- top-2 checkpoints;
- last.pt;
- run directory;
- train.log;
- summary.txt;
- code.zip;
- resolved config;
- MLflow experiment;
- MLflow run ID;
- MLflow final status;
- artifact URI.

## Test Firewall

TEST_NONE/SOFT/HARD must be computed every epoch as required by the project.

They MUST NOT influence:

- candidate architecture;
- candidate order;
- whether a candidate is considered successful;
- early stopping;
- checkpoint selection;
- candidate ranking;
- nomination;
- next-step recommendation.

The candidate manifest must be frozen before any Test metric from this task exists.

## Acceptance Criteria

Pass if:

- exactly three candidate families are implemented;
- no fourth candidate is added;
- all use the exact frozen historical temporal audio checkpoint;
- all start exactly at frozen audio predictions;
- audio remains frozen/eval-only and absent from optimizers;
- at least Candidate B uses temporal video frames;
- Candidate C uses bounded audio-confidence gating;
- all trainable parameter caps pass;
- SEARCH MANIFEST is frozen before first training run;
- no candidate source/config changes occur after first run begins;
- all three runs complete or configured early stopping ends them, unless a genuine runtime blocker is preserved;
- all four metric streams are recorded every epoch;
- each candidate selects only by dev/mean_score;
- ranking uses DEV only;
- same-row audits are complete;
- all run provenance is complete;
- no Test-driven search occurs;
- src/audio unchanged;
- src/video unchanged;
- accepted DataModule unchanged;
- RAMPS/text/description not started;
- tracked diff contains only the eight allowed paths;
- branch codex/task-004k committed and pushed;
- main/master untouched.

## Required PROGRESS_EN Update

Append a complete TASK-004K record.

It must include:

1. SEARCH MANIFEST before metric-bearing runs:
   - exact A/B/C architecture equations;
   - hidden sizes/heads;
   - parameter counts;
   - config paths;
   - run order;
   - explicit frozen-before-training statement.

2. Pre-run evidence per candidate:
   - exact baseline-at-init equality;
   - DEV frozen-audio reproduction;
   - optimizer firewall;
   - two-step gradient wake-up smoke;
   - parameter cap.

3. FULL epoch tables for A, B, and C.

4. Best selected DEV metrics and same-epoch Test metrics for A/B/C.

5. Same-row frozen-base versus final DEV audits for A/B/C.

6. Candidate C gate diagnostics.

7. DEV-only candidate ranking.

8. Safe-winner test against:
       Mean > 0.7878268765
       task drops <= 0.010

9. Exact MLflow/checkpoint/artifact provenance for all three runs.

10. Explicit statement that Test metrics did not influence search/ranking.

11. Confirmation:
    - no fourth candidate;
    - audio stayed frozen;
    - src/audio unchanged;
    - src/video unchanged;
    - RAMPS/text/description deferred.

## Required Handoff

Respond in English using exactly:

1. Outcome
2. Changed files
3. Verification
4. Plan status
5. Blockers and risks
6. Next atomic step

Explicitly include:

- branch codex/task-004k;
- final implementation/evidence commit SHA;
- pushed-to-origin status;
- main/master untouched;
- the three chosen/frozen candidate architecture summaries;
- trainable parameter counts;
- epochs/best DEV Mean for all three;
- frozen-audio DEV delta for all three;
- task Score deltas for all three;
- whether any candidate met the safe-winner rule;
- nominated candidate if any;
- same-epoch TEST_NONE/SOFT/HARD Mean_Scores for all three;
- MLflow run IDs/status for all three;
- explicit Test-firewall statement;
- frozen-audio integrity;
- no fourth candidate;
- RAMPS/text/description deferred;
- src/audio unchanged;
- src/video unchanged.

If one candidate meets the safe-winner rule:

    recommended next = manager-assigned 3-seed confirmation of that single candidate only.

If none meets the safe-winner rule:

    recommended next = manager review of the negative bounded search; do not add another fusion candidate automatically.

Stop after this task.
