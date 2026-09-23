# WSM Project Progress

## 1. Current State

Plan initialized: 2026-09-23.

Current stage: **Stage 0 complete**.

Final Test authorized: **no**.

## 2. Stage Status

| Stage | Status | Gate | Evidence |
|---|---|---|---|
| Paper/project analysis | complete | Baselines, structure, requirements, plan | BASELINES.md, PROJECT_INIT_STRUCTURE.md, PROJECT_REQUIREMENTS.md, PLAN.md |
| 0. Reproducible base | complete | Canonical frozen audio config validates and registry/smoke gates pass | Verification records TASK-000-A and TASK-000B below |
| 1. Manifest/partial-label contract | not started | Audit and masks | None |
| 2. Video | not started | At most two families; DEV winner | None |
| 3. Text/description | not started | At most two families; prompt audit | None |
| 4. Fusion baselines | not started | Comparable F0/F1/F2 | None |
| 5. RAMPS | not started | Direct reliable missing-head gradient | None |
| 6. Ablations | not started | Claims backed by multi-seed evidence | None |
| 7. Final evaluation | locked | Config freeze and manager authorization | None |

## 3. Frozen Historical Audio Reference

- run: wsm_audio_models-e0ce-006;
- selected epoch: 4;
- DEV/Mean_Score: 0.787827;
- DEV depression Score: 0.747918;
- DEV Parkinson Score: 0.827735;
- historical TEST_NONE/SOFT/HARD Mean_Score: 0.809486 / 0.815156 / 0.828135;
- WavLM-base-plus layer 9, pool 4;
- temporal Transformer: hidden 192, 3 layers, 4 heads, 128 steps.

Historical Test values came from an existing summary; they were not used for a new selection.

## 4. Open Blockers

1. The existing AV YAML is legacy/non-runnable: wsm_segment_datamodule and wsm_avsync_loss are absent from the registry.
2. The AV YAML also lacks snapshot_callback and early_stopping_callback and monitors dev/mean_macro_f1 instead of dev/mean_score.
3. The current audio datamodule still includes Test loaders in each validation epoch; this task did not change it.
4. No global multilabel manifest/observed-task mask exists.

## 5. Execution Log

### DOC-001 — Analysis and research plan

Status: complete.

- Audited the four supplied papers, SOTA review, source, configs, saved summaries, and snapshot.
- Added project requirements, staged RAMPS plan, manager contract, progress ledger, and atomic task file.
- Corrected root .gitignore so project docs/configs/scripts can be tracked.
- No Python source, training, or new Test evaluation was performed in that step.

### TASK-000-A — Verify user-restored audio provider

Status: partial.

Observed changes:

- src/fusion/models/av_sync_mamba_segment.py restored from the historical snapshot with additional autocast handling;
- src/fusion package initializers restored;
- src/chimera_plugin.py explicitly imports fusion.models.av_sync_mamba_segment;
- src/audio has no diff.

Verification:

- Commands used:
  - env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -c <plugin/registry assertions>;
  - env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -c <synthetic forward/loss/backward>;
  - .venv/bin/chimera-ml validate-config -c configs/audio_experiments/wsm_audio_mamba_multitask.yaml;
  - .venv/bin/chimera-ml validate-config -c configs/audio_experiments/wsm_avsyncmamba_multitask.yaml;
  - .venv/bin/chimera-ml registry list;
  - python3 -m py_compile on the restored provider, audio model, and plugin;
  - git diff --check and git diff -- src/audio.
- plugin registration emitted no warnings;
- MODELS contains audio_mamba_segment_model and av_sync_mamba_segment_model;
- DATAMODULES contains wsm_audio_segment_datamodule;
- LOSSES contains wsm_audio_loss;
- required audio callbacks are registered;
- AudioMambaSegmentModel instantiated with the Transformer path;
- synthetic CPU forward/loss/backward passed: preds shape (4, 2), loss 0.7533376813, all gradients finite;
- both supplied YAML files pass chimera-ml validate-config structural validation;
- git diff --check passed before documentation translation;
- git diff -- src/audio was empty;
- no training and no new Test evaluation ran.

Assessment:

- the broken audio import/registry blocker is fixed;
- the Stage 0 gate is not complete because the canonical audio config is absent;
- the legacy AV config must not be treated as runnable or as the new fusion baseline.

Recommended next task: TASK-000B in NEXT_TASK.md.


### TASK-000B — Canonicalize the frozen audio baseline

Status: complete.

Changed files:

- configs/wsm_mm_pd_dep_v1/audio/00_frozen_baseline.yaml;
- scripts/run_audio_experiments.sh (default BASE now points to the canonical config);
- docs/PROGRESS_EN.md.

Implementation:

- Copied the verified source config from configs/audio_experiments/wsm_audio_mamba_multitask.yaml.
- Set experiment_name to wsm_mm_pd_dep_v1 and run_name to frozen_audio_wavlm_l9_pool4.
- Preserved WavLM-base-plus layer 9, temporal_pool 4, Transformer hidden_dim 192 / 3 layers / 4 heads / sequence_steps 128 / dropout 0.25, label_smoothing 0.02, focal_gamma 1.0, aux_weight 0.10, AdamW lr 1e-4 / weight_decay 0.01, and seed 42.
- Used the registered wsm_segment_metrics_callback and retained checkpoint, snapshot, summary, early stopping, console-file, and MLflow instrumentation.
- checkpoint_callback and early_stopping_callback both monitor dev/mean_score in max mode.
- The supplied AV config was not modified; it remains legacy/non-runnable as documented above.

Exact verification commands/results:

- .venv/bin/chimera-ml validate-config -c configs/wsm_mm_pd_dep_v1/audio/00_frozen_baseline.yaml — passed: Config is valid.
- PYTHONPATH=src .venv/bin/python -c '<config/callback/logger/registry assertions and synthetic smoke>' — passed: all referenced audio registry keys exist; required callbacks/loggers and monitor values asserted; synthetic CPU forward/loss/backward passed; no training/Test evaluation.
- Plugin registration in the same command emitted no project-module warning. It emitted only the existing PyTorch nested-tensor warning.
- git diff --check — passed.
- git diff -- src/audio — empty.
- git status --short and full diff audit — only the canonical config, launcher default, and this progress entry are in scope.

No training ran. No Test metrics were inspected or used.

Recommended next atomic task: begin Stage 1 manifest/partial-label contract only after manager approval; do not advance within this task.
