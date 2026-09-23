# TASK-000B: Canonicalize the Frozen Audio Baseline

## Role

You are the implementing Codex. Execute only this task, update PROGRESS_EN.md, then stop. Follow AGENTS.md.

## Read First

1. docs/PROJECT_REQUIREMENTS.md;
2. Stage 0 in docs/PLAN.md;
3. docs/PROGRESS_EN.md;
4. relevant sections of docs/PROJECT_INIT_STRUCTURE.md;
5. configs/audio_experiments/wsm_audio_mamba_multitask.yaml;
6. src/chimera_plugin.py.

## Goal

Finish the remaining Stage 0 audio-config gate. Preserve the user's restored provider and keep src/audio unchanged.

## Allowed Changes

- configs/wsm_mm_pd_dep_v1/audio/00_frozen_baseline.yaml;
- scripts/audio/run_frozen_baseline.sh, only if a launcher is necessary;
- scripts/run_audio_experiments.sh, only to point it to the canonical config without changing experiment semantics;
- docs/PROGRESS_EN.md.

Do not edit the restored fusion provider or plugin unless verification reveals a concrete defect that blocks the canonical audio config. If such a defect appears, stop and report it instead of broadening scope.

## Source Config

Use:

    configs/audio_experiments/wsm_audio_mamba_multitask.yaml

Preserve the verified best-run parameters:

- WavLM-base-plus, layer 9, temporal_pool 4;
- Transformer, hidden_dim 192, 3 layers, 4 heads, sequence_steps 128, dropout 0.25;
- label_smoothing 0.02, focal_gamma 1.0, aux_weight 0.10;
- AdamW lr 1e-4, weight_decay 0.01;
- seed 42.

The canonical config MUST:

- live at configs/wsm_mm_pd_dep_v1/audio/00_frozen_baseline.yaml;
- set experiment_name to wsm_mm_pd_dep_v1;
- set run_name to frozen_audio_wavlm_l9_pool4;
- remain self-contained;
- include checkpoint_callback, snapshot_callback, early_stopping_callback, wsm_summary_callback, and a segment metrics callback;
- include console_file_logger and mlflow_logger;
- monitor dev/mean_score in max mode for checkpointing and early stopping.

## Legacy AV Config

Do not repair configs/audio_experiments/wsm_avsyncmamba_multitask.yaml in this task. Record it as legacy/non-runnable because wsm_segment_datamodule and wsm_avsync_loss are not registered and its instrumentation/monitor violate current requirements.

## Forbidden

- any src/audio change;
- audio tuning or training;
- Test evaluation;
- video/text/description/fusion development;
- restoration of the old AV datamodule/loss;
- fixing Test-in-validation in the old audio datamodule;
- changing labels, splits, or metric formulas.

## Acceptance Criteria

- canonical YAML exists at the required path;
- chimera-ml validate-config passes;
- every referenced audio registry key exists;
- required callbacks/loggers and monitor values are asserted;
- plugin registration has no project-module warning;
- synthetic audio forward/loss/backward still passes;
- git diff --check passes;
- git diff -- src/audio is empty;
- no training or Test evaluation ran.

## Handoff

Update docs/PROGRESS_EN.md with exact commands/results. Respond using the AGENTS.md manager-handoff format and stop before Stage 1.
