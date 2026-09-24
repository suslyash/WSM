# TASK-002L2: Replace Temporary V1 Training Artifact Paths with Durable WSM Log Paths

## Role

You are the implementing Codex. Execute only this corrective task, update docs/PROGRESS_EN.md, commit on the required task branch, push to origin, then stop. Follow AGENTS.md.

Required branch:

    codex/task-002l2

Do not run the real/full V1 training experiment in this task.

## Goal

Correct the only remaining TASK-002L pre-training defect.

The accepted production config:

    configs/wsm_mm_pd_dep_v1/video/00_depart_v1.yaml

currently writes checkpoints, snapshots, console logs, and MLflow state under:

    /tmp/wsm_task002l/

Those paths were appropriate only for the bounded integration smoke and are not acceptable for the real 30-epoch experiment.

Replace them with durable project log paths following the existing WSM config convention.

After this correction, the config must be ready for TASK-002M real V1 training.

## Required Reading

1. AGENTS.md
2. docs/PROJECT_REQUIREMENTS.md Sections 5, 8, 9, 13, 14
3. docs/PLAN.md Stage 2
4. docs/PROGRESS_EN.md through MANAGER-DECISION-012
5. docs/NEXT_TASK_EN.md
6. configs/wsm_mm_pd_dep_v1/video/00_depart_v1.yaml
7. configs/wsm_mm_pd_dep_v1/audio/00_frozen_baseline.yaml as the durable logging-path reference

## Allowed Tracked Files

- configs/wsm_mm_pd_dep_v1/video/00_depart_v1.yaml
- docs/PROGRESS_EN.md

No other tracked file may be modified.

Do not modify src/audio.
Do not modify any Python source.
Do not modify chimera-ml.

## Required Config Correction

Preserve every already-accepted training/model/data/selector parameter.

Change only artifact/logging locations so the production config uses durable project paths.

Use the same durable convention as the existing audio baseline:

Checkpoint callback:

    log_path: logs

Snapshot callback:

    log_path: logs

Console logger:

    log_path: logs

MLflow logger:

    tracking_uri: sqlite:///logs/mlflow.db

Do not leave any production field pointing to:

    /tmp/wsm_task002l

Do not introduce a new temporary path elsewhere.

Preserve:

- experiment_name=wsm_mm_pd_dep_v1
- run_name=depart_v1_clip_yolo_transformer
- accepted full-coverage cache root
- all model parameters
- optimizer lr/weight_decay
- train epochs/device/mixed precision/grad clip/log cadence
- collect_cache=true
- four evaluation streams through the DataModule
- wsm_segment_metrics_callback splits=[auto]
- checkpoint monitor=dev/mean_score, mode=max
- early stopping monitor=dev/mean_score, mode=max
- required instrumentation list

Do not add a scheduler.
Do not add generic sparse-incompatible metrics.

## Acceptance Criteria

- config contains no /tmp/wsm_task002l path;
- checkpoint log_path is durable;
- snapshot log_path is durable;
- console logger log_path is durable;
- MLflow SQLite path is durable;
- all accepted TASK-002L settings remain unchanged otherwise;
- checkpoint monitor remains exactly dev/mean_score;
- early stopping monitor remains exactly dev/mean_score;
- no Test metric is used as selector/monitor;
- config validation passes;
- registry/build smoke passes;
- four DataModule evaluation stream names remain dev/test_none/test_soft/test_hard;
- no full training run;
- no dependency install;
- src/audio unchanged;
- git diff --check passes;
- branch codex/task-002l2 committed and pushed;
- main/master untouched;
- tracked diff contains only the two allowed paths.

## Exact Verification Commands

Confirm no temporary production path remains:

    ! grep -Rni '/tmp/wsm_task002l' configs/wsm_mm_pd_dep_v1/video/00_depart_v1.yaml

Validate config:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml validate-config \
      --config-path configs/wsm_mm_pd_dep_v1/video/00_depart_v1.yaml

Selector firewall:

    grep -n 'monitor: dev/mean_score' \
      configs/wsm_mm_pd_dep_v1/video/00_depart_v1.yaml

    grep -RniE 'monitor:.*test_(none|soft|hard)|scheduler_monitor:.*test_(none|soft|hard)' \
      configs/wsm_mm_pd_dep_v1/video/00_depart_v1.yaml && exit 1 || true

Durable path assertions:

    grep -n 'log_path: logs' \
      configs/wsm_mm_pd_dep_v1/video/00_depart_v1.yaml

    grep -n 'tracking_uri: sqlite:///logs/mlflow.db' \
      configs/wsm_mm_pd_dep_v1/video/00_depart_v1.yaml

Run the same actual Chimera registry/build smoke used in TASK-002L, but do not call Trainer.fit and do not start the full experiment.

Assert:

- DataModule builds;
- model builds;
- loss builds;
- optimizer builds;
- callbacks build;
- loggers build;
- val_dataloader() keys are exactly:
  dev, test_none, test_soft, test_hard.

Finally:

    git diff --check
    git diff -- src/audio
    git status --short

Before commit inspect only:

    git diff -- \
      configs/wsm_mm_pd_dep_v1/video/00_depart_v1.yaml \
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
- exact durable paths now used;
- confirmation no /tmp/wsm_task002l remains in the production YAML;
- config validation result;
- registry/build result;
- four evaluation stream names;
- selector monitor/mode;
- confirmation no Test selector;
- confirmation no full training run;
- src/audio unchanged;
- Stage 2 is now ready for TASK-002M real V1 video training if all acceptance criteria pass;
- recommended next atomic step: TASK-002M real V1 video training run.

## Required Handoff

Respond in English using exactly:

1. Outcome
2. Changed files
3. Verification
4. Plan status
5. Blockers and risks
6. Next atomic step

Explicitly include:

- branch codex/task-002l2;
- implementation commit SHA;
- pushed-to-origin status;
- main/master untouched;
- corrected durable paths;
- config validation result;
- selector key;
- four validation stream names;
- no full training;
- src/audio unchanged.

Stop after this task.
