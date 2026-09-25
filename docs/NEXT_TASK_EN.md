# TASK-005D: Run the First Bounded Seed-42 R2 Student Experiment

## Task Identifier and Title

TASK-005D — Run exactly one bounded seed-42 R2 student experiment using the frozen semantic pseudo cache and frozen warm-up schedule, with DEV-only selection and no method tuning.

Required branch:

    codex/task-005d

Start from current `origin/main`, which includes:

- PR #40 merge `62f2bf84d50d02f0b13ffa15b52dc8b35bd19a51`;
- manager decision commit `9ebe541b156adcd53ac92adc795f70e8b6a10b0e`.

Create exactly one branch from that `origin/main`. Run exactly one production student experiment, record the evidence, commit the allowed tracked files, push, and stop.

## Goal

Measure the isolated effect of accepted RAMPS R2 direct pseudo-supervision on the existing F2 A+V student.

Do not change model architecture, pseudo cache, warm-up, optimizer, thresholds, teachers, or selection policy.

This is one seed-42 screening run only. It is not a multi-seed confirmation and does not complete Stage 5.

## Required Reading

Read in this order:

1. AGENTS.md
2. docs/README.md
3. docs/PROJECT_REQUIREMENTS.md, especially Sections 8, 9, 11, 12, 13, 14
4. docs/PLAN.md, especially Stage 5
5. docs/PROGRESS_EN.md through MANAGER-DECISION-039
6. docs/NEXT_TASK_EN.md
7. configs/wsm_mm_pd_dep_v1/fusion/02_f2_task_aware_directed.yaml
8. configs/wsm_mm_pd_dep_v1/fusion/04_ramps_r2_warmup_contract.yaml
9. src/fusion/data/wsm_ramps_semantic_datamodule.py
10. src/fusion/loss/ramps_observed_pseudo_loss.py
11. src/common/callbacks/wsm_pseudo_scale_warmup_callback.py
12. src/fusion/loss/ramps_r1_teacher.py
13. src/fusion/models/av_f2_task_aware_directed.py
14. src/chimera_plugin.py

Use only active English docs listed in docs/README.md.

## Allowed Files

Codex may modify only:

- configs/wsm_mm_pd_dep_v1/fusion/05_ramps_r2_student_seed42.yaml
- docs/PROGRESS_EN.md

No source file may change.

## Forbidden Actions

- Do not modify src/audio, src/video, src/fusion, src/common, or src/chimera_plugin.py.
- Do not modify the pseudo cache or any Stage-5 semantic artifact.
- Do not regenerate/reselect pseudo labels.
- Do not change accepted counts/classes.
- Do not change precision_target, min_support, folds, prompts, semantic rules, Parkinson rules, or calibrators.
- Do not change the warm-up schedule.
- Do not change the F2 architecture or parameter count.
- Do not change optimizer/lr/weight decay/batch size/grad clipping/patience based on run results.
- Do not launch a second seed or second training run.
- Do not resume/restart with altered settings after seeing results.
- Do not use TEST_NONE/SOFT/HARD to select epoch, tune, stop, rank, or decide follow-up.
- Do not run a separate Final Test.
- Do not start R3/R4, Stage 6, Stage 7, or general Text/Description Stage 3.
- Do not claim recovered true missing labels or comorbidity.

## Frozen Comparator and Screen

Sparse F2 seed-42 baseline:

- selected epoch: 5
- checkpoint:
  `logs/wsm_mm_pd_dep_v1/av_f2_task_aware_directed_2026-09-24_18-18_wsm_av_f2_task_aware_directed_model_f1b9902d/checkpoints/epoch=5_dev_mean_score=0.7746.pt`
- DEV depression Score: `0.697035`
- DEV Parkinson Score: `0.852104`
- DEV Mean_Score: `0.774569`

Frozen historical audio reference, context only:

- DEV depression Score: `0.7479183895`
- DEV Parkinson Score: `0.8277353635`
- DEV Mean_Score: `0.7878268765`

Predeclared single-seed continuation screen for direct pseudo-supervision:

- selected student DEV Mean_Score must be strictly greater than `0.774569`;
- depression DEV Score must be at least `0.687035`;
- Parkinson DEV Score must be at least `0.842104`.

This screen is fixed before the run.

If it fails, record a valid negative result and stop. Do not alter the cache, schedule, thresholds, optimizer, or model in this task.

## 1. Production Student Config

Add:

    configs/wsm_mm_pd_dep_v1/fusion/05_ramps_r2_student_seed42.yaml

It MUST be identical in all training/model/data/loss/callback/logger semantics to:

    configs/wsm_mm_pd_dep_v1/fusion/04_ramps_r2_warmup_contract.yaml

except:

    experiment_info.params.run_name: ramps_r2_student_seed42

Required frozen values include:

- seed 42;
- experiment_name `wsm_mm_pd_dep_v1`;
- `wsm_ramps_semantic_datamodule`;
- frozen pseudo cache path;
- existing `wsm_av_f2_task_aware_directed_model`;
- F2 dims `768/512/192/192/192`, dropout `0.2`, tasks `2`;
- `wsm_ramps_observed_pseudo_loss`, initial `pseudo_scale=0.0`;
- warm-up callback `observed_only_epochs=3`, `ramp_epochs=5`, `final_scale=1.0`;
- AdamW lr `1e-4`, weight decay `0.01`;
- batch size 32;
- epochs ceiling 30;
- mixed precision true;
- grad clip 0.5;
- early stopping patience 6/min_delta 0.0005;
- checkpoint and early stopping monitor only `dev/mean_score`, mode=max;
- required callback/logger stack.

Do not make the production config a new hyperparameter variant.

## 2. Pre-Run Firewall

Before training:

1. validate the config;
2. register/build DataModule/model/loss/callbacks/loggers;
3. assert canonical counts:
   - train 6325
   - dev 933
   - test_none 1364
   - test_soft 1208
   - test_hard 1014
4. assert cache SHA:
   `17cf5e67e8c244d81b7c21f0842988966a23d83d69e296f7c5a5181376d6b945`;
5. assert accepted counts:
   - depression 376
   - Parkinson 1801
6. assert the warm-up epoch vector for epochs 1-10 remains:
   `[0,0,0,0.2,0.4,0.6,0.8,1,1,1]`;
7. assert checkpoint/early-stopping selectors reference no Test key;
8. compare the model section against the frozen F2 config and prove the trainable parameter count is identical to a model built from `02_f2_task_aware_directed.yaml`;
9. run one real pre-training forward/loss/backward batch with finite gradients;
10. do not optimizer-step in this firewall smoke.

If any firewall assertion fails, stop blocked before production training.

## 3. Exact One-Run Training Authorization

If and only if the firewall passes, run exactly:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train       --config-path configs/wsm_mm_pd_dep_v1/fusion/05_ramps_r2_student_seed42.yaml

Exactly one production invocation is authorized.

Expected behavior:

- seed 42;
- max 30 epochs;
- early stopping may end sooner;
- every epoch evaluates separate DEV, TEST_NONE, TEST_SOFT, TEST_HARD streams as required by PROJECT_REQUIREMENTS;
- `dev/mean_score` alone controls checkpointing and early stopping;
- `train/pseudo_scale` follows the frozen warm-up.

Do not manually terminate because of Test behavior.

Do not rerun to improve the result.

## 4. Selection Firewall

After the run completes:

1. determine the selected checkpoint/epoch solely from maximum `dev/mean_score`;
2. freeze the selected epoch before interpreting any same-epoch Test values;
3. record full DEV depression/Parkinson UAR, MF1, Score, and Mean_Score for the selected epoch;
4. compute deltas versus sparse F2:
   - depression Score delta;
   - Parkinson Score delta;
   - Mean_Score delta;
5. evaluate the predeclared continuation screen exactly;
6. only after the DEV selection/screen is frozen, report same-epoch TEST_NONE/SOFT/HARD monitoring metrics if present in the normal epoch log;
7. Test values must not alter the selected checkpoint, pass/fail screen, or recommended next method.

## 5. Same-Row DEV Calibration Audit

Perform a post-hoc, no-update DEV-only comparison between:

A. the frozen sparse F2 checkpoint listed above;
B. the selected TASK-005D student checkpoint.

Use the exact same 933 DEV rows and observed masks.

Use existing:

    fusion.loss.ramps_r1_teacher.binary_brier
    fusion.loss.ramps_r1_teacher.binary_ece

with `bins=15`.

For each model and each disease, compute on observed DEV rows only:

- Brier;
- ECE;
- number of observed rows.

Also record:

- student minus F2 Brier delta;
- student minus F2 ECE delta.

These are diagnostics only. Do not recalibrate either model and do not use calibration results to change the selected epoch or rerun training.

## 6. Required Evidence

Record in `docs/PROGRESS_EN.md`:

- branch and commit base;
- exact config diff/equivalence to the warm-up contract;
- registry/build/firewall result;
- trainable parameter count equality to sparse F2;
- cache SHA/counts;
- exact production command;
- run directory;
- MLflow experiment/run ID/status/artifact URI;
- epochs completed and early-stop status;
- selected epoch/checkpoint based only on DEV;
- complete epoch table with:
  - epoch;
  - `train/pseudo_scale`;
  - train loss;
  - DEV depression Score;
  - DEV Parkinson Score;
  - DEV Mean_Score;
  - TEST_NONE/SOFT/HARD Mean_Score as monitoring-only columns;
- selected DEV UAR/MF1/Score per disease and Mean_Score;
- exact DEV deltas versus sparse F2;
- comparison to frozen audio DEV reference as context only;
- predeclared screen pass/fail;
- DEV-only Brier/ECE table for F2 and student;
- confirmation Test did not affect selection;
- confirmation no second training run occurred;
- confirmation src/audio/src/video/fusion/common source files remained unchanged;
- no missing-label correctness/comorbidity claim.

## Acceptance Criteria

TASK-005D is successfully executed if:

- exactly one seed-42 production training invocation runs after a passing firewall;
- production config differs from the accepted warm-up config only by run_name;
- same F2 architecture/parameter count is used;
- frozen cache/warm-up/counts remain unchanged;
- full required four-stream epoch monitoring runs;
- selected epoch is determined only by DEV/Mean_Score;
- same-row DEV calibration audit is completed without parameter updates;
- no Test-driven tuning/selection occurs;
- no second training run occurs;
- no source file changes occur;
- `git diff --check` passes;
- `docs/PROGRESS_EN.md` records all evidence;
- branch is committed/pushed and main/master remains untouched by Codex.

Research outcome is separately classified by the frozen screen:

PASS only if:

- DEV Mean_Score > `0.774569`;
- depression Score >= `0.687035`;
- Parkinson Score >= `0.842104`.

A screen failure is still a successfully completed TASK-005D with a negative research result.

No follow-up experiment is authorized inside TASK-005D.

## Exact Verification Commands

Before training:

    python3 -m py_compile       src/common/callbacks/wsm_pseudo_scale_warmup_callback.py       src/fusion/data/wsm_ramps_semantic_datamodule.py       src/fusion/loss/ramps_observed_pseudo_loss.py       src/fusion/models/av_f2_task_aware_directed.py       src/chimera_plugin.py

    .venv/bin/chimera-ml validate-config       -c configs/wsm_mm_pd_dep_v1/fusion/05_ramps_r2_student_seed42.yaml

Run a pre-run inline firewall that implements all assertions in Section 2 and record the exact command/result in PROGRESS_EN.md.

Production command:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train       --config-path configs/wsm_mm_pd_dep_v1/fusion/05_ramps_r2_student_seed42.yaml

After training:

- perform the DEV-only same-row calibration audit from Section 5;
- `git diff --check`;
- `git diff origin/main -- src/audio`;
- `git diff origin/main -- src/video`;
- `git diff origin/main -- src/fusion`;
- `git diff origin/main -- src/common`;
- `git status --short`;
- `git diff --stat origin/main...HEAD`;
- `git log -4 --oneline --decorate`.

## Required Handoff

Respond in English using exactly:

1. Outcome
2. Changed files
3. Verification
4. Plan status
5. Blockers and risks
6. Next atomic step

Explicitly include:

- branch `codex/task-005d`;
- implementation/evidence commit SHA;
- pushed-to-origin status;
- main/master untouched;
- manager base commit preserved;
- exact production config path;
- confirmation exactly one production training invocation occurred;
- model parameter-count equality versus sparse F2;
- cache SHA/counts;
- exact warm-up schedule;
- run directory and MLflow metadata;
- epochs completed/early stopping;
- selected DEV epoch/checkpoint;
- selected DEV D/P/Mean metrics;
- deltas versus F2;
- contextual delta versus frozen audio;
- frozen screen pass/fail;
- DEV-only F2 vs student Brier/ECE;
- same-epoch Test monitoring values only after DEV selection was frozen;
- no Test-driven selection/tuning;
- no second run;
- no pseudo regeneration/reselection;
- no missing-label correctness/comorbidity claim;
- R3/R4 not started;
- general Text/Description remains deferred;
- all source code unchanged;
- Stage 5 remains active.

Stop after TASK-005D. Do not authorize or begin another experiment.
