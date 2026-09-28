# TASK-005I-R4-CONFIRM: True-Seed Confirmation of Frozen R4 RA-STCH Trial 012

## Authority and branch

This task follows MANAGER-DECISION-056.

Required branch:

    codex/task-005i-r4-confirm

Start from current `origin/main`, which includes manager decision commit:

    418330a8b11786d62af43f449281045c7bfdb894

Create exactly one new task branch from that `origin/main`.

## Goal

Confirm the exact Optuna-selected corrected R4 RA-STCH trial `18c6-012` on true seeds 43 and 44.

Do not search, tune, redesign, or rerun seed42.

Exactly two new production training invocations are authorized:

1. seed43 once;
2. seed44 once.

The seed42 selected run is existing frozen evidence.

## Required reading

Read in this exact order:

1. AGENTS.md
2. docs/README.md
3. docs/PROJECT_REQUIREMENTS.md
4. docs/PLAN.md
5. docs/PROGRESS_EN.md through MANAGER-DECISION-056
6. docs/NEXT_TASK_EN.md
7. configs/wsm_mm_pd_dep_v1/fusion/35_optuna_r4_ra_stch_base.yaml
8. configs/wsm_mm_pd_dep_v1/fusion/36_optuna_r4_ra_stch_sweep.yaml
9. src/fusion/models/av_r3_disease_query.py
10. src/fusion/loss/r4_ramps_balance_loss.py
11. src/common/callbacks/wsm_r4_balance_callback.py
12. src/common/callbacks/wsm_pseudo_scale_warmup_callback.py

Also inspect read-only the selected generated trial config:

    logs/wsm_mm_pd_dep_v1/optuna_r4_ra_stch_base_seed42_2026-09-28_01-54_wsm_av_r3_disease_query_model_r4-ra-stch-optuna-v1-18c6-012_0ce9c7a4/r4-ra-stch-optuna-v1-18c6-012.yaml

and selected seed42 checkpoint:

    logs/wsm_mm_pd_dep_v1/optuna_r4_ra_stch_base_seed42_2026-09-28_01-54_wsm_av_r3_disease_query_model_r4-ra-stch-optuna-v1-18c6-012_0ce9c7a4/checkpoints/epoch=11_dev_mean_score=0.8191.pt

Required seed42 checkpoint SHA256:

    104b79e409832839502b4d20a98c054f5efda633930e313ce74f34327e668b2a

## Frozen selected hyperparameters

The selected trial has exactly:

- model.hidden_dim: `160`
- model.gate_hidden_dim: `96`
- model.dropout: `0.10034630487649018`
- loss.aux_weight: `0.41215983081110485`
- loss.agreement_weight: `0.48126684525457264`
- loss.tau: `0.0487101634503887`
- loss.progress_temperature: `0.7232081822527684`
- loss.controller_ema: `0.7848422072252504`
- optimizer.lr: `2.1217831107180076e-05`
- optimizer.weight_decay: `0.0001855295766514827`

All other R4 semantics remain exactly frozen from the selected generated config, including:

- model `wsm_av_r3_disease_query_model`;
- data `wsm_ramps_semantic_datamodule`;
- loss `wsm_r4_ramps_balance_loss`;
- mode `ra_stch`;
- pseudo_scale initial `0.0`;
- progress references D/P `0.697035 / 0.852104`;
- grad_ema `0.9`;
- reliability_ema `0.9`;
- weight_min/max `0.2 / 0.8`;
- eps `1e-8`;
- pseudo warm-up: 3 observed-only epochs + 5 ramp epochs to 1.0;
- batch size 32;
- epochs 30;
- CUDA;
- mixed precision true;
- grad clip 0.5;
- checkpoint/early stop only on `dev/mean_score`, mode max;
- patience 6;
- min_delta 0.0005;
- required DEV/TEST_NONE/TEST_SOFT/TEST_HARD monitoring.

Frozen R4 pseudo cache:

    /media/maxim/Programs/Features/WSM/ramps_r2_semantic_v1/train_missing_targets.pt

Required SHA256:

    17cf5e67e8c244d81b7c21f0842988966a23d83d69e296f7c5a5181376d6b945

## Frozen comparator

Frozen audio DEV:

- D `0.7479183895`;
- P `0.8277353635`;
- Mean `0.7878268765`.

Existing selected R4 seed42 DEV:

- D `0.759025`;
- P `0.879259`;
- Mean `0.8191424538`.

## Allowed tracked files

Codex may add/modify only:

- configs/wsm_mm_pd_dep_v1/fusion/37_r4_ra_stch_optuna_selected_seed42.yaml
- configs/wsm_mm_pd_dep_v1/fusion/38_r4_ra_stch_optuna_confirm_seed43.yaml
- configs/wsm_mm_pd_dep_v1/fusion/39_r4_ra_stch_optuna_confirm_seed44.yaml
- docs/PROGRESS_EN.md

No other tracked file may change.

## Forbidden actions

- Do not modify source code.
- Do not modify src/audio or src/video.
- Do not modify model/loss/DataModule/callback/plugin code.
- Do not modify configs 31–36.
- Do not modify the selected hyperparameters.
- Do not change pseudo cache or pseudo semantics.
- Do not run Optuna or another sweep.
- Do not rerun seed42.
- Do not run seeds other than 43 and 44.
- Do not run multiple restarts for a seed after a metric-bearing run begins.
- Do not use Test metrics to choose, stop, rerun, reinterpret, or promote anything.
- Do not start Stage 6/7, Text/Description, or Final Test.
- Do not make significance/final-method/generalization claims.

## 1. Freeze tracked selected config

Create:

    configs/wsm_mm_pd_dep_v1/fusion/37_r4_ra_stch_optuna_selected_seed42.yaml

It MUST be an exact tracked copy of the existing generated selected trial config.

Do not edit its scientific/training semantics.

Record:

- source generated-config path;
- source generated-config SHA256;
- tracked config SHA256;
- byte-for-byte equality result.

If byte-for-byte equality cannot be achieved because the generated artifact contains only runtime path metadata that Chimera regenerates, STOP and report before training. Do not normalize silently.

## 2. Create true-seed confirmation configs

Create:

    configs/wsm_mm_pd_dep_v1/fusion/38_r4_ra_stch_optuna_confirm_seed43.yaml
    configs/wsm_mm_pd_dep_v1/fusion/39_r4_ra_stch_optuna_confirm_seed44.yaml

Config 38 must differ from config 37 ONLY in:

- top-level `seed: 43`;
- `experiment_info.params.run_name`.

Config 39 must differ from config 37 ONLY in:

- top-level `seed: 44`;
- `experiment_info.params.run_name`.

Use run names:

- `r4_ra_stch_optuna_trial012_confirm_seed43`
- `r4_ra_stch_optuna_trial012_confirm_seed44`

No other YAML value may differ.

## 3. Mandatory pre-run firewall

Before either production run:

1. verify selected seed42 checkpoint SHA exactly;
2. verify pseudo-cache SHA exactly;
3. verify config37 byte equality to selected generated config;
4. parse configs37/38/39 and prove:
   - 38 differs from 37 only by seed/run_name;
   - 39 differs from 37 only by seed/run_name;
   - all frozen hyperparameters exactly match MANAGER-DECISION-056;
5. validate configs38/39 with Chimera;
6. verify required instrumentation and DEV-only selector;
7. verify no source/config31–36 diff;
8. append the complete confirmation firewall to PROGRESS_EN.md;
9. commit and push one firewall commit before seed43 starts.

No production invocation may begin before the firewall commit exists on origin.

## 4. Exact production commands

Run seed43 exactly once:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train \
      --config-path configs/wsm_mm_pd_dep_v1/fusion/38_r4_ra_stch_optuna_confirm_seed43.yaml

Then run seed44 exactly once:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train \
      --config-path configs/wsm_mm_pd_dep_v1/fusion/39_r4_ra_stch_optuna_confirm_seed44.yaml

Maximum new production invocations:

    2

If an infrastructure failure occurs before any metric-bearing epoch for a seed, STOP and report it. Do not silently restart.

## 5. Required per-seed evidence

For existing seed42 and new seeds43/44 record:

- seed;
- run directory/name;
- selected epoch;
- selected checkpoint path;
- checkpoint SHA256;
- DEV depression UAR/MF1/Score;
- DEV Parkinson UAR/MF1/Score;
- DEV Mean_Score;
- delta D/P/Mean vs frozen audio;
- selected-epoch R4 task weights D/P;
- selected-epoch progress D/P;
- selected-epoch gradient norm EMA D/P;
- selected-epoch gradient cosine EMA;
- selected-epoch reliability EMA D/P;
- pseudo scale at selected epoch.

Test streams may be recorded only as monitoring in the raw run artifacts; do not copy Test numbers into the manager comparison table.

## 6. Robust confirmation gate

Use ONLY DEV.

For each seed 42, 43, and 44 individually require:

- Mean > `0.7878268765`;
- D >= `0.7379183895`;
- P >= `0.8177353635`.

Also require:

- arithmetic three-seed mean Mean > `0.7878268765`;
- arithmetic three-seed mean D >= `0.7379183895`;
- arithmetic three-seed mean P >= `0.8177353635`;
- selected checkpoint SHA256 values pairwise distinct.

If all conditions pass, report:

    ROBUST THREE-SEED CONFIRMATION PASS

Otherwise report:

    ROBUST THREE-SEED CONFIRMATION FAIL

Do not alter the gate after seeing seeds43/44.

## 7. Strict diagnostic

Separately report, diagnostic only, for each seed whether both hold:

- D >= `0.7479183895`;
- P >= `0.8277353635`.

Report how many of 3 seeds satisfy exact task-wise non-regression.

This strict diagnostic does not replace the robust gate.

## 8. Aggregate reporting

Compute for D, P, and Mean across seeds42/43/44:

- arithmetic mean;
- sample standard deviation.

Also report:

- min/max by task/Mean;
- count of seeds beating frozen audio Mean;
- count passing the relaxed safe gate;
- count passing strict D/P non-regression.

Descriptive only. No significance claim.

## 9. Scope verification

Run exactly:

    git diff --check
    git diff origin/main -- src/audio
    git diff origin/main -- src/video
    git diff origin/main -- src/fusion/models
    git diff origin/main -- src/fusion/loss
    git diff origin/main -- src/fusion/data
    git diff origin/main -- src/common
    git diff origin/main -- src/chimera_plugin.py
    git diff origin/main -- configs/wsm_mm_pd_dep_v1/fusion/31_optuna_candidate_b_base.yaml
    git diff origin/main -- configs/wsm_mm_pd_dep_v1/fusion/32_optuna_candidate_b_sweep.yaml
    git diff origin/main -- configs/wsm_mm_pd_dep_v1/fusion/33_optuna_r3b_base.yaml
    git diff origin/main -- configs/wsm_mm_pd_dep_v1/fusion/34_optuna_r3b_sweep.yaml
    git diff origin/main -- configs/wsm_mm_pd_dep_v1/fusion/35_optuna_r4_ra_stch_base.yaml
    git diff origin/main -- configs/wsm_mm_pd_dep_v1/fusion/36_optuna_r4_ra_stch_sweep.yaml
    git status --short
    git diff --stat origin/main...HEAD
    git log -12 --oneline --decorate

Expected source/config31–36 diffs: empty.

## Acceptance criteria

TASK-005I-R4-CONFIRM passes only if:

- branch exactly `codex/task-005i-r4-confirm`;
- starts from manager decision commit `418330a8b11786d62af43f449281045c7bfdb894`;
- config37 is exact selected generated config copy;
- configs38/39 differ only by seed/run_name;
- selected hyperparameters remain exact;
- firewall committed/pushed before production;
- exactly two new production runs occurred, seeds43 and 44 once each;
- no seed42 rerun;
- no Optuna/search/tuning;
- no source changes;
- no Test-driven decision;
- complete seed42/43/44 DEV and controller evidence recorded;
- checkpoint SHA values pairwise distinct;
- robust gate and strict diagnostic computed exactly as frozen;
- branch pushed;
- main/master untouched by Codex;
- no Stage6/7/Text/Final Test.

## Required handoff

Respond in English using exactly:

1. Outcome
2. Changed files
3. Verification
4. Plan status
5. Blockers and risks
6. Next atomic step

Explicitly include:

- branch `codex/task-005i-r4-confirm`;
- firewall commit SHA;
- final evidence commit SHA;
- pushed status;
- main/master untouched;
- selected seed42 generated config path/SHA;
- config37 SHA and byte-equality proof;
- config38/39 diff-only proof;
- seed42 checkpoint path/SHA;
- pseudo-cache SHA;
- exact frozen 10 selected hyperparameters;
- exact two production commands;
- seed42/43/44 run names;
- selected epochs;
- selected checkpoint paths/SHAs;
- DEV D/P UAR/MF1/Score and Mean for all three;
- deltas vs audio;
- controller diagnostics for all three;
- three-seed mean/std;
- robust gate PASS/FAIL;
- strict diagnostic count;
- no Test-driven decision;
- no seed42 rerun;
- no extra restart;
- no Optuna/post-hoc tuning;
- source/config31–36 diffs empty;
- Stage 5 active/reopened;
- Stage 6 paused;
- Stage 7 and Final Test locked.

For section 6 write only:

    Manager review of TASK-005I-R4-CONFIRM; do not start another task.

Stop after TASK-005I-R4-CONFIRM.
