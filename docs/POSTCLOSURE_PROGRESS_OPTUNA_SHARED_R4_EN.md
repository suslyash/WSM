# POST-CLOSURE DEV-ONLY EXTENSION — NOT PART OF THE FROZEN STAGE-7/FINAL-TEST EVIDENCE

Status: pre-production firewall frozen; production evidence pending.

## Authority and boundary

- Task: TASK-011A.
- Base: `eff19d9987e79d99f29b0f5c7fff257341fe350d`.
- Branch: `codex/task-011a-progress-optuna-shared-r4`.
- This is a new equal-budget DEV-only optimization comparison. It does not
  revise frozen Stage-7, Final-Test, paper, or closure evidence.
- No confirmation seeds, Final Test, or cross-family leaderboard claim is
  authorized by this task.

## Frozen configurations and studies

| Role | Config |
|---|---|
| Shared + Progress baseline | `configs/wsm_mm_pd_dep_v1/postclosure_progress_optuna/00_shared_progress_baseline_seed42.yaml` |
| Full R4 + Progress baseline | `configs/wsm_mm_pd_dep_v1/postclosure_progress_optuna/01_r4_progress_baseline_seed42.yaml` |
| Shared + Progress Optuna base | `configs/wsm_mm_pd_dep_v1/postclosure_progress_optuna/02_shared_progress_optuna_base.yaml` |
| Full R4 + Progress Optuna base | `configs/wsm_mm_pd_dep_v1/postclosure_progress_optuna/03_r4_progress_optuna_base.yaml` |
| Shared sweep | `configs/wsm_mm_pd_dep_v1/postclosure_progress_optuna/04_shared_progress_sweep.yaml` |
| Full R4 sweep | `configs/wsm_mm_pd_dep_v1/postclosure_progress_optuna/05_r4_progress_sweep.yaml` |

Studies use exactly 30 trials each, `dev/mean_score` in max mode, seed 42,
30-epoch maximum, and the same eight-variable search space. The only family
difference is `task_aware_fusion: false` versus `true`.

Canonical pseudo cache SHA256:
`17cf5e67e8c244d81b7c21f0842988966a23d83d69e296f7c5a5181376d6b945`.
Accepted pseudo counts are D/P `376/1801`; observed truth overrides pseudo
supervision.

## Firewall

- Firewall commit: pending commit.
- Config validation, dry-run trial counts, DataModule/model/loss/callback
  construction, equal baseline parameter count `295239`, cache identity,
  detached pseudo fields, two-logit shape, Progress linear objective,
  Progress/RA independence, parameter ceiling, and TRAIN-only forward/loss/
  backward smoke are required before production.
- No DEV or Test loader is iterated by the firewall.

## Production evidence

Production is not yet run. Required exact invocation count after the firewall:
2 untuned seed42 baselines, 30 Shared + Progress trials, and 30 Full R4 +
Progress trials. Confirmation status: `NOT RUN`.

Test quarantine status: inherited infrastructure may emit Test monitoring rows;
they must not be inspected, tabulated, ranked, or used. This extension is
DEV-only.

Frozen paper roles and Stage-7/Final-Test evidence remain unchanged.
