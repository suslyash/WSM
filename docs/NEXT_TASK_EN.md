# TASK-007A: Final Candidate / Config / Seed / Statistical Freeze

## Authority and branch

This task follows **MANAGER-DECISION-078**.

Required branch:

    codex/task-007a

Start from current manager-updated `origin/main`.

Stage 7 is active, but this task is **freeze/preflight only**.

No production training.
No inference/performance evaluation.
No Test metric inspection.
No model/source/cache change.

## Frozen Stage-7 compared-method set

Manager has selected exactly three methods using DEV-only pre-final evidence:

1. **Frozen temporal audio baseline**
2. **Full optimized R4 trial012**
3. **Equal-parameter shared-fusion trial012**

No fourth method may be added in TASK-007A.

### Why these three are frozen

Audio:
- mandatory frozen reference;
- matched-seed D/P/Mean `0.7303377965/0.8162961212/0.7733169588`;
- Mean std `0.0138512531`.

Full R4:
- optimized multimodal trial012 reference;
- D/P/Mean `0.7352563333/0.8385606667/0.7869088179`;
- Mean std `0.0294384420`;
- higher Mean than audio on existing seeds42/43/44.

Shared fusion:
- exact TASK-006I equal-parameter comparator;
- same `295239` trainable parameters as full;
- D/P/Mean `0.7407873333/0.8467100000/0.7935663333`;
- full-minus-shared Mean `-0.0066575154`;
- shared Mean exceeded full on existing 3/3 seeds;
- `TASK-AWARE FUSION CONTRIBUTION NOT SUPPORTED`.

Excluded from final five-seed method set:
- T1: standalone baseline only; Mean `0.7357653333`, Mean std `0.0646553057`, no additive A+V evidence;
- T2: blocked before implementation/training;
- no-semantic-depression: material D trade-off (full-minus-ablation D `+0.020809`);
- paired task-isolated: two separately trained models, not equal total deployment size;
- other Stage-6 ablations/controls remain claim evidence, not finalists.

Do not revisit this selection in TASK-007A.

## Frozen seeds

For every compared method:

    42, 43, 44, 45, 46

Seeds42-44 are already accepted evidence and MUST NOT be rerun.

TASK-007A creates only missing seed45/46 configs.

The next production task will run exactly six new jobs if manager accepts this freeze.

## Existing accepted seed42-44 identities

Audit and freeze from committed PROGRESS/history:

### Audio
Configs:
- seed42 `configs/wsm_mm_pd_dep_v1/audio/00_frozen_baseline.yaml`
- seed43 `configs/wsm_mm_pd_dep_v1/audio/01_frozen_baseline_seed43.yaml`
- seed44 `configs/wsm_mm_pd_dep_v1/audio/02_frozen_baseline_seed44.yaml`

DEV:
- seed42 D/P/Mean `0.7479183895/0.8277353635/0.7878268765`
- seed43 `0.708699/0.811771/0.760235`
- seed44 `0.734396/0.809382/0.771889`

Checkpoint SHA:
- seed42 `0873c7cb5e32d415cdd301058949f5dd140c16b874cc5230d027a4ee33e3daf2`
- seed43 `1a65f8dd605a6d942bed828f126e79a1f02930dd578176ba31f49c4dc10899dd`
- seed44 `7832d20a5431666365e2214423633f2be87bb27f37406d5a5dda8adda21e9852`

### Full R4 trial012
Configs:
- seed42 `configs/wsm_mm_pd_dep_v1/fusion/37_r4_ra_stch_optuna_selected_seed42.yaml`
- seed43 `configs/wsm_mm_pd_dep_v1/fusion/38_r4_ra_stch_optuna_confirm_seed43.yaml`
- seed44 `configs/wsm_mm_pd_dep_v1/fusion/39_r4_ra_stch_optuna_confirm_seed44.yaml`

DEV:
- seed42 `0.759025/0.879259/0.8191424538`
- seed43 `0.718527/0.804363/0.761445`
- seed44 `0.728217/0.832060/0.780139`

Checkpoint SHA:
- seed42 `104b79e409832839502b4d20a98c054f5efda633930e313ce74f34327e668b2a`
- seed43 `6af4a4ed21040aff4db2906b0adc8aa5fd27a4b72c375fbb67f68de04ab8446e`
- seed44 `b06fec6912b27cd19d811b72b2c3c08cd1b0a6c826b637f87a314e3976b88c17`

### Shared fusion
Configs:
- seed42 `configs/wsm_mm_pd_dep_v1/ablations/55_shared_fusion_seed42.yaml`
- seed43 `configs/wsm_mm_pd_dep_v1/ablations/56_shared_fusion_seed43.yaml`
- seed44 `configs/wsm_mm_pd_dep_v1/ablations/57_shared_fusion_seed44.yaml`

Retrieve from accepted TASK-006I PROGRESS:
- per-seed selected epoch;
- D/P/Mean;
- checkpoint path/SHA;
- MLflow/run identity.

Do not invent missing identity fields.

## Missing seed45/46 config files

Create exactly:

### Audio
- `configs/wsm_mm_pd_dep_v1/audio/03_final_seed45.yaml`
- `configs/wsm_mm_pd_dep_v1/audio/04_final_seed46.yaml`

Each must be semantically identical to `audio/00_frozen_baseline.yaml` except:
- seed;
- run_name.

Run names:
- `stage7_audio_final_seed45`
- `stage7_audio_final_seed46`

### Full R4
- `configs/wsm_mm_pd_dep_v1/fusion/40_r4_trial012_final_seed45.yaml`
- `configs/wsm_mm_pd_dep_v1/fusion/41_r4_trial012_final_seed46.yaml`

Each must be semantically identical to config37 except:
- seed;
- run_name.

Run names:
- `stage7_r4_trial012_final_seed45`
- `stage7_r4_trial012_final_seed46`

### Shared fusion
- `configs/wsm_mm_pd_dep_v1/fusion/42_shared_fusion_final_seed45.yaml`
- `configs/wsm_mm_pd_dep_v1/fusion/43_shared_fusion_final_seed46.yaml`

Each must be semantically identical to ablation55 except:
- seed;
- run_name.

Run names:
- `stage7_shared_fusion_final_seed45`
- `stage7_shared_fusion_final_seed46`

Moving the new shared-fusion final configs into the fusion directory does not authorize any semantic change.

## Final statistical plan — freeze now

Create `docs/STAGE7_FINAL_FREEZE_EN.md`.

Freeze:

### Per-method five-seed summaries
For D Score, P Score, and Mean_Score over seeds42-46 report:
- arithmetic mean;
- sample standard deviation (ddof=1);
- min/max/range;
- two-sided 95% Student-t confidence interval for the mean:

    mean ± t(0.975, df=4) * s / sqrt(5)

Use the frozen multiplier:

    2.7764451051977987

No alternative CI method after seeing seed45/46 results.

### Paired comparisons
For same-seed deltas, report D/P/Mean:
- full R4 minus audio;
- shared fusion minus audio;
- shared fusion minus full R4.

For each delta vector over seeds42-46 report:
- five seed deltas;
- mean delta;
- sample std;
- wins/ties/losses by Mean;
- two-sided 95% Student-t CI using the same df=4 multiplier.

Do not predeclare a significance winner. CI interpretation is descriptive.

### Calibration
For each final selected checkpoint, DEV-only Brier and ECE-15 remain required.
Report five-seed mean/std/95% CI per method/task.

No threshold fitting.

## Final Test firewall — freeze now

Final Test is **NOT authorized in TASK-007A**.

Training configs continue to expose mandatory epoch-level TEST_NONE/SOFT/HARD monitoring because PROJECT_REQUIREMENTS requires it. However, for new seeds45/46:

- Codex must not inspect, transcribe, compare, or use those Test values during the missing-seed training task;
- only DEV may determine checkpoints;
- all five selected checkpoint identities for all three methods must be frozen before any final Test reporting;
- the separate final-Test task will perform/read one final TEST_NONE/SOFT/HARD evaluation/report per frozen checkpoint;
- no model/config/threshold change is allowed after that final-Test task begins.

Existing historical Test monitoring for seeds42-44 is also not candidate-selection evidence.

## Next production order — freeze now

If TASK-007A passes, the next task will run exactly six jobs in this order:

1. audio seed45;
2. full R4 seed45;
3. shared fusion seed45;
4. audio seed46;
5. full R4 seed46;
6. shared fusion seed46.

No retry for metric improvement.
No extra seed.
No config changes after TASK-007A freeze.

## Mandatory preflight

Without training:

1. verify current main/base and clean task branch;
2. create the six configs and prove semantic equivalence to their references except seed/run_name;
3. validate all six configs;
4. instantiate DataModule/model/loss/callbacks for all six;
5. verify trainable parameter counts:
   - audio: retrieve/freeze current count from config instantiation;
   - full R4: `295239`;
   - shared fusion: `295239`;
6. verify full/shared pseudo cache SHA and canonical feature-cache identities unchanged;
7. verify audio frozen feature contract unchanged;
8. verify existing seed42-44 config/checkpoint/run identities from accepted evidence; do not execute them;
9. one shape-only TRAIN/DEV batch per method family; no optimizer step and no metric calculation;
10. verify `dev/mean_score` is checkpoint/early-stopping selector in every new config;
11. verify Test protocols are present but no Test loader iteration/metric inspection occurs;
12. write the final freeze dossier;
13. `git diff --check`;
14. forbidden source diffs empty;
15. append preflight evidence to PROGRESS;
16. commit and push.

## Allowed tracked files

Only:

- six new config files listed above;
- `docs/STAGE7_FINAL_FREEZE_EN.md`;
- `docs/PROGRESS_EN.md`.

No source changes.
No existing config changes.
No scripts.
No caches.
No dependencies.

## No production run

TASK-007A MUST NOT run any production training command.

It MUST NOT inspect Test metrics.

## Required PROGRESS evidence

Record:

- branch/final SHA;
- frozen finalist set;
- frozen seeds42-46;
- exact existing seed42-44 checkpoint/config/run identities;
- six new config paths and equivalence proof;
- parameter counts;
- frozen statistical plan and CI multiplier;
- frozen paired-comparison plan;
- final-Test firewall;
- exact next six-run order;
- no training/Test analysis;
- Stage 7 active;
- Final Test locked.

## Final scope checks

Run:

    git diff --check
    git diff origin/main -- src
    git diff origin/main -- scripts
    git diff origin/main -- pyproject.toml
    git status --short
    git diff --stat origin/main...HEAD
    git log -10 --oneline --decorate

Only the eight authorized files may differ.

## Acceptance criteria

TASK-007A passes only if:

- branch exactly `codex/task-007a`;
- finalist set exactly audio/full-R4/shared-fusion;
- seeds exactly 42-46;
- no rerun of 42-44;
- six new configs differ only seed/run_name from references;
- all configs instantiate and selectors remain DEV-only;
- existing 42-44 identities are traceable;
- statistical/paired/CI plan frozen before new runs;
- Final Test firewall frozen;
- no production training;
- no Test metric inspection;
- no source/cache/dependency change;
- branch pushed;
- main/master untouched.

Passing TASK-007A authorizes no training automatically. Manager must review and then assign the six missing runs.

## Required handoff

Respond in English using exactly:

1. Outcome
2. Changed files
3. Verification
4. Plan status
5. Blockers and risks
6. Next atomic step

Include branch, final SHA, pushed status, main/master untouched, finalist set, seed set, six new config paths, semantic-equivalence result, existing seed42-44 identity audit, parameter counts, statistical/paired/CI freeze, final-Test firewall, no training/Test inspection, Stage 7 active, Final Test locked.

For section 6 write only:

    Manager review of TASK-007A; do not start another task.

Stop.
