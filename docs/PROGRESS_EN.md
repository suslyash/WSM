# WSM Project Progress

## 1. Current State

Plan initialized: 2026-09-23.

Current stage: **Stage 6 — ACTIVE (ablations and claims audit)**.

Stage 5: **CLOSED TO OPTIMIZATION — R4 RETAINED AS LEADING MATCHED-SEED CANDIDATE**.

Stage 7: **LOCKED**.

Final Test authorized: **no**.

Current atomic task: **TASK-006A-BUNDLE — Stage-6 shuffled pseudo-target negative control on the Equal composition**.

Expected Codex branch: `codex/task-006a`.

`TASK-006-PRE-AUDIO-CONFIRM` is complete and merged. The previously paused `TASK-006A-BUNDLE` is now **RESUMED** as the sole active task.

## 2. Default Context Policy

This file is the compact authoritative status/roadmap entrypoint.

Default agents should read:

1. `AGENTS.md`;
2. `docs/README.md`;
3. `docs/PROJECT_REQUIREMENTS.md`;
4. this file;
5. `docs/NEXT_TASK_EN.md`;
6. only the active stage plan and source/config files explicitly named by the task.

Do **not** load closed-stage plans or the historical progress archive unless the active task or manager explicitly requires them.

Optional historical evidence:

- [archive/PROGRESS_EN_THROUGH_STAGE5_2026-09-28.md](archive/PROGRESS_EN_THROUGH_STAGE5_2026-09-28.md) — complete former progress ledger through Stage-5 closure;
- [plan/README.md](plan/README.md) — stage-plan index;
- [PLAN.md](PLAN.md) — cross-stage formulation, architecture, experiment matrix, promotion rules, risks, and orchestration details.

## 3. High-Level Project Roadmap

| Stage | Status | High-level goal | Detailed plan |
|---|---|---|---|
| 0. Reproducible base | COMPLETE | Freeze and validate the historical audio baseline and instrumentation contract. | [plan/STAGE_0.md](plan/STAGE_0.md) |
| 1. Manifest / partial labels | COMPLETE | Canonical sparse-label manifest, split contract, masking, and evaluation separation. | [plan/STAGE_1.md](plan/STAGE_1.md) |
| 2. Video | COMPLETE | Compare the bounded video families and establish the accepted V2 video reference. | [plan/STAGE_2.md](plan/STAGE_2.md) |
| 3. Text / description | DEFERRED | Bounded text/description study after the core Stage-6 research cycle. | [plan/STAGE_3.md](plan/STAGE_3.md) |
| 4. Fusion baselines | COMPLETE | Honest A+V baselines and strong-audio fusion search; no safe robust winner. | [plan/STAGE_4.md](plan/STAGE_4.md) |
| 5. RAMPS | CLOSED TO OPTIMIZATION | Optimization/search is complete. Matched-seed audio confirmation superseded the prior negative robustness interpretation; R4 is retained as the leading Stage-6 candidate, not yet a final promoted method. | [plan/STAGE_5.md](plan/STAGE_5.md) |
| 6. Ablations / claims audit | ACTIVE | Determine which component/causal claims are supported; retain negative controls. | [plan/STAGE_6.md](plan/STAGE_6.md) |
| 7. Final evaluation | LOCKED | Final multi-seed evaluation only after configuration/claim freeze and manager authorization. | [plan/STAGE_7.md](plan/STAGE_7.md) |

## 4. Frozen Scientific References

### Matched-seed audio and R4 references

Frozen temporal audio is now confirmed on seeds42/43/44 without any method change:

| Seed | Audio D | Audio P | Audio Mean |
|---:|---:|---:|---:|
| 42 | 0.7479183895 | 0.8277353635 | 0.7878268765 |
| 43 | 0.708699 | 0.811771 | 0.760235 |
| 44 | 0.734396 | 0.809382 | 0.771889 |

Audio three-seed DEV aggregate:

- D mean/std: `0.7303377965 / 0.0199221457`;
- P mean/std: `0.8162961212 / 0.0099784282`;
- Mean mean/std: `0.7733169588 / 0.0138512531`;
- Mean range: `0.0275918765`.

Frozen R4 RA-STCH trial `18c6-012` on the same seeds:

- D mean/std: `0.7352563333 / 0.0211467766`;
- P mean/std: `0.8385606667 / 0.0378688091`;
- Mean mean/std: `0.7869088179 / 0.0294384420`;
- Mean range: `0.0576974538`.

Matched-seed R4 minus audio DEV Mean:

- seed42: `+0.0313155773`;
- seed43: `+0.0012100000`;
- seed44: `+0.0082500000`;
- R4 Mean > audio Mean on `3/3` seeds;
- R4 three-seed Mean advantage: `+0.0135918591`.

Interpretation: audio is **more seed-stable** by Mean std/range, but R4 has the higher matched-seed DEV Mean on every tested seed and higher three-seed mean. Therefore the former phrase “strongest robust audio reference” is retired. This does not by itself establish final promotion or significance.

`src/audio` remains frozen.

### Accepted Stage-2 video reference

- model: V2 prototype-aware video;
- DEV depression Score: `0.6201013364`;
- DEV Parkinson Score: `0.7930427585`;
- DEV Mean_Score: `0.7065720475`.

### Frozen semantic pseudo cache

- path: `/media/maxim/Programs/Features/WSM/ramps_r2_semantic_v1/train_missing_targets.pt`;
- SHA256: `17cf5e67e8c244d81b7c21f0842988966a23d83d69e296f7c5a5181376d6b945`;
- TRAIN rows: `6325`;
- accepted missing depression: `376/2665`, all positive;
- accepted missing Parkinson: `1801/3660`, positive/negative `212/1589`;
- observed truth always overrides pseudo supervision;
- no missing-label correctness or comorbidity claim is authorized.

## 5. Stage-5 Closure — Revised Scientific Interpretation

Stage-5 optimization/search remains **closed**. No further tuning against seeds43/44 is authorized.

Historical **MANAGER-DECISION-057** closed Stage 5 negative because R4 three-seed Mean `0.7869088179` was compared against the seed42-only audio value `0.7878268765`. The owner-authorized matched-seed audio audit showed that this was not a valid robustness comparator.

With audio seeds43/44 added under the exact frozen method:

- R4 beats matched audio DEV Mean on `3/3` seeds;
- R4 three-seed Mean is `0.7869088179` vs audio `0.7733169588`;
- R4 mean advantage is `+0.0135918591`;
- audio is more stable by Mean sample std (`0.0138512531` vs R4 `0.0294384420`) and Mean range (`0.0275918765` vs R4 `0.0576974538`);
- task-specific R4-audio regressions occur only on P at seed43 (`-0.007408`) and D at seed44 (`-0.006179`), both smaller than the PLAN's initial 1.0 percentage-point unacceptable-drop recommendation.

Manager interpretation: the **negative robustness verdict is superseded**, but Stage-5 optimization stays closed. R4 RA-STCH trial 012 is retained as the **leading matched-seed multimodal candidate for Stage-6 audit**, not a final promoted method. Promotion still requires mechanism/diagnostic support, and final methods require at least five seeds under PROJECT_REQUIREMENTS.

Candidate B and corrected R3-B remain non-substitutes under their frozen Stage-5 evidence. Seeds43/44 remain confirmation evidence, not tuning targets.

## 6. Stage-6 General Plan

Stage 6 is a **claims/ablation audit, not a tuning stage**.

The detailed contract is in [plan/STAGE_6.md](plan/STAGE_6.md). At high level it covers:

1. sparse MTL / task-aware fusion contribution;
2. direct pseudo-supervision contribution;
3. uncertainty/reliability contribution;
4. semantic-evidence contribution;
5. balancing/RA-STCH contribution;
6. modality removals;
7. shuffled/mismatched pseudo-target negative control;
8. corpus probe;
9. equal-parameter control if model size becomes a confound;
10. calibration, gradient, negative-transfer, coverage, and gate diagnostics.

Each scientific claim must map to an ablation or negative control. Negative results are retained. Test protocols remain monitoring-only and cannot drive decisions.

## 7. Active Task — TASK-006A-BUNDLE

The previously paused Stage-6 shuffled/mismatched pseudo-target negative control is resumed.

Purpose: test the **sample-specific pseudo alignment** claim without changing coverage, class balance, pseudo-value/reliability distributions, architecture, loss, optimizer, or warm-up.

Frozen matched reference remains the three-seed **Equal** composition:

| Seed | D Score | P Score | Mean |
|---:|---:|---:|---:|
| 42 | 0.712498 | 0.843342 | 0.777920 |
| 43 | 0.703299 | 0.856043 | 0.779671 |
| 44 | 0.707977 | 0.851878 | 0.779927 |

Exactly three shuffled-control production runs are authorized: seeds42/43/44. This is claims audit, not tuning. Full executable requirements are authoritative in [NEXT_TASK_EN.md](NEXT_TASK_EN.md).

## 8. Non-Negotiable Current Boundaries

- Do not tune or modify `src/audio`.
- Do not reopen Stage-5 optimization or tune against confirmation seeds.
- Seeds43/44 from R4 confirmation are not tuning targets.
- Unknown labels remain masked, never converted to negative.
- Observed ground truth overrides pseudo labels.
- Test metrics cannot drive epoch, architecture, hyperparameter, threshold, ablation, or follow-up selection.
- General Text/Description remains deferred.
- Stage 7 and Final Test remain locked.
- Do not claim verified recovery of genuinely missing disease labels or comorbidity without dual-annotated evidence.

## 9. Recent Authoritative Manager Decisions

Only recent state-changing decisions are repeated here; older decisions are in the archive.

### MANAGER-DECISION-056 — Triple Optuna accepted; R4 trial 012 selected for confirmation

- Candidate B objective winner: Mean `0.8165907903`, but 0/20 relaxed-safe.
- corrected R3-B objective winner: Mean `0.8159243728`, but 0/20 relaxed-safe.
- corrected R4 RA-STCH objective winner `18c6-012`: D/P/Mean `0.759025/0.879259/0.8191424538`;
- R4 had 4/20 relaxed-safe trials; trial 012 was the only objective winner with exact audio D/P non-regression;
- this was DEV-only family selection, not final promotion.

### MANAGER-DECISION-057 — R4 confirmation failure accepted; Stage 5 closed negative; Stage 6 activated

- true seed43 and seed44 confirmations failed the frozen robust gate;
- three-seed Mean `0.7869088179` remained below frozen audio Mean `0.7878268765`;
- Stage 5 closed negative;
- Stage 6 activated;
- no tuning against confirmation seeds is allowed.

### MANAGER ASSIGNMENT — TASK-006A-BUNDLE

- shuffled/mismatched pseudo-target negative control on matched Equal was assigned, then paused before execution for the owner-requested audio matched-seed check;
- after TASK-006-PRE-AUDIO-CONFIRM, it is RESUMED unchanged as the sole active Stage-6 task.

### MANAGER-OVERRIDE-058 — Confirm frozen audio on seeds43/44 before Stage-6 negative controls

- owner requested a matched-seed check because the prior "robust audio reference" interpretation relied on a seed42 audio result while R4 had true seeds42/43/44;
- authorize exactly two frozen-audio production runs: seeds43 and44;
- no audio method or hyperparameter tuning is allowed;
- only seed and run_name may differ from the canonical frozen seed42 config;
- compare audio and frozen R4 on matched seeds42/43/44 using DEV metrics only;
- TASK-006A-BUNDLE is paused until manager review of this evidence.

### MANAGER-DECISION-059 — Accept matched-seed audio confirmation; supersede negative robustness interpretation; resume TASK-006A

- TASK-006-PRE-AUDIO-CONFIRM passed manager review and was merged via PR #49;
- firewall commit `71a5d54375928387392e6dadfda182842515748b` correctly precedes final evidence commit `daec15685fb618161bf68f6b32e5df4b566b1875`;
- configs43/44 are semantically identical to the frozen seed42 config except seed/run_name; no `src` changes occurred;
- exact two production runs occurred, seed43 then seed44, with DEV-only checkpoint selection and no tuning/rerun;
- the Codex handoff statement that audio is “less stable than R4” is rejected: audio has lower Mean std/range;
- however, R4 exceeds matched audio DEV Mean on 3/3 seeds and by `+0.0135918591` in three-seed mean;
- therefore the prior Stage-5 negative robustness interpretation against seed42-only audio is superseded;
- Stage-5 optimization remains closed; R4 trial 012 is retained as the leading matched-seed candidate pending Stage-6 claim/mechanism audit;
- TASK-006A-BUNDLE is resumed as the only next atomic task.

## 10. Historical Evidence

The full former 5,000+ line ledger is intentionally no longer part of default context.

Use it only when a historical implementation detail, exact old command, old PR/commit, or prior corrective review is material:

[archive/PROGRESS_EN_THROUGH_STAGE5_2026-09-28.md](archive/PROGRESS_EN_THROUGH_STAGE5_2026-09-28.md)


### TASK-006-PRE-AUDIO-CONFIRM - pre-run firewall complete

Status: partial; the frozen-audio seed43/44 firewall passed and was committed before production.

Created configs:

- `configs/wsm_mm_pd_dep_v1/audio/01_frozen_baseline_seed43.yaml`
- `configs/wsm_mm_pd_dep_v1/audio/02_frozen_baseline_seed44.yaml`

Firewall evidence:

- Both configs validate with `chimera-ml validate-config`.
- Config 01 differs from config00 only by `seed: 43` and `run_name: frozen_audio_wavlm_l9_pool4_seed43`.
- Config 02 differs from config00 only by `seed: 44` and `run_name: frozen_audio_wavlm_l9_pool4_seed44`.
- The registered audio DataModule, model, and loss instantiated successfully for both configs.
- TRAIN-only forward/loss/backward smoke passed for both configs with finite loss and nonzero finite trainable gradients: config01 batch shape `(8, 1245, 768)`, loss `0.3348849714`; config02 batch shape `(8, 1250, 768)`, loss `0.3805317283`; each reported 113 nonzero-gradient tensors.
- No optimizer step was executed. No DEV or Test loader was iterated.
- Frozen audio checkpoint SHA256 remains `0873c7cb5e32d415cdd301058949f5dd140c16b874cc5230d027a4ee33e3daf2`.
- `git diff --check` passed and `git diff origin/main -- src` is empty.
- No source file or existing config was changed.

No production training has started. The firewall commit is pushed before the exactly two authorized invocations, in order: seed43 once, then seed44 once. TASK-006A-BUNDLE remains paused/unstarted; Stage 5 remains historically closed negative.

### TASK-006-PRE-AUDIO-CONFIRM - final frozen-audio seed confirmation evidence

Status: complete. Exactly two new production invocations occurred, in order: seed43 once, then seed44 once. No seed42 rerun, sweep, tuning, restart, or shuffled-pseudo work occurred. Stage 5 remains historically closed negative; TASK-006A-BUNDLE remains paused/unstarted.

Run and MLflow identities:

- Seed43 run: `frozen_audio_wavlm_l9_pool4_seed43_2026-09-28_13-12_audio_mamba_segment_model_08192c1b`; MLflow run ID `4119e9a654ec481ca42606f7f207b969`; status `FINISHED`; artifact URI `/media/maxim/Programs/Projects/WSM/mlruns/5/4119e9a654ec481ca42606f7f207b969/artifacts`; resolved config `configs/wsm_mm_pd_dep_v1/audio/01_frozen_baseline_seed43.yaml`; firewall commit `71a5d54`.
- Seed44 run: `frozen_audio_wavlm_l9_pool4_seed44_2026-09-28_13-21_audio_mamba_segment_model_8720f109`; MLflow run ID `ab9a3a73a0e94a2c889a69b25038d141`; status `FINISHED`; artifact URI `/media/maxim/Programs/Projects/WSM/mlruns/5/ab9a3a73a0e94a2c889a69b25038d141/artifacts`; resolved config `configs/wsm_mm_pd_dep_v1/audio/02_frozen_baseline_seed44.yaml`; firewall commit `71a5d54`.
- Both runs stopped by the frozen DEV-only early-stopping monitor after 11 and 10 epochs respectively; selected checkpoints were frozen by maximum `dev/mean_score` before reading same-epoch Test monitoring.

Audio DEV table:

| Seed | Selected epoch | DEV D Score | DEV P Score | DEV Mean | Selected checkpoint SHA256 |
|---:|---:|---:|---:|---:|---|
| 42 | 4 | 0.7479183895 | 0.8277353635 | 0.7878268765 | `0873c7cb5e32d415cdd301058949f5dd140c16b874cc5230d027a4ee33e3daf2` |
| 43 | 5 | 0.708699 | 0.811771 | 0.760235 | `1a65f8dd605a6d942bed828f126e79a1f02930dd578176ba31f49c4dc10899dd` |
| 44 | 4 | 0.734396 | 0.809382 | 0.771889 | `7832d20a5431666365e2214423633f2be87bb27f37406d5a5dda8adda21e9852` |

Complete new-seed DEV metrics:

- Seed43 epoch 5: D UAR/MF1/Score = 0.709057/0.708341/0.708699; P UAR/MF1/Score = 0.811042/0.812500/0.811771; Mean = 0.760235.
- Seed44 epoch 4: D UAR/MF1/Score = 0.735201/0.733592/0.734396; P UAR/MF1/Score = 0.797447/0.821317/0.809382; Mean = 0.771889.

Same-epoch Test monitoring only (not used for selection or interpretation):

- Seed43 epoch 5: TEST_NONE Mean = 0.791690; TEST_SOFT Mean = 0.802602; TEST_HARD Mean = 0.818189.
- Seed44 epoch 4: TEST_NONE Mean = 0.811919; TEST_SOFT Mean = 0.820354; TEST_HARD Mean = 0.831768.

Audio three-seed DEV aggregate:

- D Score mean/std = 0.7303377965 / 0.0199221457; min/max/range = 0.708699 / 0.7479183895 / 0.0392193895.
- P Score mean/std = 0.8162961212 / 0.0099784282; min/max/range = 0.809382 / 0.8277353635 / 0.0183533635.
- Mean mean/std = 0.7733169588 / 0.0138512531; min/max/range = 0.760235 / 0.7878268765 / 0.0275918765.

Matched R4 minus audio DEV deltas:

| Seed | D delta | P delta | Mean delta | R4 Mean > audio Mean |
|---:|---:|---:|---:|---|
| 42 | +0.0111066105 | +0.0515236365 | +0.0313155773 | yes |
| 43 | +0.0098280000 | -0.0074080000 | +0.0012100000 | yes |
| 44 | -0.0061790000 | +0.0226780000 | +0.0082500000 | yes |

R4-vs-audio three-seed variability comparison:

| Metric | R4 mean/std | Audio mean/std | R4 min/max/range | Audio min/max/range |
|---|---:|---:|---|---|
| D | 0.7352563333 / 0.0211467766 | 0.7303377965 / 0.0199221457 | 0.718527 / 0.759025 / 0.0404980 | 0.708699 / 0.7479183895 / 0.0392194 |
| P | 0.8385606667 / 0.0378688091 | 0.8162961212 / 0.0099784282 | 0.804363 / 0.879259 / 0.0748960 | 0.809382 / 0.8277353635 / 0.0183534 |
| Mean | 0.7869088179 / 0.0294384420 | 0.7733169588 / 0.0138512531 | 0.761445 / 0.8191424538 / 0.0576975 | 0.760235 / 0.7878268765 / 0.0275919 |

Interpretation boundary: R4 exceeds matched audio DEV Mean on 3/3 seeds, but this is a robustness comparator audit only. No significance test, promotion, demotion, Stage-6 ablation choice, or final-method claim is made. Test remains monitoring-only.

Final integrity: production invocation count exactly 2; `src/audio` and all `src/*` unchanged; no Test-driven decision; no post-hoc tuning/rerun; no Stage 6 shuffled-pseudo work, Stage 7, Text/Description, or Final Test work started. Stage 6 is active at the plan level but this task does not start it.


### TASK-006A-BUNDLE   pre-training negative-control firewall

Status: firewall complete; production has not started. The exact three shuffled-control configs are frozen before the first production invocation.

Changed paths before firewall commit:

- `scripts/common/build_ramps_shuffled_pseudo_negative_control.py`;
- `configs/wsm_mm_pd_dep_v1/ablations/31_shuffled_pseudo_equal_seed42.yaml`;
- `configs/wsm_mm_pd_dep_v1/ablations/32_shuffled_pseudo_equal_seed43.yaml`;
- `configs/wsm_mm_pd_dep_v1/ablations/33_shuffled_pseudo_equal_seed44.yaml`;
- `docs/PROGRESS_EN.md`.

Cache build command:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python scripts/common/build_ramps_shuffled_pseudo_negative_control.py --source /media/maxim/Programs/Features/WSM/ramps_r2_semantic_v1/train_missing_targets.pt --output /media/maxim/Programs/Features/WSM/ramps_r2_semantic_v1_negative_control/shuffled_missing_targets_seed6001.pt --shuffle-seed 6001
```

Builder result: `RAMPS_SHUFFLED_NEGATIVE_CONTROL_PASS`. Source SHA256 is `17cf5e67e8c244d81b7c21f0842988966a23d83d69e296f7c5a5181376d6b945`; derived cache SHA256 is `678571febcf6ce85f2bf7aedb4770b255a138a0cdaff5ea3f04a92606afbd814`. Rows are `6325`; missing D/P are `2665/3660`; accepted D/P are `376/1801`; accepted class counts are D positive/negative `376/0` and P positive/negative `212/1589`. Acceptance-assignment Hamming differences are D/P `646/1774`. Within-task permutation fixed points are D/P `0/0`; permutation SHA256 values are D `1a3853bb4ad61c627d320be9f1c90fa7f6493a3ea922613f08be458c0e963f6b` and P `82df2450281b6cb1bbc5ecdfcc9c733b707dd53553c5ecdf0979032b836d6af1`. Source segment IDs, observed masks, observed targets, and the complete missing-row tuple multisets are unchanged; source cache was not modified.

Config validation result: all three configs are valid under `chimera-ml validate-config`. Each is a matched Equal copy differing only in pseudo-cache path and run name; seeds remain `42/43/44`, and the model, loss, optimizer, warm-up, callbacks, logging, and training parameters are unchanged.

TRAIN-only registry/data/model/loss smoke passed with the derived cache and no optimizer step. It selected one TRAIN batch containing accepted shuffled pseudo rows and ran forward/loss/backward at pseudo_scale=1.0. Result: `FROZEN_DATAMODULE_MODEL_LOSS_FIREWALL_PASS`; TRAIN rows `6325`; accepted pseudo rows in smoke batch `17`; trainable parameters `403079`; finite loss `0.7944685221`; pseudo-target and reliability gradients are `None`; model gradients are finite. Plugin registration imported without a project-module warning. No DEV/Test loader was iterated and no optimizer step was executed.

Required frozen hashes checked: audio checkpoint `0873c7cb5e32d415cdd301058949f5dd140c16b874cc5230d027a4ee33e3daf2`; source pseudo cache `17cf5e67e8c244d81b7c21f0842988966a23d83d69e296f7c5a5181376d6b945`. `git diff --check` passed and `git diff -- src/audio` is empty.

The firewall commit must precede exactly these production commands, in order: config31 seed42, config32 seed43, config33 seed44. No sweep, tuning, Test-driven selection, or other Stage-6 work is authorized.
