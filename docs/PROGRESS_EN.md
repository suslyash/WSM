# WSM Project Progress

## 1. Current State

Plan initialized: 2026-09-23.

Current stage: **Stage 6 — ACTIVE (ablations and claims audit)**.

Stage 5: **CLOSED TO OPTIMIZATION — R4 RETAINED AS LEADING MATCHED-SEED CANDIDATE**.

Stage 7: **LOCKED**.

Final Test authorized: **no**.

Current atomic task: **TASK-006B — frozen R4 corpus-identity probe on TRAIN→DEV**.

Expected Codex branch: `codex/task-006b`.

`TASK-006A` is complete and merged. Stage-6 shuffled/mismatched pseudo-target negative-control item 8 is CLOSED with narrow support for sample-specific pseudo alignment.

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

## 7. Active Task — TASK-006B

Purpose: execute the PLAN-required **corpus probe** on the frozen R4 RA-STCH candidate without retraining the main model.

Use the three frozen R4 checkpoints:

| Seed | Config | Selected epoch | Checkpoint SHA256 | DEV Mean |
|---:|---|---:|---|---:|
| 42 | `fusion/37_r4_ra_stch_optuna_selected_seed42.yaml` | 11 | `104b79e409832839502b4d20a98c054f5efda633930e313ce74f34327e668b2a` | 0.8191424538 |
| 43 | `fusion/38_r4_ra_stch_optuna_confirm_seed43.yaml` | 10 | `6af4a4ed21040aff4db2906b0adc8aa5fd27a4b72c375fbb67f68de04ab8446e` | 0.7614450000 |
| 44 | `fusion/39_r4_ra_stch_optuna_confirm_seed44.yaml` | 11 | `b06fec6912b27cd19d811b72b2c3c08cd1b0a6c826b637f87a314e3976b88c17` | 0.7801390000 |

The probe fits deterministic diagnostic linear classifiers on TRAIN and evaluates **DEV only** for corpus identity (`depression` vs `parkinson`) from frozen internal representations. It also records gate distributions by corpus and a shuffled-TRAIN-label sanity control.

This audit may establish that corpus identity is linearly decodable, and whether task-conditioned fusion amplifies that decodability relative to the equal-dimensional projected A+V representation. It **cannot** by itself establish that corpus identity causes the disease scores, replaces disease signal, or invalidates R4.

No main-model training, no Test loader iteration, no hyperparameter tuning, and no Stage-5 reopening are authorized. Full executable requirements are authoritative in [NEXT_TASK_EN.md](NEXT_TASK_EN.md).

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

### MANAGER-DECISION-060 — Reject TASK-006A implementation; exact derangement FIX1 required

- branch `codex/task-006a` starts from the correct manager base and changes only the authorized five paths;
- firewall commit `70e5eec9d3710ac7bd0bd05cae6848f32107d190` correctly precedes final commit `c34bb6a20a42c59c7d3aa0c0413ab1c70ba7f7dc`;
- config equivalence to matched Equal passes and no `src` file changed;
- the reported shuffled results show a large matched-over-shuffled DEV Mean gap, but the result is **not accepted** because the derived cache did not follow the frozen permutation mapping literally;
- rejected builder used `source_rows = roll(random_order, 1)` but assigned `derived[field][missing] = source[field][source_rows]`; the frozen contract requires destinations to be `random_order`, i.e. `derived[field][random_order] = source[field][roll(random_order, 1)]`;
- rejected builder also included an unauthorized conditional fixed-point repair; exact circular shift requires no repair;
- calibration reporting used `shuffled - matched` per seed and omitted the required three-seed mean `matched - shuffled` calibration deltas;
- the existing `codex/task-006a` must not be merged until FIX1 passes; its cache SHA `678571febcf6ce85f2bf7aedb4770b255a138a0cdaff5ea3f04a92606afbd814` and first three runs are superseded diagnostic evidence only;
- assign exactly one corrective pass, `TASK-006A-FIX1`, on the **same** `codex/task-006a` branch;
- FIX1 is a reproducibility correction, not tuning. It authorizes exactly three new corrective shuffled Equal runs on seeds42/43/44.

### MANAGER-OVERRIDE-061 — Reuse codex/task-006a for FIX1

- owner explicitly prefers one branch per logical task; no new `codex/task-006a-fix1` branch;
- reuse of existing `codex/task-006a` is explicitly authorized despite the normal no-reuse rule;
- Codex must fetch and merge current `origin/main` into `codex/task-006a` before corrective implementation so MANAGER-DECISION-060/OVERRIDE-061 are in branch history;
- this one merge from `origin/main` into the task branch is explicitly manager-authorized;
- do not reset, rebase, overwrite, or force-push the branch;
- preserve the original rejected commits/runs as superseded evidence, then append a new corrective firewall commit and final evidence commit;
- exactly three new corrective production runs remain authorized.

### MANAGER-DECISION-062 — Reject FIX1 production for wrong cache binding; assign FIX2 on same branch

- FIX1 branch history correctly merged manager main before corrective work;
- corrected builder implementation at firewall commit `096b720c875b427fc6d17d1dd04b65a67614b802` implements the frozen roll-1 mapping literally and removes the unauthorized repair;
- FIX1 produced and audited the exact cache SHA `e4baf2a2eea9dc7703e73b7956971b0d571eedda55fc8e6984750270712cf0f9`;
- however, configs31/32/33 in both `096b720` and final `d5fbbc5b9ac0fb7136996caf4090fd5d309f5fe7` point to the old rejected cache `shuffled_missing_targets_seed6001.pt`, not `shuffled_missing_targets_seed6001_exact_fix1.pt`;
- the reported FIX1 DEV metrics and checkpoint SHA256 values exactly equal the earlier rejected runs, so FIX1 production evidence is superseded and cannot close the claim;
- do not rebuild or overwrite the verified exact cache; verify it read-only by path/SHA/metadata/invariants before production;
- assign exactly one narrow correction: update the three configs to the exact cache, use distinct FIX2 run names, firewall, then run exactly seeds42/43/44 again;
- reuse `codex/task-006a`; no new Codex branch; no tuning.

### MANAGER-DECISION-063 — Accept TASK-006A-FIX2; close negative-control item 8; assign corpus probe

- TASK-006A-FIX2 passed manager review and was merged via PR #54;
- accepted exact negative-control cache SHA256: `e4baf2a2eea9dc7703e73b7956971b0d571eedda55fc8e6984750270712cf0f9`;
- FIX2 firewall `24fd49ba614514b462422b46baf8ae1c5d0b3c98` preceded final evidence `07c149d4661a4a8afd11f53fcc557e57b5239126`;
- exactly three accepted FIX2 runs used the exact cache and seeds42/43/44;
- shuffled three-seed DEV Mean `0.7371180000` vs matched Equal `0.7791726667`;
- matched-minus-shuffled Mean delta `+0.0420546667`; matched exceeds shuffled on `3/3` seeds;
- therefore `SAMPLE-SPECIFIC PSEUDO ALIGNMENT CLAIM SUPPORTED BY THIS NEGATIVE CONTROL`;
- interpretation remains narrow: no missing-label correctness, comorbidity, significance, or final-promotion claim;
- calibration is mixed by task/metric and is diagnostic only;
- Stage-6 item 8 is CLOSED;
- next sole atomic task is TASK-006B, the frozen R4 corpus-identity probe required by Stage-6 item 9.

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


### TASK-006A-BUNDLE   shuffled pseudo negative-control production evidence

Status: complete. Exactly three production runs completed, in the authorized order: shuffled Equal seed42, seed43, seed44. No sweep, extra seed, optimizer-only run, post-hoc tuning, or source/config change occurred.

Selected only by maximum DEV `dev/mean_score`:

| Seed | Run | Selected epoch | DEV D Score | DEV P Score | DEV Mean | Selected checkpoint SHA256 |
|---:|---|---:|---:|---:|---:|---|
| 42 | `stage6_shuffled_pseudo_equal_seed42_2026-09-28_13-49_wsm_av_r3_disease_query_model_dea1d1da` | 1 | 0.707618 | 0.746877 | 0.727247 | `b24a8be454db99ba735fadf3b040b07a154bfbb78caa52d3ce0664df717c2302` |
| 43 | `stage6_shuffled_pseudo_equal_seed43_2026-09-28_13-52_wsm_av_r3_disease_query_model_a77a85b2` | 3 | 0.689827 | 0.776322 | 0.733074 | `fb0b863509ff10fa19adfc5e07d89e8d8350aa49ea8f58be219a65734da3b8fa` |
| 44 | `stage6_shuffled_pseudo_equal_seed44_2026-09-28_13-55_wsm_av_r3_disease_query_model_7c7c3c9f` | 3 | 0.674850 | 0.807394 | 0.741122 | `95c523d35dc747cdcd1f102661866b9cb4e846d09fb61bf5d5efa12bf9ca2392` |

The three shuffled DEV Mean values have mean `0.7338143333`, sample standard deviation `0.0069670637`, and range `0.727247 0.741122`. Matched Equal DEV Mean values are `0.777920/0.779671/0.779927`, mean `0.7791726667`, sample standard deviation `0.0010923664`, and range `0.777920 0.779927`. Shuffled minus matched paired deltas are seed42 `-0.0506730`, seed43 `-0.0465970`, and seed44 `-0.0388050`; three-seed mean delta is `-0.0453583333`. Matched Equal exceeds shuffled on `3/3` seeds and in the three-seed mean.

Same-epoch Test protocol outputs were recorded only as post-freeze monitoring, never for selection or interpretation. Shuffled TEST_NONE/SOFT/HARD Mean values were seed42 `0.820707/0.837643/0.836956`, seed43 `0.745789/0.745173/0.726297`, and seed44 `0.780918/0.783803/0.780038`.

DEV calibration audit used identical DEV rows and observed masks, with binary Brier score and 15-bin confidence ECE. Observed counts were D/P `621/312` for every seed and both methods.

| Seed | Method | D Brier | D ECE-15 | P Brier | P ECE-15 |
|---:|---|---:|---:|---:|---:|
| 42 | matched Equal | 0.250991 | 0.216796 | 0.090156 | 0.041386 |
| 42 | shuffled | 0.183690 | 0.078068 | 0.133782 | 0.054640 |
| 43 | matched Equal | 0.250396 | 0.221254 | 0.093445 | 0.048379 |
| 43 | shuffled | 0.258899 | 0.218367 | 0.128615 | 0.111257 |
| 44 | matched Equal | 0.254806 | 0.228324 | 0.088165 | 0.038778 |
| 44 | shuffled | 0.262067 | 0.219220 | 0.120834 | 0.082013 |

Calibration deltas (shuffled minus matched) were D Brier `-0.067301/+0.008503/+0.007262` and D ECE `-0.138728/-0.002886/-0.009104` for seeds42/43/44; P Brier `+0.043627/+0.035171/+0.032668` and P ECE `+0.013254/+0.062877/+0.043235`. This is a DEV calibration audit only.

Interpretation: `SAMPLE-SPECIFIC PSEUDO ALIGNMENT CLAIM SUPPORTED BY THIS NEGATIVE CONTROL`, because matched Equal DEV Mean exceeds shuffled on all 3/3 seeds and the matched three-seed mean exceeds shuffled by `0.0453583333`. This supports the sample-specific alignment claim for this frozen negative control; it does not establish missing-label correctness, comorbidity recovery, final-method promotion, or statistical significance.

Production commands were exactly:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train --config-path configs/wsm_mm_pd_dep_v1/ablations/31_shuffled_pseudo_equal_seed42.yaml
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train --config-path configs/wsm_mm_pd_dep_v1/ablations/32_shuffled_pseudo_equal_seed43.yaml
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train --config-path configs/wsm_mm_pd_dep_v1/ablations/33_shuffled_pseudo_equal_seed44.yaml
```

No Test protocol was used for selection, comparison, calibration, or interpretation. No Stage-5 optimization was reopened; no Stage-7, Text/Description, or Final Test work started. Stage 6 remains active for the remaining manager-assigned claims audit.


### TASK-006A-FIX1   corrective pre-training firewall

Status: firewall complete; corrective production has not started. The existing `codex/task-006a` branch was reused under explicit manager authorization. Current `origin/main` `3c413acd7b0f9c7c909f944d611650224bab0c38` was merged first in merge commit `9e6b6aa`. The original TASK-006A implementation and three runs remain in history as NOT MERGED / superseded diagnostic evidence and are excluded from the FIX1 claim decision.

The rejected builder/config working state was corrected before this firewall commit. The exact mapping is: for each task, `order = missing[torch.randperm(...)]`, `source_order = torch.roll(order, shifts=1, dims=0)`, then each permuted field is assigned exactly as `derived[field][order, task] = source[field][source_order, task].clone()`. D uses generator seed `6001`; P uses `6002`. No fixed-point repair, swap, sorting, rejection sampling, remapping, or post-hoc permutation change exists.

Fresh cache command:

```bash
PATH=/media/maxim/Programs/Projects/WSM/.venv/bin:$PATH PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/common/build_ramps_shuffled_pseudo_negative_control.py --source /media/maxim/Programs/Features/WSM/ramps_r2_semantic_v1/train_missing_targets.pt --output /media/maxim/Programs/Features/WSM/ramps_r2_semantic_v1_negative_control/shuffled_missing_targets_seed6001_exact_fix1.pt --shuffle-seed 6001
```

Result: `RAMPS_SHUFFLED_NEGATIVE_CONTROL_PASS`. Source SHA before generation and after generation: `17cf5e67e8c244d81b7c21f0842988966a23d83d69e296f7c5a5181376d6b945`. Fresh FIX1 cache SHA256: `e4baf2a2eea9dc7703e73b7956971b0d571eedda55fc8e6984750270712cf0f9`. Rows `6325`; missing D/P `2665/3660`; accepted D/P `376/1801`; class counts D `376/0`, P `212/1589`; acceptance Hamming D/P `646/1788`. Destination/source sets each equal the complete missing-row set; source equals exact roll-1 destination order; fixed points D/P `0/0`. Permutation SHA256 over ordered int64 `[destination_order, source_order]` pairs: D `1dcfde36463f21d2d2d525d70f6642c6b599cc23534f3cb47ffc41953ec4d4b8`; P `847f8e2cf156cd0d4ecb28f5269d5cc79fd5943c6605f44c57c78528ecc6d767`. Segment IDs, observed masks/targets, tuple multisets, neutral rejected fields, accepted target/probability equality, and observed-truth precedence all passed. The rejected external cache was untouched.

All three FIX1 configs passed `chimera-ml validate-config`. Programmatic equivalence passed for each matched Equal reference: only `data.params.pseudo_cache_path` and `run_name` differ; seeds, architecture, Equal loss, optimizer, warm-up, callbacks, logging, and all other semantics are unchanged. The FIX1 cache loads through the existing semantic DataModule.

TRAIN-only firewall smoke was run for each config at `pseudo_scale=1.0`, with no DEV/Test loader iteration and no optimizer step:

- seed42: `FIX1_CONFIG_FIREWALL_SMOKE_PASS`, trainable params `403079`, finite loss `0.7784184813`;
- seed43: `FIX1_CONFIG_FIREWALL_SMOKE_PASS`, trainable params `403079`, finite loss `0.7689405670`;
- seed44: `FIX1_CONFIG_FIREWALL_SMOKE_PASS`, trainable params `403079`, finite loss `0.8094737520`.

For each smoke, observed supervision and accepted missing-head pseudo supervision produced non-zero finite model gradients; pseudo target/reliability gradients were `None`; main heads, audio/video projections, task queries, and gate gradients were finite and non-zero. `git diff --check` passed and `git diff origin/main -- src` was empty.

No corrective production run may start before this firewall is committed and pushed. The only authorized subsequent commands are the three FIX1 production commands in order: seeds42, 43, 44.


### TASK-006A-FIX1   corrective production and claim evidence

Status: complete. Exactly three NEW corrective production runs completed in order: FIX1 seed42, seed43, seed44. The prior rejected runs remain superseded diagnostic evidence and are not reused, averaged, selected among, or counted.

Run identities:

- seed42: `stage6_shuffled_pseudo_equal_seed42_exact_fix1_2026-09-28_14-48_wsm_av_r3_disease_query_model_71480159`; MLflow ID `75fbd0da0dc9439389ca8026a8902405`; status `FINISHED`; artifact URI `/media/maxim/Programs/Projects/WSM/mlruns/5/75fbd0da0dc9439389ca8026a8902405/artifacts`;
- seed43: `stage6_shuffled_pseudo_equal_seed43_exact_fix1_2026-09-28_14-51_wsm_av_r3_disease_query_model_79faa7ba`; MLflow ID `9a72fa9c8ec5469b821ea0b74b44a379`; status `FINISHED`; artifact URI `/media/maxim/Programs/Projects/WSM/mlruns/5/9a72fa9c8ec5469b821ea0b74b44a379/artifacts`;
- seed44: `stage6_shuffled_pseudo_equal_seed44_exact_fix1_2026-09-28_14-54_wsm_av_r3_disease_query_model_b4590a31`; MLflow ID `6e760048f620431da3c2dc4f430b2a17`; status `FINISHED`; artifact URI `/media/maxim/Programs/Projects/WSM/mlruns/5/6e760048f620431da3c2dc4f430b2a17/artifacts`.

DEV-only selected checkpoints:

| Seed | Epoch | D UAR/MF1/Score | P UAR/MF1/Score | Mean | Checkpoint SHA256 |
|---:|---:|---|---|---:|---|
| 42 | 1 | 0.707796/0.707439/0.707618 | 0.737681/0.756072/0.746877 | 0.727247 | `b24a8be454db99ba735fadf3b040b07a154bfbb78caa52d3ce0664df717c2302` |
| 43 | 3 | 0.690056/0.689597/0.689827 | 0.766322/0.786323/0.776322 | 0.733074 | `fb0b863509ff10fa19adfc5e07d89e8d8350aa49ea8f58be219a65734da3b8fa` |
| 44 | 3 | 0.675023/0.674677/0.674850 | 0.801794/0.812994/0.807394 | 0.741122 | `95c523d35dc747cdcd1f102661866b9cb4e846d09fb61bf5d5efa12bf9ca2392` |

FIX1 shuffled DEV D/P/Mean: seed42 `0.707618/0.746877/0.727247`; seed43 `0.689827/0.776322/0.733074`; seed44 `0.674850/0.807394/0.741122`. Three-seed means are D `0.690765`, P `0.7768643333`, Mean `0.7338143333`; sample standard deviations are D `0.0164041257`, P `0.0302621449`, Mean `0.0069670637`.

Matched Equal minus FIX1 shuffled deltas are:

| Seed | D delta | P delta | Mean delta |
|---:|---:|---:|---:|
| 42 | +0.004880 | +0.096465 | +0.050673 |
| 43 | +0.013472 | +0.079721 | +0.046597 |
| 44 | +0.033127 | +0.044484 | +0.038805 |

Aggregate matched-minus-FIX1 deltas are D `+0.0171600000`, P `+0.0735566667`, and Mean `+0.0453583333`. Matched DEV Mean exceeds FIX1 on `3/3` seeds and in the three-seed mean.

Corrected DEV calibration audit used identical DEV rows/masks; observed counts were D/P `621/312` for every seed and method. Values and deltas are matched minus FIX1:

| Seed | D matched Brier | D FIX1 Brier | D delta | D matched ECE-15 | D FIX1 ECE-15 | D delta | P matched Brier | P FIX1 Brier | P delta | P matched ECE-15 | P FIX1 ECE-15 | P delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.250991 | 0.183690 | +0.067301 | 0.216796 | 0.078068 | +0.138728 | 0.090156 | 0.133782 | -0.043627 | 0.041386 | 0.054640 | -0.013254 |
| 43 | 0.250396 | 0.258899 | -0.008503 | 0.221254 | 0.218367 | +0.002886 | 0.093445 | 0.128615 | -0.035171 | 0.048379 | 0.111257 | -0.062877 |
| 44 | 0.254806 | 0.262067 | -0.007262 | 0.228324 | 0.219220 | +0.009103 | 0.088165 | 0.120834 | -0.032668 | 0.038778 | 0.082013 | -0.043235 |

Three-seed mean matched-minus-FIX1 calibration deltas: D Brier `+0.0171788832`; D ECE-15 `+0.0502391444`; P Brier `-0.0371552979`; P ECE-15 `-0.0397886039`. This audit is diagnostic only; no recalibration or rerun occurred.

Same-epoch Test monitoring, read only after checkpoint freeze and never used for selection or interpretation: seed42 TEST_NONE/SOFT/HARD Mean `0.820707/0.837643/0.836956`; seed43 `0.745789/0.745173/0.726297`; seed44 `0.780918/0.783803/0.780038`.

Frozen claim result: `SAMPLE-SPECIFIC PSEUDO ALIGNMENT CLAIM SUPPORTED BY THIS NEGATIVE CONTROL`. This is narrow sample-specific pseudo-alignment evidence only. No claim is made about missing-label correctness, comorbidity recovery, significance, or final promotion. No tuning, sweep, retry for metric improvement, post-hoc config change, source change, or additional run occurred.


### TASK-006A-FIX2 — cache-binding firewall

Status: firewall complete; FIX2 production has not started. FIX1 production is rejected/superseded because configs31/32/33 were bound to the old rejected cache path `/media/maxim/Programs/Features/WSM/ramps_r2_semantic_v1_negative_control/shuffled_missing_targets_seed6001.pt`. The accepted exact cache is `/media/maxim/Programs/Features/WSM/ramps_r2_semantic_v1_negative_control/shuffled_missing_targets_seed6001_exact_fix1.pt`.

The existing `codex/task-006a` branch was reused after merging current manager `origin/main` `4722fca3464ca60f8840b0f67efc2ff927d24c41`. The corrected FIX1 builder and exact cache were not modified or regenerated.

Read-only exact-cache verification passed:

- cache SHA256: `e4baf2a2eea9dc7703e73b7956971b0d571eedda55fc8e6984750270712cf0f9`;
- source SHA256 remains `17cf5e67e8c244d81b7c21f0842988966a23d83d69e296f7c5a5181376d6b945`;
- mapping: `destination_order = order`, `source_order = roll(order, 1)`, and `derived[field][order, task] = source[field][source_order, task]`;
- permutation SHA256: D `1dcfde36463f21d2d2d525d70f6642c6b599cc23534f3cb47ffc41953ec4d4b8`, P `847f8e2cf156cd0d4ecb28f5269d5cc79fd5943c6605f44c57c78528ecc6d767`;
- fixed points D/P `0/0`; acceptance Hamming D/P `646/1788`;
- missing D/P `2665/3660`; accepted D/P `376/1801`; class counts D `376/0`, P `212/1589`;
- semantic DataModule loaded the exact cache and resolved its exact path/SHA.

Corrected configs:

- `31_shuffled_pseudo_equal_seed42.yaml` -> run name `stage6_shuffled_pseudo_equal_seed42_exact_fix2`;
- `32_shuffled_pseudo_equal_seed43.yaml` -> run name `stage6_shuffled_pseudo_equal_seed43_exact_fix2`;
- `33_shuffled_pseudo_equal_seed44.yaml` -> run name `stage6_shuffled_pseudo_equal_seed44_exact_fix2`.

All three passed config validation and programmatic equivalence against matched Equal configs13/19/20: only the exact cache path and run name differ.

FIX2 firewall smoke passed for all three configs with exact DataModule resolution, R3 trainable parameters `403079`, finite loss, nonzero observed/pseudo model gradients, finite main-head/projection/query/gate gradients, and no pseudo-target/reliability gradients:

- seed42 loss `0.7388503551`;
- seed43 loss `0.7636069059`;
- seed44 loss `0.7564906478`.

No optimizer step and no DEV/Test loader iteration occurred. `git diff --check` passed and `git diff origin/main -- src` was empty.

The firewall commit must precede exactly three FIX2 production commands in order: seeds42, 43, 44. No sweep, tuning, retry, or post-hoc config change is authorized.


### TASK-006A-FIX2 — corrected cache binding production evidence

Status: complete. FIX1 production is rejected because configs31/32/33 used the old wrong cache path `/media/maxim/Programs/Features/WSM/ramps_r2_semantic_v1_negative_control/shuffled_missing_targets_seed6001.pt`. The accepted exact cache path is `/media/maxim/Programs/Features/WSM/ramps_r2_semantic_v1_negative_control/shuffled_missing_targets_seed6001_exact_fix1.pt`, SHA256 `e4baf2a2eea9dc7703e73b7956971b0d571eedda55fc8e6984750270712cf0f9`. The exact cache was verified read-only; it was not regenerated or overwritten.

The new FIX2 firewall commit is `24fd49b`. Exactly three NEW FIX2 production runs completed in order: seed42, seed43, seed44. No prior shuffled run was reused, averaged, selected among, or counted.

Run identities:

- seed42: `stage6_shuffled_pseudo_equal_seed42_exact_fix2_2026-09-28_15-17_wsm_av_r3_disease_query_model_fdd3d5a9`; MLflow ID `1783b4c6424842b6a6cb88aa38c4c164`; status `FINISHED`; artifact URI `/media/maxim/Programs/Projects/WSM/mlruns/5/1783b4c6424842b6a6cb88aa38c4c164/artifacts`;
- seed43: `stage6_shuffled_pseudo_equal_seed43_exact_fix2_2026-09-28_15-21_wsm_av_r3_disease_query_model_21343cc1`; MLflow ID `23d9b071710347d7983d79b7ea43351f`; status `FINISHED`; artifact URI `/media/maxim/Programs/Projects/WSM/mlruns/5/23d9b071710347d7983d79b7ea43351f/artifacts`;
- seed44: `stage6_shuffled_pseudo_equal_seed44_exact_fix2_2026-09-28_15-25_wsm_av_r3_disease_query_model_6c38e6e8`; MLflow ID `8941f5f4bb0d485cb07bbd00e56b1980`; status `FINISHED`; artifact URI `/media/maxim/Programs/Projects/WSM/mlruns/5/8941f5f4bb0d485cb07bbd00e56b1980/artifacts`.

DEV-only selected results:

| Seed | Epoch | D UAR/MF1/Score | P UAR/MF1/Score | Mean | Checkpoint SHA256 |
|---:|---:|---|---|---:|---|
| 42 | 5 | 0.675490/0.674841/0.675166 | 0.785645/0.812657/0.799151 | 0.737158 | `baa2336955ea192acc1304a26c5f3aad3affadc6df3e13aa278228a5a91832a7` |
| 43 | 3 | 0.690056/0.689597/0.689827 | 0.766322/0.786323/0.776322 | 0.733074 | `fb0b863509ff10fa19adfc5e07d89e8d8350aa49ea8f58be219a65734da3b8fa` |
| 44 | 3 | 0.675023/0.674677/0.674850 | 0.801794/0.812994/0.807394 | 0.741122 | `95c523d35dc747cdcd1f102661866b9cb4e846d09fb61bf5d5efa12bf9ca2392` |

FIX2 shuffled D/P/Mean values are seed42 `0.675166/0.799151/0.737158`, seed43 `0.689827/0.776322/0.733074`, and seed44 `0.674850/0.807394/0.741122`. Three-seed means are D `0.6799476667`, P `0.7942890000`, Mean `0.7371180000`; sample standard deviations are D `0.0085572124`, P `0.0160964772`, Mean `0.0040241491`.

Matched Equal minus FIX2 deltas:

| Seed | D delta | P delta | Mean delta |
|---:|---:|---:|---:|
| 42 | +0.037332 | +0.044191 | +0.040762 |
| 43 | +0.013472 | +0.079721 | +0.046597 |
| 44 | +0.033127 | +0.044484 | +0.038805 |

Aggregate matched-minus-FIX2 deltas are D `+0.0279770000`, P `+0.0561320000`, and Mean `+0.0420546667`. Matched DEV Mean exceeds FIX2 on `3/3` seeds and in the three-seed mean.

DEV calibration audit used identical rows/masks, with observed counts D/P `621/312` for every seed. Values are matched, FIX2, then matched-minus-FIX2:

| Seed | D Brier | D ECE-15 | P Brier | P ECE-15 |
|---:|---|---|---|---|
| 42 | 0.250991 / 0.249330 / +0.001661 | 0.216796 / 0.189452 / +0.027344 | 0.090156 / 0.122209 / -0.032053 | 0.041386 / 0.095453 / -0.054067 |
| 43 | 0.250396 / 0.258899 / -0.008503 | 0.221254 / 0.218367 / +0.002886 | 0.093445 / 0.128615 / -0.035171 | 0.048379 / 0.111257 / -0.062877 |
| 44 | 0.254806 / 0.262067 / -0.007262 | 0.228324 / 0.219220 / +0.009103 | 0.088165 / 0.120834 / -0.032668 | 0.038778 / 0.082013 / -0.043235 |

Three-seed mean matched-minus-FIX2 calibration deltas: D Brier `-0.0047011574`; D ECE-15 `+0.0131110949`; P Brier `-0.0332974344`; P ECE-15 `-0.0533929869`. No recalibration or rerun occurred.

Same-epoch Test monitoring was read only after checkpoint freeze and did not affect selection or interpretation: seed42 TEST_NONE/SOFT/HARD Mean `0.777046/0.771482/0.778540`; seed43 `0.745789/0.745173/0.726297`; seed44 `0.780918/0.783803/0.780038`.

Frozen claim result: `SAMPLE-SPECIFIC PSEUDO ALIGNMENT CLAIM SUPPORTED BY THIS NEGATIVE CONTROL`. This is narrow sample-specific pseudo-alignment evidence only. No missing-label correctness, comorbidity recovery, significance, or final-promotion claim is made. No tuning, sweep, retry, post-hoc config change, or source change occurred.


### TASK-006B — corpus-probe firewall

Status: firewall complete; full probe not yet executed. Branch: `codex/task-006b`, based on manager `origin/main` `6487f1e`.

Implemented only [`scripts/common/run_r4_corpus_probe.py`](../scripts/common/run_r4_corpus_probe.py). The firewall command was:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python scripts/common/run_r4_corpus_probe.py --firewall

Firewall passed for all three frozen configurations/checkpoints: checkpoint files exist and SHA256 matches seed42 `104b79e409832839502b4d20a98c054f5efda633930e313ce74f34327e668b2a`, seed43 `6af4a4ed21040aff4db2906b0adc8aa5fd27a4b72c375fbb67f68de04ab8446e`, and seed44 `b06fec6912b27cd19d811b72b2c3c08cd1b0a6c826b637f87a314e3976b88c17`. Correct model state loaded at epochs 11/10/11 and models were in eval mode. Tiny two-row canonical TRAIN+DEV extraction passed for every seed; all six representations were finite with exact dimensions audio/video/projected A+V/task fused/gates/logits = `160/160/320/320/4/2`. Corpus labels were read from `sample_meta[*]["corpus"]` only. No Test loader was iterated.

Local AUROC synthetic checks passed for perfect `1.0`, reversed `0.0`, and tied constant `0.5`; the deterministic zero-initialized float64 LBFGS probe sanity check passed. Frozen probe settings are encoded in the script: TRAIN-only population standardization, class-balanced weighted BCE, weight-only L2 `1e-4`, and LBFGS `lr=1.0`, `max_iter=250`, `tolerance_grad=1e-10`, `tolerance_change=1e-12`, `history_size=50`, `strong_wolfe`. `git diff --check` passed and `git diff origin/main -- src` was empty. No main-model training occurred.

Mandatory firewall commit and push must precede the single full-probe command.
