# WSM Project Progress

## 1. Current State

Plan initialized: 2026-09-23.

Current stage: **Stage 6 — ACTIVE (ablations and claims audit)**.

Stage 5: **CLOSED TO OPTIMIZATION — R4 RETAINED AS LEADING MATCHED-SEED CANDIDATE**.

Stage 7: **LOCKED**.

Final Test authorized: **no**.

Current atomic task: **TASK-006F — semantic-evidence contribution via the depression pseudo-acceptance path**.

Expected Codex branch: `codex/task-006f`.

`TASK-006E` is complete and merged. Stage-6 uncertainty/reliability contribution is CLOSED as supported under the predeclared three-seed gate, with a small aggregate DEV Mean effect.

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

## 7. Active Task — TASK-006F

Purpose: isolate the **semantic-evidence-enabled depression pseudo-acceptance contribution** inside the leading optimized R4 trial-012 composition.

Historical Stage-5 evidence is structurally important:

- pre-semantic audio+video reliability could not deploy any depression side at the unchanged precision/support gate;
- Parkinson audio+video rules were already valid and frozen;
- fixed CLIP semantic evidence was introduced only to recover the blocked depression pseudo path;
- the accepted semantic cache therefore contains D/P pseudo counts `376/1801`, where the 376 depression pseudo entries are the semantic-enabled addition.

TASK-006F derives a no-semantic-depression cache by neutralizing exactly those 376 depression pseudo entries while preserving the Parkinson column exactly.

Because the current semantic DataModule hardcodes the full cache class/count contract, TASK-006F may make one narrow source change: parameterize the expected accepted/positive/negative counts with defaults equal to the current frozen values. Existing full-method configs must behave identically when those optional parameters are omitted.

Exactly three production runs are authorized: seeds42/43/44. No model/loss/callback/audio/video change and no tuning are authorized. Full executable requirements are authoritative in [NEXT_TASK_EN.md](NEXT_TASK_EN.md).

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

### MANAGER-DECISION-064 — Accept TASK-006B; close corpus probe; assign isolated RA-STCH audit

- TASK-006B passed manager review and was merged via PR #56;
- firewall commit `3ba0fcce68550eccf61a34d4e20e55a4111212f4` preceded final evidence `ec258e16373f38adabc03213bde8b4d65f14d30b`;
- probe report SHA256: `3361de90e0cb26f14e15f639bb54ef4f7d29d54232b72fb39f154d7f7105a4dc`;
- no main-model retraining, Test-loader iteration, source change, config change, or probe tuning occurred;
- three-seed true corpus AUROC means: audio `0.791292`, video `0.705505`, projected A+V `0.750650`, task-fused `0.759832`, gates `0.551605`, logits `0.575197`;
- task-fused minus projected-A+V AUROC deltas were `+0.024588/-0.000165/+0.003123`, mean `+0.009182`;
- frozen interpretations are `STRONG CORPUS-ID DECODABILITY FLAG NOT TRIGGERED` and `FUSION AMPLIFICATION OF CORPUS DECODABILITY NOT SUPPORTED`;
- corpus identity remains structurally coupled to observed-task identity; no causal shortcut or disease-signal replacement claim is authorized;
- no separate domain-mitigation experiment is triggered by this probe;
- Stage-6 corpus-probe item 9 is CLOSED;
- next sole atomic task is TASK-006C: complete the original fixed-composition RA-STCH ablation on true seeds43/44 and compare three-seed evidence against Equal/Static/Progress.

### MANAGER-DECISION-065 — Accept TASK-006C; close RA-STCH contribution claim negative; assign direct pseudo ablation

- TASK-006C passed manager review and was merged via PR #58;
- firewall commit `45b17ebb6d54acb602ba9698677a37b47292e38d` preceded final evidence `c7d201fe9ff463187827dc3d4a60a7884349c032`;
- exactly two new frozen RA-STCH runs occurred, seed43 then44; retained seed42 was not rerun;
- three-seed fixed-composition RA D/P/Mean = `0.6931503333/0.8519640000/0.7725573333`;
- RA minus Equal Mean by seed = `+0.004725/+0.002163/-0.026734`; aggregate Mean delta `-0.0066153333`;
- RA minus best simpler balancing Mean by seed = `-0.004654/-0.010530/-0.024654`;
- therefore `RA-STCH BALANCING CONTRIBUTION OVER EQUAL NOT SUPPORTED`;
- and `RA-STCH ADVANTAGE OVER SIMPLER BALANCING NOT SUPPORTED`;
- controller diagnostics remained finite/bounded but do not rescue the failed DEV gates;
- this negative mechanism result does not automatically demote the separately optimized trial-012 composition, whose hyperparameters differ and which remains the leading matched-seed candidate pending remaining Stage-6 component audits;
- Stage-6 balancing/RA-STCH claim is CLOSED as negative;
- next sole atomic task is TASK-006D: remove only the direct pseudo-supervision loss term from optimized trial-012 on seeds42/43/44 while retaining pseudo acceptance/reliability/controller information.

### MANAGER-DECISION-066 — Accept TASK-006D; direct pseudo contribution supported; assign reliability ablation

- TASK-006D passed manager review and was merged via PR #60;
- firewall commit `5a5a657c56cdc2eee35c7e85c990022083ab70a1` preceded final evidence `7239d38333ec0019f94644bd10bc5a63aff48e5f`;
- configs40/41/42 differ from full trial-012 only by run_name and pseudo warm-up `final_scale=0.0`;
- exactly three runs occurred, seeds42/43/44;
- no-direct-pseudo three-seed D/P/Mean = `0.7134376667/0.8263193333/0.7698780000`;
- full trial-012 minus no-direct-pseudo aggregate D/P/Mean = `+0.0218186667/+0.0122413333/+0.0170308179`;
- full DEV Mean exceeded the ablation on 2/3 seeds and in aggregate, with no aggregate D/P unacceptable regression under the frozen gate;
- therefore `DIRECT PSEUDO-SUPERVISION CONTRIBUTION SUPPORTED`;
- interpretation is limited to the direct pseudo BCE term; pseudo-label correctness, missing-label recovery, comorbidity recovery, significance, and final promotion remain unauthorized;
- Stage-6 direct pseudo-supervision claim is CLOSED as supported;
- next sole atomic task is TASK-006E: remove graded reliability while preserving pseudo acceptance and targets, then compare three seeds against full trial-012.

### MANAGER-DECISION-067 — Accept TASK-006E; reliability contribution supported; assign semantic-evidence ablation

- TASK-006E passed manager review and was merged via PR #62;
- firewall commit `699115362466518b7a9a02ef6f2aec89c5b70a62` preceded final evidence `cda75f32d1915d9e71cd029a5ba45feee501731b`;
- derived uniform-reliability cache SHA256: `713b5a3d963c8759e4b2c12f148be12818e72c135b6d1420207cf5aa150bcc40`;
- accepted rows/targets/classes/calibrated probabilities were unchanged and accepted reliability was exactly 1.0;
- uniform-reliability three-seed D/P/Mean = `0.7352583333/0.8345853333/0.7849220000`;
- full-minus-ablation aggregate D/P/Mean = `-0.0000020000/+0.0039753333/+0.0019868179`;
- full Mean exceeded the ablation on 2/3 seeds and in aggregate, so `UNCERTAINTY/RELIABILITY CONTRIBUTION SUPPORTED`;
- the effect is small and the claim is limited to the combined implemented graded-reliability signal; BCE weighting and controller reliability are not separately identified;
- Stage-6 uncertainty/reliability item is CLOSED as supported;
- next sole atomic task is TASK-006F: remove the semantic-enabled depression pseudo-acceptance path while preserving the frozen Parkinson pseudo path and all full trial-012 training semantics.

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


### TASK-006B — frozen R4 corpus probe

Status: complete; Stage-6 corpus-probe item 9 evidence produced. Full command executed exactly once after firewall push:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python scripts/common/run_r4_corpus_probe.py --output /media/maxim/Programs/Features/WSM/stage6_corpus_probe/r4_corpus_probe_v1.json

External report: `/media/maxim/Programs/Features/WSM/stage6_corpus_probe/r4_corpus_probe_v1.json`; SHA256 `3361de90e0cb26f14e15f639bb54ef4f7d29d54232b72fb39f154d7f7105a4dc`. Firewall commit: `3ba0fcce68550eccf61a34d4e20e55a4111212f4`; final evidence commit follows. Frozen checkpoint paths/SHA: seed42 `logs/wsm_mm_pd_dep_v1/optuna_r4_ra_stch_base_seed42_2026-09-28_01-54_wsm_av_r3_disease_query_model_r4-ra-stch-optuna-v1-18c6-012_0ce9c7a4/checkpoints/epoch=11_dev_mean_score=0.8191.pt` / `104b79e409832839502b4d20a98c054f5efda633930e313ce74f34327e668b2a`; seed43 `logs/wsm_mm_pd_dep_v1/r4_ra_stch_optuna_trial012_confirm_seed43_2026-09-28_12-01_wsm_av_r3_disease_query_model_12038580/checkpoints/epoch=10_dev_mean_score=0.7614.pt` / `6af4a4ed21040aff4db2906b0adc8aa5fd27a4b72c375fbb67f68de04ab8446e`; seed44 `logs/wsm_mm_pd_dep_v1/r4_ra_stch_optuna_trial012_confirm_seed44_2026-09-28_12-07_wsm_av_r3_disease_query_model_41bc602b/checkpoints/epoch=11_dev_mean_score=0.7801.pt` / `b06fec6912b27cd19d811b72b2c3c08cd1b0a6c826b637f87a314e3976b88c17`.

Canonical counts were TRAIN `6325` (depression `2665`, parkinson `3660`) and DEV `933` (depression `312`, parkinson `621`) for each seed. All six representations completed with dimensions `audio_projected=160`, `video_projected=160`, `projected_av=320`, `task_fused=320`, `gates=4`, `logits=2`. Probes used TRAIN-only standardization, CPU float64 zero initialization, class-balanced weighted BCE, weight-only L2 `1e-4`, and exact LBFGS settings from NEXT_TASK. No Test loader was iterated or inspected; no main-model training or re-selection occurred; no source/config changes occurred.

Per-seed DEV AUROC (true / shuffled; true-minus-shuffled):

- seed42: audio `0.774320 / 0.559834; +0.214486`, video `0.708318 / 0.460408; +0.247910`, projected_av `0.746470 / 0.510075; +0.236395`, task_fused `0.771058 / 0.609119; +0.161939`, gates `0.674398 / 0.670047; +0.004351`, logits `0.619214 / 0.637986; -0.018771`.
- seed43: audio `0.785747 / 0.608938; +0.176808`, video `0.715998 / 0.511288; +0.204710`, projected_av `0.740028 / 0.577640; +0.162388`, task_fused `0.739863 / 0.476501; +0.263362`, gates `0.711853 / 0.707002; +0.004852`, logits `0.616097 / 0.629000; -0.012903`.
- seed44: audio `0.813808 / 0.613521; +0.200287`, video `0.692199 / 0.483515; +0.208684`, projected_av `0.765453 / 0.530611; +0.234841`, task_fused `0.768575 / 0.541605; +0.226971`, gates `0.268565 / 0.278382; -0.009817`, logits `0.490281 / 0.646946; -0.156664`.

Three-seed true AUROC mean/sample std: audio `0.791292/0.020300`, video `0.705505/0.011934`, projected_av `0.750650/0.013012`, task_fused `0.759832/0.017222`, gates `0.551605/0.244405`, logits `0.575197/0.073718`. Shuffled true-control means: audio `0.594098`, video `0.485070`, projected_av `0.539442`, task_fused `0.542408`, gates `0.551810`, logits `0.637977`. Task-fused minus projected_av AUROC deltas by seed: `+0.024588`, `-0.000165`, `+0.003123`; mean `+0.009182`, below the frozen amplification gate.

Gate audit audio-weight mean depression-minus-parkinson: seed42 depression query `-0.123735`, Parkinson query `-0.125313`; seed43 `-0.090391`, `-0.091659`; seed44 `-0.037737`, `-0.046210`. Full mean/q25/median/q75/sample-std distributions are in the external report.

Frozen interpretation strings: `STRONG CORPUS-ID DECODABILITY FLAG NOT TRIGGERED`; `FUSION AMPLIFICATION OF CORPUS DECODABILITY NOT SUPPORTED`. Corpus identity is structurally coupled to which disease label is observed, so this remains a risk diagnostic; no causal shortcut, disease-signal replacement, R4 promotion/demotion, mitigation, or significance claim is made. Stage-5 optimization remains closed; Stage 6 remains active; Stage 7, Text/Description, and Final Test remain locked.


### TASK-006C — RA-STCH firewall

Status: firewall complete; production runs not started. Branch: `codex/task-006c`, based on manager `origin/main` `b228191`. Only `docs/PROGRESS_EN.md` is authorized to change.

Validated exactly:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml validate-config --config-path configs/wsm_mm_pd_dep_v1/fusion/16_r4_ra_stch_seed42.yaml
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml validate-config --config-path configs/wsm_mm_pd_dep_v1/fusion/17_r4_ra_stch_seed43.yaml
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml validate-config --config-path configs/wsm_mm_pd_dep_v1/fusion/18_r4_ra_stch_seed44.yaml

All three configs were valid. Programmatic equivalence passed: configs17/18 differ from config16 only by top-level seed and run_name; seeds are exactly `42/43/44`. Retained seed42 checkpoint exists with SHA256 `53397e3bbcbfc95abd30bdf63fec018a28f0effc2d92d66231061fb2d051254a`; semantic cache exists with SHA256 `17cf5e67e8c244d81b7c21f0842988966a23d83d69e296f7c5a5181376d6b945`.

Firewall smoke passed for configs17/18 using TRAIN-only canonical mixed-corpus rows, with no optimizer step and no DEV/Test iteration:

- trainable parameters: `403079` for both;
- semantic accepted pseudo counts: D/P `376/1801`;
- finite loss: seed43 `0.5106716156`, seed44 `0.5835766196`;
- finite nonzero model gradients;
- detached pseudo target/reliability tensors;
- finite RA diagnostics; seed43 grad-norm EMA D/P `0.094643/0.079261`, grad-cosine EMA `0.0`, reliability EMA D/P `0.650805/0.876119`; seed44 `0.141683/0.091648`, `0.0`, `0.650805/0.876119`;
- controller weights remained within frozen `[0.2,0.8]` bounds.

`git diff --check`, `git diff origin/main -- src`, and `git diff origin/main -- configs` passed. No source/config/script changes occurred. The mandatory firewall commit must be pushed before exactly two production runs in order: seed43, then seed44.


### TASK-006C — isolated RA-STCH balancing audit

Status: complete. Branch: `codex/task-006c`. Firewall commit `45b17ebb6d54acb602ba9698677a37b47292e38d` was pushed before production. Only the two authorized production runs occurred, in order, with no seed42 rerun, sweep, retry, config change, or optimized trial-012 run:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train --config-path configs/wsm_mm_pd_dep_v1/fusion/17_r4_ra_stch_seed43.yaml
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train --config-path configs/wsm_mm_pd_dep_v1/fusion/18_r4_ra_stch_seed44.yaml

Retained seed42 was not rerun: selected epoch 13, DEV D/P/Mean `0.700992/0.864298/0.782645`, checkpoint SHA256 `53397e3bbcbfc95abd30bdf63fec018a28f0effc2d92d66231061fb2d051254a`. New production identities: seed43 run `r4_ra_stch_seed43_c1_2026-09-28_16-07_wsm_av_r3_disease_query_model_39bc2679`, MLflow ID `25b36566c88046cd98614595fa682523`, FINISHED, selected epoch 4, DEV D/P/Mean `0.698839/0.864829/0.781834`, checkpoint SHA256 `aa75298c6b557eda3d7e8594a77b0c55661abd9c67cd4ed6b4f1687b8eba4262`; seed44 run `r4_ra_stch_seed44_c1_2026-09-28_16-12_wsm_av_r3_disease_query_model_49a0dea2`, MLflow ID `3d3ade0154b64f73817c6d5dc2685411`, FINISHED, selected epoch 4, DEV D/P/Mean `0.679620/0.826765/0.753193`, checkpoint SHA256 `7d9510cfef6f4a871e5447c7142aa30e4ef23798576d735f45efc7ff4774f916`. Selection used DEV/mean_score only.

Three-seed RA DEV evidence (seed42/43/44): D mean/std/range `0.6931503333/0.0117669580/0.021372`; P `0.8519640000/0.0218245890/0.038064`; Mean `0.7725573333/0.0167749060/0.029452`. Same-seed RA minus Equal D/P/Mean: seed42 `-0.011506/+0.020956/+0.004725`; seed43 `-0.004460/+0.008786/+0.002163`; seed44 `-0.028357/-0.025113/-0.026734`. Three-seed RA minus Equal means: D/P/Mean `-0.0147743333/+0.0015430000/-0.0066153333`.

Same-seed RA minus Static-STCH D/P/Mean: seed42 `-0.024085/+0.014813/-0.004636`; seed43 `-0.000106/+0.011645/+0.005770`; seed44 `-0.022452/-0.026857/-0.024654`. Same-seed RA minus corrected Progress D/P/Mean: seed42 `-0.000253/-0.009056/-0.004654`; seed43 `-0.022758/+0.001698/-0.010530`; seed44 `-0.022351/-0.020443/-0.021397`. Best-simple (`max(Static,Progress)`) RA Mean deltas: seed42 `-0.004654`; seed43 `-0.010530`; seed44 `-0.024654`; mean `-0.0132793333`. RA minus contextual corrected R3-B means (D/P/Mean `0.696178/0.857996/0.777087`) is `-0.0030276667/-0.0060320000/-0.0045296667`.

Frozen claims: `RA-STCH BALANCING CONTRIBUTION OVER EQUAL NOT SUPPORTED`; `RA-STCH ADVANTAGE OVER SIMPLER BALANCING NOT SUPPORTED`. Rule A fails because RA three-seed Mean is below Equal despite exceeding Equal on 2/3 seeds; Rule B fails all same-seed best-simple Mean comparisons and both simpler three-seed Mean comparisons.

Selected-epoch controller diagnostics (alpha D/P; progress D/P; grad-norm EMA D/P; grad-cosine EMA; reliability EMA D/P; pseudo scale): seed42 `0.269192/0.730808; +0.020070/-0.151804; 0.248995/0.121096; +0.009814; 0.671974/0.773563; 1.0`; seed43 `0.353600/0.646400; +0.001794/-0.366308; 0.451421/0.248391; -0.030544; 0.672728/0.770895; 0.2`; seed44 `0.362685/0.637315; -0.051906/-0.264870; 0.280293/0.130971; +0.014310; 0.671132/0.781050; 0.2`. All diagnostics are finite; alpha is within `[0.2,0.8]`, seed-specific rather than identical, grad-cosine range is `-0.030544` to `+0.014310`, reliability D range `0.671132–0.672728`, P range `0.770895–0.781050`.

Post-freeze DEV-only calibration/gate audit used observed counts D/P `621/312` for every seed and no Test rows. RA three-seed means: D Brier/ECE-15 `0.2505394063/0.2257061193`, P `0.0952518079/0.0757453021`; task audio-gate mean/sample-std D `0.3630868495/0.1935827980`, P `0.3701813122/0.1922030300`. RA minus frozen Equal calibration means: D Brier/ECE `-0.0015246377/-0.0058562747`, P `+0.0046632639/+0.0142875391`. RA minus corrected Progress: D `+0.0025287953/+0.0075696203`, P `+0.0100845399/+0.0059937391`.

Same-epoch Test monitoring was read only after each selected checkpoint was frozen and did not affect selection, calibration, comparison, or claims: seed42 TEST_NONE/SOFT/HARD `0.777133/0.767804/0.760071`; seed43 `0.746138/0.739737/0.729451`; seed44 `0.797072/0.798145/0.805330`. No Test row was used in post-hoc diagnostics. Frozen pseudo coverage remains D/P `376/1801`, with class counts D `376/0` and P `212/1589`. Stage-5 optimization remains closed; Stage 6 remains active; Stage 7, Text/Description, and Final Test remain locked.


### TASK-006D — no direct pseudo-supervision firewall

Status: firewall complete; production runs not started. Branch: `codex/task-006d`, based on manager `origin/main` `b927bf8`. Added only configs40/41/42; no source, existing-config, or script changes.

Validated all three configs with `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml validate-config --config-path ...`: all valid. Programmatic comparison against fusion refs37/38/39 passed: only `experiment_info.params.run_name` and warmup `final_scale` differ; required seeds are 42/43/44, `loss.params.pseudo_scale=0.0`, observed-only epochs `3`, ramp epochs `5`, and cache/reliability settings unchanged. `scale_for_epoch(1..30)==0.0`.

Firewall command completed with TRAIN-only forward/loss/backward checks for all three configs: frozen semantic cache SHA `17cf5e67e8c244d81b7c21f0842988966a23d83d69e296f7c5a5181376d6b945`; accepted counts D/P `376/1801`, classes D `376/0`, P `212/1589`; accepted rows present; trainable parameters `403079`; pseudo targets/reliability detached; controller diagnostics finite and reliability EMA updated from accepted reliability. Deterministic pseudo-target perturbation left total loss and model gradients unchanged within strict `1e-12` tolerance because pseudo scale is zero. No optimizer step and no DEV/Test iteration occurred. `git diff origin/main -- src` and `git diff origin/main -- configs` were empty; `git diff --check` passed.

Mandatory firewall commit/push precedes exactly three production runs in order seed42, seed43, seed44.


### TASK-006D — three-seed no direct pseudo-supervision ablation

Status: complete; Stage-6 direct pseudo-supervision ablation evidence produced. Firewall commit `5a5a657` was pushed before production. Exactly three production runs occurred, in order, with no retries, sweep, extra seed, optimized-trial rerun, or post-firewall config change:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train --config-path configs/wsm_mm_pd_dep_v1/ablations/40_no_direct_pseudo_seed42.yaml
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train --config-path configs/wsm_mm_pd_dep_v1/ablations/41_no_direct_pseudo_seed43.yaml
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train --config-path configs/wsm_mm_pd_dep_v1/ablations/42_no_direct_pseudo_seed44.yaml

Runs were FINISHED in MLflow: seed42 `06355f9a57154ccf8201595ff3982098`, seed43 `8b2eeb1879a840bbbeaf72d7e081999e`, seed44 `289fd655d74e4e75a89a50af7df32b49`. DEV/mean_score-only selected checkpoints:

| Seed | Epoch | DEV D/P/Mean | Checkpoint SHA256 |
|---:|---:|---|---|
| 42 | 7 | `0.750015/0.849786/0.799900` | `d0991ddf92b8fd556d1c57dfa3c17b3b083e5b16a1e21d04592cece804b57669` |
| 43 | 19 | `0.683769/0.770704/0.727236` | `f5958bd638b021836e9ead892815c17822579fb796d7e074ad61c6a86fc96892` |
| 44 | 12 | `0.706529/0.858468/0.782498` | `d67232b2ba3395ced7a1ac95e95aacfbdaa354ce68ee5e507f37954f54fd968c` |

No-direct three-seed DEV means/sample SDs: D `0.7134376667/0.0336590313`, P `0.8263193333/0.0483595209`, Mean `0.7698780000/0.0379402494`. Frozen full trial-012 means are D/P/Mean `0.7352563333/0.8385606667/0.7869088179`; full-minus-ablation mean deltas are `+0.0218186667/+0.0122413333/+0.0170308179`. Full minus ablation per-seed Mean deltas are seed42 `+0.0192424538`, seed43 `+0.0342090000`, seed44 `-0.0023590000`. The predeclared gate passes (`2/3` per-seed DEV Mean wins, higher three-seed Mean, and task means no more than `0.010` lower), so the exact frozen claim is: `DIRECT PSEUDO-SUPERVISION CONTRIBUTION SUPPORTED`. This concerns only the direct pseudo BCE term.

Selected-epoch controller diagnostics (alpha D/P; progress D/P; grad-norm EMA D/P; grad-cosine EMA; reliability EMA D/P; pseudo scale): seed42 `0.312667/0.687333; +0.190917/-0.493318; 0.279859/0.266622; 0.000000; 0.674348/0.757363; 0.000000`; seed43 `0.212604/0.787396; -0.137111/-0.608890; 0.437248/0.243887; 0.000000; 0.673802/0.779818; 0.000000`; seed44 `0.256048/0.743952; +0.094125/-0.245362; 0.386038/0.230013; 0.000000; 0.663937/0.782318; 0.000000`. All finite; alpha remains within `[0.2,0.8]`. Accepted pseudo coverage remained D/P `376/1801`, classes D `376/0` and P `212/1589`; reliability/controller paths remained active.

Post-freeze DEV-only calibration/gate audit iterated only `dm.val_dataset`, with observed counts D/P `621/312` per seed and no Test row. Per-seed `(D Brier, D ECE-15, D audio-gate mean/std; P Brier, P ECE-15, P audio-gate mean/std)`:

- seed42: `0.180180/0.110249/0.433490/0.195184; 0.084544/0.045940/0.559094/0.182210`
- seed43: `0.264118/0.231195/0.376595/0.144412; 0.160619/0.153661/0.437323/0.140837`
- seed44: `0.218597/0.182075/0.430556/0.156280; 0.087115/0.079878/0.490820/0.121299`

Three-seed no-direct calibration/gate means: D Brier/ECE `0.2209647345/0.1745065810`, P `0.1107593512/0.0931596507`, audio gate D mean/std `0.4135470291/0.1652921041`, P `0.4957457086/0.1481153195`. Frozen full means were D Brier/ECE `0.1789253730/0.0700041175`, P `0.0899480473/0.0898388979`, gate D `0.4017960926/0.1487621441`, P `0.4908300142/0.1301527048`; full-minus-ablation deltas respectively `-0.0420393616/-0.1045024635`, `-0.0208113039/-0.0033207529`, `-0.0117509365/-0.0165299600`, `-0.0049156944/-0.0179626147`. No recalibration or threshold tuning occurred.

Same-epoch Test monitoring was read only after each DEV checkpoint was frozen: seed42 TEST_NONE/SOFT/HARD `0.803498/0.812067/0.803166`, seed43 `0.805364/0.801955/0.811135`, seed44 `0.796826/0.802473/0.793822`. Test did not affect selection, audit, comparison, or claim; no Test row was used in post-hoc diagnostics. No pseudo-label correctness, missing-label recovery, comorbidity recovery, significance, domain mitigation, promotion, or demotion claim is made. Stage-5 optimization remains closed; Stage 6 remains active; Stage 7, Text/Description, and Final Test remain locked.

Final scope checks: `git diff --check` passed; `git diff origin/main -- src` and `git diff origin/main -- configs` are empty; only configs40/41/42 and this ledger differ from origin/main.


### TASK-006E — uniform accepted-reliability firewall

Status: firewall complete; production runs not started. Branch: `codex/task-006e`, based on manager `origin/main` `919a5a7`.

Added [`scripts/common/build_ramps_uniform_reliability_ablation.py`](../scripts/common/build_ramps_uniform_reliability_ablation.py) and configs43/44/45. The builder command was:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python scripts/common/build_ramps_uniform_reliability_ablation.py --source /media/maxim/Programs/Features/WSM/ramps_r2_semantic_v1/train_missing_targets.pt --output /media/maxim/Programs/Features/WSM/ramps_r2_semantic_v1_reliability_ablation/uniform_accepted_reliability_v1.pt

Source SHA before/after: `17cf5e67e8c244d81b7c21f0842988966a23d83d69e296f7c5a5181376d6b945`. Derived SHA: `713b5a3d963c8759e4b2c12f148be12818e72c135b6d1420207cf5aa150bcc40`. The builder refuses overwrite, preserves canonical version and all non-reliability fields, and records only explicit `reliability_ablation` metadata. Accepted counts D/P `376/1801`; classes D `376/0`, P `212/1589`; accepted reliability is exactly `1.0`; rejected/observed reliability exactly `0.0`. Source accepted reliability min/mean/max: D `0.6104820873/0.6726164732/0.7853315456`, P `0.5000977972/0.7718680412/0.9037118705`. Changed accepted reliability entries D/P: `376/1801`.

All configs validated successfully. Programmatic equivalence against refs37/38/39 passed: only run_name and pseudo_cache_path differ; seeds remain 42/43/44. Frozen full checkpoint files and all three required SHA256 values verified. TRAIN-only firewall smoke passed for every config: derived cache loaded through the existing DataModule; trainable parameters `295239`; accepted pseudo rows present; pseudo targets/reliability detached; direct pseudo supervision at scale `1.0` produced finite nonzero model gradients; RA reliability EMA updated exactly to `[1.0, 1.0]`; controller diagnostics were finite. No optimizer step and no DEV/Test iteration occurred. `git diff --check` passed and `git diff origin/main -- src` was empty.

Mandatory firewall commit/push precedes exactly three production runs in order seed42, seed43, seed44.


### TASK-006E — uniform accepted-reliability production ablation

Status: complete; Stage-6 uncertainty/reliability evidence produced. Firewall commit `6991153` was pushed before production. Exactly three production runs occurred in order, with no sweep, retry, extra seed, or post-firewall cache/config change:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train --config-path configs/wsm_mm_pd_dep_v1/ablations/43_uniform_reliability_seed42.yaml
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train --config-path configs/wsm_mm_pd_dep_v1/ablations/44_uniform_reliability_seed43.yaml
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train --config-path configs/wsm_mm_pd_dep_v1/ablations/45_uniform_reliability_seed44.yaml

All MLflow runs were FINISHED: seed42 `017b81c94ec24726a61e448723634dd5`, seed43 `c8247e56925f434e8c41b39eac764e82`, seed44 `ebe5d6fcd91749d5893613dc3d8b9d7d`. DEV/mean_score-only selected checkpoints:

| Seed | Epoch | DEV D/P/Mean | Checkpoint SHA256 |
|---:|---:|---|---|
| 42 | 11 | `0.755716/0.871962/0.813839` | `e3eeab3eb974c65b4410b35a183e42cbe0d484cc0e21f231543ef0f4ffe51065` |
| 43 | 10 | `0.720095/0.799734/0.759915` | `76098413ad16314f0575a1b574a8d8bf851fa6bc67039f154acc0272eba94554` |
| 44 | 11 | `0.729964/0.832060/0.781012` | `3de5862828ddabe36c507990780fe690269079520d99d8a9111f67c11ffb78b7` |

Uniform-reliability three-seed DEV mean/sample SD/range: D `0.7352583333/0.0183912040/0.035621`, P `0.8345853333/0.0361801600/0.072228`, Mean `0.7849220000/0.0271738021/0.053924`. Frozen full trial-012 mean is D/P/Mean `0.7352563333/0.8385606667/0.7869088179`; aggregate full-minus-ablation deltas are D `-0.0000020000`, P `+0.0039753333`, Mean `+0.0019868179`. Per-seed full-minus-ablation D/P/Mean deltas: seed42 `+0.003309/+0.007297/+0.0053034538`, seed43 `-0.001568/+0.004629/+0.0015300000`, seed44 `-0.001747/+0.000000/-0.0008730000`.

The exact frozen gate passes: full DEV Mean wins `2/3` seeds, full three-seed Mean is higher, and full D/P means are not more than `0.010000` below the ablation. Record exactly: `UNCERTAINTY/RELIABILITY CONTRIBUTION SUPPORTED`. This concerns the implemented graded reliability signal as a whole, jointly covering pseudo-BCE reliability weighting and RA reliability input; it does not separate those effects.

Selected controller diagnostics (alpha D/P; progress D/P; grad-norm EMA D/P; grad-cosine EMA; reliability EMA D/P; pseudo scale): seed42 `0.391610/0.608390; +0.143760/+0.082452; 0.379403/0.259262; +0.045782; 1.000000/1.000000; 1.000000`; seed43 `0.249567/0.750433; +0.050214/-0.701800; 0.356957/0.284160; -0.005376; 1.000000/1.000000; 1.000000`; seed44 `0.291637/0.708363; +0.119063/-0.194524; 0.344999/0.259215; +0.009421; 1.000000/1.000000; 1.000000`. All diagnostics were finite and reliability EMA reached exactly `1.0/1.0` for every selected run.

Post-freeze DEV-only calibration/gate audit used observed counts D/P `621/312` per seed and no Test rows. Per-seed `(D Brier, D ECE-15, D audio-gate mean/std; P Brier, P ECE-15, P audio-gate mean/std)`:

- seed42: `0.175110/0.075461/0.419239/0.196150; 0.071968/0.069305/0.549952/0.170355`
- seed43: `0.188070/0.093503/0.349542/0.111722; 0.111041/0.088769/0.433419/0.103559`
- seed44: `0.176722/0.062885/0.435787/0.134986; 0.087910/0.115290/0.488573/0.111126`

Uniform-reliability three-seed means: D Brier/ECE `0.1799669747/0.0772831558`, P `0.0903061726/0.0911213222`, audio gate D mean/std `0.4015226364/0.1476193691`, P `0.4906478624/0.1283467487`. Full-minus-ablation deltas: D Brier/ECE `-0.0010416017/-0.0072790382`, P `-0.0003581253/-0.0012824243`, gate D mean/std `+0.0002734562/+0.0011427750`, P `+0.0001821518/+0.0018059562`. No recalibration or threshold search occurred.

Same-epoch Test monitoring was read only after checkpoint freeze: seed42 TEST_NONE/SOFT/HARD `0.801394/0.811302/0.808134`; seed43 `0.820499/0.828142/0.841820`; seed44 `0.822560/0.833204/0.850724`. Test did not affect selection, interpretation, or follow-up, and no Test rows entered the post-hoc audit. No pseudo-label correctness, semantic acceptance correctness, missing-label recovery, comorbidity recovery, significance, or final-promotion claim is made. Stage-5 optimization remains closed; Stage 6 remains active; Stage 7, Text/Description, and Final Test remain locked.

Final scope checks: source cache remained SHA `17cf5e67e8c244d81b7c21f0842988966a23d83d69e296f7c5a5181376d6b945`; derived cache remains external at SHA `713b5a3d963c8759e4b2c12f148be12818e72c135b6d1420207cf5aa150bcc40`; `git diff --check` passed; `git diff origin/main -- src` was empty; only the builder, configs43/44/45, and this ledger are tracked changes.


### TASK-006F — no semantic-enabled depression pseudo path firewall

Status: firewall complete; production runs not started. Branch: `codex/task-006f`, based on manager `origin/main` `6fd035c`.

Authorized source change: [`src/fusion/data/wsm_ramps_semantic_datamodule.py`](../src/fusion/data/wsm_ramps_semantic_datamodule.py) now accepts optional expected accepted/positive/negative count contracts, defaulting exactly to `(376,1801)`, `(376,212)`, `(0,1589)`. Supplied contracts require exactly two non-negative integers; expected missing counts remain fixed at `(2665,3660)` and all other cache invariants are unchanged.

Full backward compatibility passed for configs37/38/39 with omitted parameters: full counts remained D/P `376/1801`, classes D `376/0`, P `212/1589`, and a canonical TRAIN batch was tensor-identical against an isolated clean `origin/main` DataModule implementation.

Builder: [`scripts/common/build_ramps_no_semantic_depression_ablation.py`](../scripts/common/build_ramps_no_semantic_depression_ablation.py). Source SHA before/after: `17cf5e67e8c244d81b7c21f0842988966a23d83d69e296f7c5a5181376d6b945`. Derived cache: `/media/maxim/Programs/Features/WSM/ramps_r2_semantic_v1_semantic_ablation/no_semantic_depression_v1.pt`, SHA `aae9d250c5030eab1e68e07ee8af43fe8ebb7286eca621f9fa5c6dd7af283fe6`. Exactly 376 D accepted rows were removed; all 1801 P accepted rows, targets, reliability, and classes were preserved exactly. D fields are neutralized to accept false, NaN targets, zero reliability, and class `-1`.

Configs46/47/48 validated and differ from refs37/38/39 only by run_name, cache path, and the three explicit expected-count contracts. Frozen full checkpoint paths/SHA values were verified. TRAIN-only firewall passed for all three ablation configs: trainable parameters `295239`; D accepted `0`; P accepted `1801`, classes `212/1589`; P pseudo gradients finite/nonzero at pseudo scale `1.0`; D pseudo contribution absent; pseudo tensors detached; P reliability EMA finite (`0.738575935` in the deterministic smoke batch); D reliability EMA uninitialized with controller fallback `0.5`; controller weights/diagnostics finite. No optimizer step and no DEV/Test loader iteration occurred. `git diff --check` passed; all forbidden source scopes (`src/audio`, `src/video`, `src/fusion/models`, `src/fusion/loss`, `src/common/callbacks`) were empty.

Mandatory firewall commit/push precedes exactly three production runs in order seed42, seed43, seed44.
