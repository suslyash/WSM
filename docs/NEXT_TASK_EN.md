# TASK-005E-BUNDLE: Complete R3 Contract and Run a Bounded Predeclared Experiment Bundle

## Authority and branch

This task is an explicit owner/manager override for the current R3 cycle only. It supersedes the earlier "stop after TASK-005E" restriction and allows several predeclared experiments in one Codex cycle. All PROJECT_REQUIREMENTS invariants remain mandatory.

Required branch:

    codex/task-005e

If the branch does not exist, create it from current origin/main containing manager override commit:

    64eac820be7ddfcc726b88e2e6a6fe11bb074050

If the branch already exists from the earlier TASK-005E assignment, reuse is explicitly authorized. Preserve all work and history; fetch origin and merge current origin/main into the task branch if needed. Do not reset, rebase away work, force-push, or touch main/master.

## High-level objective

Finish the frozen R3 disease-query A+V model contract, then execute a maximum of four production training runs from a predeclared experiment tree:

1. R3-A — R3 model + ordinary observed sparse loss, seed 42.
2. R3-B — same R3 model + one fixed auxiliary unimodal agreement loss, seed 42.
3. If neither A nor B passes the frozen DEV screen, stop.
4. If at least one passes, choose exactly one continuation variant by the frozen DEV-only rule and run that same variant at seeds 43 and 44.
5. Stop after the bundle. Do not invent another variant.

No R2 pseudo cache/loss/warm-up may appear in any R3 run.

## Required reading

Read in this order:

1. AGENTS.md
2. docs/README.md
3. docs/PROJECT_REQUIREMENTS.md
4. docs/PLAN.md, especially Stage 5 R3
5. docs/PROGRESS_EN.md through OWNER/MANAGER-OVERRIDE-041
6. docs/NEXT_TASK_EN.md
7. docs/SOTA_REVIEW_EN.md Sections 9–11
8. src/fusion/models/av_f2_task_aware_directed.py
9. src/fusion/data/wsm_av_fusion_datamodule.py
10. src/common/loss/wsm_masked_sparse_loss.py
11. src/chimera_plugin.py
12. configs/wsm_mm_pd_dep_v1/fusion/02_f2_task_aware_directed.yaml

## Non-negotiable experiment firewall

Before the first production training invocation, commit all of the following:

- final R3 model implementation;
- final R3-B auxiliary loss implementation;
- plugin registrations;
- all seed-42 and conditional seed-43/44 YAMLs for both A and B;
- exact loss weights;
- frozen DEV screen;
- continuation/tie-break rule.

After the first production run begins, do not change architecture, loss formula, loss weights, optimizer, schedule, data, early stopping, screen, seed policy, or branching logic based on observed metrics.

TEST_NONE/SOFT/HARD are monitoring only and MUST NOT influence any branch decision.

Maximum production training invocations: 4.

## Allowed tracked files

- src/fusion/models/av_r3_disease_query.py
- src/fusion/loss/r3_aux_agreement_loss.py
- src/chimera_plugin.py
- configs/wsm_mm_pd_dep_v1/fusion/06_ramps_r3_disease_query_contract.yaml
- configs/wsm_mm_pd_dep_v1/fusion/07_r3_a_sparse_seed42.yaml
- configs/wsm_mm_pd_dep_v1/fusion/08_r3_b_agreement_seed42.yaml
- configs/wsm_mm_pd_dep_v1/fusion/09_r3_a_sparse_seed43.yaml
- configs/wsm_mm_pd_dep_v1/fusion/10_r3_a_sparse_seed44.yaml
- configs/wsm_mm_pd_dep_v1/fusion/11_r3_b_agreement_seed43.yaml
- configs/wsm_mm_pd_dep_v1/fusion/12_r3_b_agreement_seed44.yaml
- docs/PROGRESS_EN.md

No other tracked file may change.

## Phase 1 — R3 model contract

Implement the previously assigned R3 model contract unchanged:

Registry key:

    wsm_av_r3_disease_query_model

Required architecture:

- audio_cls 768 and pooled video 512 projected to hidden 192;
- learned disease queries shape [2,192], normal init std 0.02;
- task-specific candidate LayerNorms;
- one shared query-conditioned audio/video gate;
- exact modality availability masking;
- task-specific fused features;
- two independent main disease heads;
- auxiliary audio-only and video-only logits with validity masks;
- no corpus identity, observed mask, disease label, or sample task_id as an input feature;
- trainable parameter count <= frozen F2 count 736004.

Required synthetic availability/invariance, all-unavailable guard, registry/config, and real TRAIN-only forward/loss/backward smokes remain exactly as in the previous TASK-005E contract.

Do not train until all Phase-1 contract checks pass.

## Phase 2 — fixed R3-B auxiliary loss

Implement:

    src/fusion/loss/r3_aux_agreement_loss.py

Registry key:

    wsm_r3_aux_agreement_loss

Constructor defaults, frozen for this bundle:

    aux_weight = 0.25
    agreement_weight = 0.10
    eps = 1e-8

Require non-negative finite weights and eps.

The loss consumes:

- main logits output.preds [B,2];
- observed targets and observed_mask;
- output.aux["audio_aux_logits"] [B,2];
- output.aux["video_aux_logits"] [B,2];
- output.aux["audio_aux_valid"] [B,2];
- output.aux["video_aux_valid"] [B,2].

Unknown labels remain masked everywhere.

Define:

    L_main = ordinary observed masked BCE on main logits

For auxiliary supervised BCE, use only observed AND modality-valid entries. Pool all eligible audio/video task elements together:

    L_aux =
        sum(BCE(audio_aux, target) over observed & audio_valid
          + BCE(video_aux, target) over observed & video_valid)
        / (eligible_aux_count + eps)

If no eligible auxiliary entry exists, use an exact differentiable zero.

For agreement, use only observed entries where both modalities are valid:

    p_a = sigmoid(audio_aux_logits)
    p_v = sigmoid(video_aux_logits)

    L_agree =
        mean((p_a - p_v)^2 over observed & audio_valid & video_valid)

If no eligible agreement entry exists, use an exact differentiable zero.

Total:

    L = L_main + 0.25 * L_aux + 0.10 * L_agree

Do not detach the auxiliary logits from this loss.

Do not introduce pseudo targets, teacher logits, semantic scores, Test information, or additional loss terms.

Add a synthetic smoke proving:

- unknown targets never enter any term;
- main and auxiliary heads get finite gradients;
- both modality projections get finite gradients;
- agreement term is finite;
- zero-valid auxiliary/agreement cases remain finite;
- no optimizer step.

## Phase 3 — predeclare all configs before training

Create all six production configs before the first run.

Common semantics for every config:

- experiment_name: wsm_mm_pd_dep_v1
- canonical wsm_av_fusion_datamodule
- R3 model and exact frozen dimensions
- AdamW lr 1e-4, weight_decay 0.01
- batch_size 32
- max epochs 30
- mixed precision true
- grad clip 0.5
- checkpoint and early stopping monitor only dev/mean_score, mode=max
- patience 6, min_delta 0.0005
- required four-stream instrumentation/loggers
- no pseudo cache/loss/warm-up

Variant A configs:

- seed 42: 07_r3_a_sparse_seed42.yaml
- seed 43: 09_r3_a_sparse_seed43.yaml
- seed 44: 10_r3_a_sparse_seed44.yaml
- loss: wsm_masked_sparse_loss

Variant B configs:

- seed 42: 08_r3_b_agreement_seed42.yaml
- seed 43: 11_r3_b_agreement_seed43.yaml
- seed 44: 12_r3_b_agreement_seed44.yaml
- loss: wsm_r3_aux_agreement_loss
- aux_weight: 0.25
- agreement_weight: 0.10

Run names must encode variant and seed.

Validate all six configs before any production run.

## Frozen comparator and seed-42 DEV screen

Frozen sparse F2 comparator:

- depression Score = 0.697035
- Parkinson Score = 0.852104
- Mean_Score = 0.774569

A seed-42 R3 variant passes only if:

- selected DEV Mean_Score > 0.774569;
- depression DEV Score >= 0.687035;
- Parkinson DEV Score >= 0.842104.

Selection of each checkpoint uses DEV/Mean_Score only.

A failed variant remains a valid negative result.

## Frozen continuation rule

After both seed-42 runs finish:

- If neither passes: stop the bundle.
- If exactly one passes: continue that variant.
- If both pass:
  1. choose higher selected DEV Mean_Score;
  2. if absolute Mean difference <= 1e-6, choose the variant with higher minimum of the two task-score deltas versus F2;
  3. if still tied, choose R3-A as the simpler method.

Only the chosen continuation variant may run seeds 43 and 44.

Do not use any Test metric in this rule.

## Production run authorization

After every contract/firewall/config check passes, run exactly:

1. R3-A seed 42.
2. R3-B seed 42.
3. Conditionally, chosen variant seed 43.
4. Conditionally, chosen variant seed 44.

Do not rerun a failed/crashed completed production run with modified settings. If an infrastructure failure occurs before meaningful training begins, record it and only retry the identical command/settings; do not change the method.

## Required reporting

For every executed run record in docs/PROGRESS_EN.md:

- exact config and command;
- run directory and MLflow metadata;
- epochs completed and early-stop status;
- selected epoch/checkpoint by DEV only;
- selected DEV depression/Parkinson UAR, MF1, Score, Mean;
- delta versus F2;
- pass/fail of the frozen screen;
- same-epoch TEST_NONE/SOFT/HARD monitoring values only after DEV selection is frozen;
- confirmation Test did not affect selection or branching.

For seed-42 A and B also report:

- trainable parameter count;
- post-hoc DEV task modality-weight means/stds for each disease;
- auxiliary audio/video DEV Score on observed+valid rows for R3-B as diagnostics only.

If seeds 43/44 execute, report for the chosen variant:

- DEV D/P/Mean for seeds 42/43/44;
- arithmetic mean and sample standard deviation across the three seeds;
- do not declare final significance or final-model status from only three seeds.

## Scope and integrity

Always preserve:

- two independent disease outputs;
- masked unknown labels;
- observed truth authority;
- src/audio unchanged;
- src/video unchanged;
- no R2 pseudo path;
- no text/description;
- no R4;
- no Stage 6/7;
- no Final Test;
- no missing-label correctness/comorbidity claims.

Run git diff --check and explicit source-scope diffs before handoff.

## Final handoff

Respond in English using exactly:

1. Outcome
2. Changed files
3. Verification
4. Plan status
5. Blockers and risks
6. Next atomic step

Include:

- branch and all implementation/evidence commit SHAs;
- pushed-to-origin status;
- main/master untouched;
- exact R3 parameter count;
- model/loss registry keys;
- contract smoke results;
- confirmation all configs/loss weights/screens were frozen before first run;
- number of production invocations actually executed;
- seed-42 A and B DEV results and screen status;
- continuation decision and exact DEV-only reason;
- seed 43/44 results if executed;
- three-seed mean/std if available;
- Test monitoring-only confirmation;
- no post-hoc tuning;
- R2 remains negative and unused;
- no R4/text/Stage6/7/Final Test;
- src/audio/src/video unchanged;
- Stage 5 status.

Stop after this bounded bundle. Do not invent or execute another experiment.
