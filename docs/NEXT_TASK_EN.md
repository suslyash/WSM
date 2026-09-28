# TASK-006I: Equal-Parameter Task-Aware Fusion Ablation on Optimized R4 Trial-012

## Authority and branch

This task follows **MANAGER-DECISION-070**.

Required branch:

    codex/task-006i

Start from current manager-updated `origin/main`.

Create exactly one new task branch from that main.

Stage-5 optimization remains closed. This is a Stage-6 component ablation only.

## Scientific question

Does **task-aware fusion specialization** contribute repeatably to the leading optimized R4 trial-012 composition when sparse MTL, separate disease heads, pseudo supervision, reliability, RA controller, optimizer, and model parameter count are held fixed?

This task removes task-conditioned fusion specialization while retaining sparse multi-task learning.

It does NOT test whether sparse MTL itself is useful. That is a separate Stage-6 claim.

## Required reading

Read in order:

1. `AGENTS.md`
2. `docs/README.md`
3. `docs/PROJECT_REQUIREMENTS.md`
4. `docs/PROGRESS_EN.md`
5. `docs/plan/STAGE_6.md`
6. `docs/NEXT_TASK_EN.md`
7. `src/fusion/models/av_r3_disease_query.py`
8. `src/fusion/loss/r4_ramps_balance_loss.py`
9. `src/fusion/data/wsm_ramps_semantic_datamodule.py`
10. configs37/38/39.

Historical Stage-4 F1/F2 evidence may be read only for context. It does not replace this required three-seed trial-012 ablation.

## Allowed tracked files

Only:

- `src/fusion/models/av_r3_disease_query.py`
- `configs/wsm_mm_pd_dep_v1/ablations/55_shared_fusion_seed42.yaml`
- `configs/wsm_mm_pd_dep_v1/ablations/56_shared_fusion_seed43.yaml`
- `configs/wsm_mm_pd_dep_v1/ablations/57_shared_fusion_seed44.yaml`
- `docs/PROGRESS_EN.md`

Forbidden:

- DataModule changes;
- loss/callback changes;
- `src/audio/*` and `src/video/*` changes;
- existing config changes;
- pseudo-cache changes;
- tuning.

## Minimal backward-compatible model switch

Extend `WSMAVR3DiseaseQueryModel` with:

    task_aware_fusion: bool = true

Validation:
- value must be boolean;
- default true.

Registry/provider behavior remains unchanged except accepting the optional model parameter.

### Full/default path

When `task_aware_fusion=true`, forward behavior must be tensor-identical to current `origin/main`.

Do not refactor the full path in a way that changes numerical ordering.

### Shared-fusion ablation path

When `task_aware_fusion=false`, retain all existing parameter tensors but remove task-specific fusion specialization as follows.

Let:

    q_shared = mean(task_queries, dim=0)

For each batch, create the same `query_batch` from `q_shared` for both tasks.

Shared audio candidate:

    audio_candidate =
        0.5 * (
            task_candidate_norms[0](features_audio + query_batch)
            + task_candidate_norms[1](features_audio + query_batch)
        )

Shared video candidate:

    video_candidate =
        0.5 * (
            task_candidate_norms[0](features_video + query_batch)
            + task_candidate_norms[1](features_video + query_batch)
        )

Compute exactly one audio score and one video score using the existing `shared_query_gate`, apply the existing availability mask, and obtain one two-modality softmax weight vector.

Let:

    fused_pre =
        query_batch
        + weight_audio * audio_candidate
        + weight_video * video_candidate

Shared fused representation:

    fused_shared =
        0.5 * (
            task_fusion_norms[0](fused_pre)
            + task_fusion_norms[1](fused_pre)
        )

Then:

- `task_audio_features[:,0]` and `[:,1]` are both `audio_candidate`;
- `task_video_features[:,0]` and `[:,1]` are both `video_candidate`;
- `task_modality_weights[:,0]` and `[:,1]` are the same shared weights;
- `task_features[:,0]` and `[:,1]` are both `fused_shared`;
- main head 0 consumes `fused_shared` for depression;
- main head 1 consumes `fused_shared` for Parkinson;
- audio/video auxiliary heads remain task-specific and consume the shared audio/video candidate respectively.

Do not tie, delete, freeze, or resize any existing parameter tensor.

Both task-query parameters, both candidate norms, both fusion norms, both task heads, both audio aux heads, both video aux heads, projections, and shared gate must remain trainable and participate in the graph.

## Equal-parameter contract

Full and ablation trainable parameter counts MUST both be exactly:

    295239

Because parameter count is identical, no separate size-matched control is required for this comparison.

Record this explicitly.

## Frozen full reference

Full trial-012 DEV:

| Seed | D | P | Mean |
|---:|---:|---:|---:|
| 42 | 0.759025 | 0.879259 | 0.8191424538 |
| 43 | 0.718527 | 0.804363 | 0.7614450000 |
| 44 | 0.728217 | 0.832060 | 0.7801390000 |

Three-seed full:

- D mean/std `0.7352563333/0.0211467766`;
- P mean/std `0.8385606667/0.0378688091`;
- Mean `0.7869088179/0.0294384420`.

Full checkpoint SHA256:

- seed42 `104b79e409832839502b4d20a98c054f5efda633930e313ce74f34327e668b2a`;
- seed43 `6af4a4ed21040aff4db2906b0adc8aa5fd27a4b72c375fbb67f68de04ab8446e`;
- seed44 `b06fec6912b27cd19d811b72b2c3c08cd1b0a6c826b637f87a314e3976b88c17`.

Verify these before production.

## Exact ablation configs

Create configs55/56/57 from refs37/38/39 respectively.

Allowed semantic differences only:

1. `experiment_info.params.run_name`;
2. `model.params.task_aware_fusion: false`.

Required run names:

- seed42: `stage6_shared_fusion_trial012_seed42`
- seed43: `stage6_shared_fusion_trial012_seed43`
- seed44: `stage6_shared_fusion_trial012_seed44`

Everything else must match the corresponding full config.

## Mandatory pre-run firewall

Before any production run:

1. implement the model switch;
2. prove default/full backward compatibility:
   - instantiate a clean `origin/main` model and the modified model with `task_aware_fusion=true`;
   - same fixed RNG initialization/state dict;
   - same deterministic TRAIN batch;
   - in eval mode, every `preds` and relevant `aux` tensor must be bit/tensor-identical;
3. validate configs55/56/57;
4. prove each differs from refs37/38/39 only by run_name/task_aware_fusion;
5. verify seeds remain 42/43/44;
6. verify full checkpoint paths/SHA values;
7. instantiate DataModule/model/loss/callbacks;
8. verify full and ablation trainable params both exactly `295239`;
9. verify pseudo-cache path/SHA/counts/classes unchanged;
10. shared-fusion forward invariants:
    - task audio candidates exactly equal across tasks;
    - task video candidates exactly equal across tasks;
    - modality weights exactly equal across tasks;
    - task fused features exactly equal across tasks;
    - preds remain two distinct task logits from separate main heads;
11. verify task-specific heads remain independent:
    - deterministic perturbation of depression main-head parameters changes depression logits but not Parkinson logits on the same frozen shared representation;
12. verify both task-query parameter rows receive finite nonzero gradients and, under the shared mean construction, their gradients are exactly equal within strict tolerance;
13. verify both candidate norm modules and both fusion norm modules receive finite nonzero gradients;
14. verify both main heads and all active auxiliary heads receive finite gradients;
15. verify projections/shared gate receive finite nonzero gradients;
16. verify observed and accepted pseudo supervision remain active at pseudo_scale=1;
17. pseudo target/reliability tensors remain detached;
18. controller diagnostics finite;
19. no optimizer step;
20. no DEV/Test iteration;
21. `git diff --check`;
22. verify forbidden source scopes unchanged;
23. append firewall evidence to PROGRESS;
24. commit and push one firewall commit.

No production run before the firewall commit is visible on origin.

## Exactly three production runs

Run exactly in order:

1. seed42;
2. seed43;
3. seed44.

Commands:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train \
      --config-path configs/wsm_mm_pd_dep_v1/ablations/55_shared_fusion_seed42.yaml

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train \
      --config-path configs/wsm_mm_pd_dep_v1/ablations/56_shared_fusion_seed43.yaml

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train \
      --config-path configs/wsm_mm_pd_dep_v1/ablations/57_shared_fusion_seed44.yaml

No sweep, retry for metric improvement, extra seed, or post-firewall change.

## Selection/Test firewall

For every run:

- select checkpoint only by maximum `dev/mean_score`;
- freeze checkpoint identity before reading same-epoch Test monitoring;
- Test cannot affect the claim or next-step decision.

## Three-seed comparison

Compute shared-fusion D/P/Mean by seed and three-seed mean/sample std/range.

Compute same-seed and aggregate:

    full trial012 - shared-fusion

for D/P/Mean.

## Frozen claim rule

Record exactly:

    TASK-AWARE FUSION CONTRIBUTION SUPPORTED

only if ALL hold:

1. full DEV Mean > shared-fusion DEV Mean on at least 2/3 seeds;
2. full three-seed Mean > shared-fusion three-seed Mean;
3. full three-seed D is not more than `0.010000` below shared-fusion D;
4. full three-seed P is not more than `0.010000` below shared-fusion P.

Otherwise record exactly:

    TASK-AWARE FUSION CONTRIBUTION NOT SUPPORTED

This claim concerns task-conditioned fusion specialization only.

Do not claim sparse-MTL contribution from this experiment.

## Fusion / gradient diagnostics

At each selected checkpoint record:

- pseudo scale;
- alpha D/P;
- progress D/P;
- controller grad-norm EMA D/P;
- controller grad-cosine EMA;
- reliability EMA D/P;
- shared audio gate mean/std on DEV;
- maximum absolute difference between task0 and task1 modality weights — MUST be zero within numerical tolerance;
- maximum absolute difference between task0 and task1 fused features — MUST be zero within numerical tolerance.

Additionally, on a deterministic TRAIN diagnostic batch with observed support for both tasks and no parameter update:

- compute D-only and P-only gradient L2 norms over the shared fusion parameter set:
  - audio_projection;
  - video_projection;
  - task_queries;
  - task_candidate_norms;
  - shared_query_gate;
  - task_fusion_norms;
- compute cosine similarity between D-only and P-only gradient vectors;
- record finite values.

This is diagnostic only and cannot rescue a failed DEV gate.

## DEV-only calibration audit

After checkpoint freeze, use DEV only.

For each seed/task report:

- observed count;
- Brier;
- ECE-15.

Report three-seed means and full-minus-ablation calibration deltas.

No recalibration or threshold search.
No Test rows.

## Required PROGRESS evidence

Record:

- task/branch;
- model source change and full-path backward-compatibility proof;
- exact shared-fusion equations/invariants;
- equal parameter count proof;
- configs55/56/57 equivalence;
- full checkpoint SHAs;
- pseudo-cache identity;
- firewall evidence/SHA;
- exact three production commands;
- exactly three run/MLflow identities;
- selected epochs/checkpoint SHAs;
- DEV D/P UAR/MF1/Score/Mean;
- Test monitoring after freeze only;
- three-seed shared-fusion aggregate;
- full-minus-ablation deltas;
- exact frozen claim string;
- fusion/controller/gradient diagnostics;
- DEV calibration audit;
- explicit statement that sparse MTL and separate disease heads remain;
- no tuning/retry/Test-driven decision.

## Final scope checks

Run:

    git diff --check
    git diff origin/main -- src/audio
    git diff origin/main -- src/video
    git diff origin/main -- src/fusion/data
    git diff origin/main -- src/fusion/loss
    git diff origin/main -- src/common/callbacks
    git status --short
    git diff --stat origin/main...HEAD
    git log -12 --oneline --decorate

Only the authorized R3 model file, configs55/56/57, and PROGRESS may differ.

## Acceptance criteria

TASK-006I passes only if:

- branch exactly `codex/task-006i`;
- default full model is tensor-identical to current origin/main behavior;
- shared-fusion path follows the exact frozen construction;
- trainable params remain exactly `295239`;
- all retained parameter groups remain trainable/active;
- configs differ only run_name/task_aware_fusion;
- pseudo/training paths unchanged;
- firewall pushed before production;
- exactly three runs seeds42/43/44;
- DEV-only selection;
- Test monitoring only;
- frozen claim applied exactly;
- gradient/fusion/calibration diagnostics recorded;
- no tuning/sweep/retry;
- branch pushed;
- main/master untouched.

Passing TASK-006I closes only the Stage-6 task-aware fusion item. Sparse-MTL contribution remains separate and is not authorized automatically.

## Required handoff

Respond in English using exactly:

1. Outcome
2. Changed files
3. Verification
4. Plan status
5. Blockers and risks
6. Next atomic step

Include branch, firewall/final SHA, pushed status, main/master untouched, source scope, backward compatibility, equal parameter count, exactly three runs, per-seed/aggregate metrics, full-minus-ablation deltas, exact claim string, fusion/gradient/controller/calibration summary, explicit sparse-MTL-retained statement, no Test-driven decision, Stage 6 active, Stage-5 closed, Stage7/Text/Final Test locked.

For section 6 write only:

    Manager review of TASK-006I; do not start another task.

Stop.
