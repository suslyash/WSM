# TASK-006B: Frozen R4 Corpus-Identity Probe on TRAIN→DEV

## Authority and branch

This task follows **MANAGER-DECISION-063**.

Required branch:

    codex/task-006b

Start from current manager-updated `origin/main`.

Create exactly one new task branch from that `origin/main`.

This is a Stage-6 diagnostic/claims audit. It does **not** retrain the main model, does not reopen Stage-5 optimization, and does not authorize another ablation.

## Goal

Execute Stage-6 required item 9: **corpus probe**.

Question:

> How linearly decodable is corpus identity (`depression` vs `parkinson`) from frozen R4 representations on DEV, and does task-conditioned fusion amplify corpus decodability relative to the equal-dimensional projected A+V representation?

This is a domain/confound diagnostic only.

A positive corpus probe does NOT prove that corpus identity causes disease predictions or that corpus identity “replaces disease signal.”

## Required reading

Read in this exact order:

1. `AGENTS.md`
2. `docs/README.md`
3. `docs/PROJECT_REQUIREMENTS.md`
4. `docs/PROGRESS_EN.md`
5. `docs/plan/STAGE_6.md`
6. `docs/NEXT_TASK_EN.md`
7. `src/common/data/wsm_manifest.py`
8. `src/fusion/data/wsm_av_fusion_datamodule.py`
9. `src/fusion/data/wsm_ramps_semantic_datamodule.py`
10. `src/fusion/models/av_r3_disease_query.py`
11. `configs/wsm_mm_pd_dep_v1/fusion/37_r4_ra_stch_optuna_selected_seed42.yaml`
12. `configs/wsm_mm_pd_dep_v1/fusion/38_r4_ra_stch_optuna_confirm_seed43.yaml`
13. `configs/wsm_mm_pd_dep_v1/fusion/39_r4_ra_stch_optuna_confirm_seed44.yaml`

The Stage-5 historical archive may be read only to verify the three frozen checkpoint paths/hashes listed below.

## Frozen R4 checkpoints

Use exactly these selected checkpoints; do not retrain or reselect.

### Seed42

Config:

    configs/wsm_mm_pd_dep_v1/fusion/37_r4_ra_stch_optuna_selected_seed42.yaml

Checkpoint:

    logs/wsm_mm_pd_dep_v1/optuna_r4_ra_stch_base_seed42_2026-09-28_01-54_wsm_av_r3_disease_query_model_r4-ra-stch-optuna-v1-18c6-012_0ce9c7a4/checkpoints/epoch=11_dev_mean_score=0.8191.pt

SHA256:

    104b79e409832839502b4d20a98c054f5efda633930e313ce74f34327e668b2a

Frozen DEV Mean:

    0.8191424538

### Seed43

Config:

    configs/wsm_mm_pd_dep_v1/fusion/38_r4_ra_stch_optuna_confirm_seed43.yaml

Checkpoint:

    logs/wsm_mm_pd_dep_v1/r4_ra_stch_optuna_trial012_confirm_seed43_2026-09-28_12-01_wsm_av_r3_disease_query_model_12038580/checkpoints/epoch=10_dev_mean_score=0.7614.pt

SHA256:

    6af4a4ed21040aff4db2906b0adc8aa5fd27a4b72c375fbb67f68de04ab8446e

Frozen DEV Mean:

    0.7614450000

### Seed44

Config:

    configs/wsm_mm_pd_dep_v1/fusion/39_r4_ra_stch_optuna_confirm_seed44.yaml

Checkpoint:

    logs/wsm_mm_pd_dep_v1/r4_ra_stch_optuna_trial012_confirm_seed44_2026-09-28_12-07_wsm_av_r3_disease_query_model_41bc602b/checkpoints/epoch=11_dev_mean_score=0.7801.pt

SHA256:

    b06fec6912b27cd19d811b72b2c3c08cd1b0a6c826b637f87a314e3976b88c17

Frozen DEV Mean:

    0.7801390000

Before any probe execution, verify all three checkpoint files exist and SHA256 matches exactly.

If any checkpoint is missing or mismatched, stop and report. Do not substitute a different checkpoint.

## Dataset contract

Use canonical TRAIN to fit diagnostic probes and canonical DEV to evaluate them.

Corpus label:

- `depression` -> 0
- `parkinson` -> 1

Derive corpus only from canonical sample metadata `meta["corpus"]`.

Do not derive corpus from observed masks, targets, filenames, path substrings, or disease outputs.

Important structural fact to record in the report: in the canonical programme, corpus identity is coupled to which disease label is observed. Therefore high corpus decodability is a **risk diagnostic**, not causal proof of shortcut use.

No Test loader may be iterated. No TEST_NONE/SOFT/HARD metric may be read or reported in this task.

## Allowed tracked files

Codex may add/modify only:

- `scripts/common/run_r4_corpus_probe.py`
- `docs/PROGRESS_EN.md`

No `src/*` file may change.
No config may change.

Generated probe results are external artifacts and MUST NOT be committed.

## External output

Write one external JSON report:

    /media/maxim/Programs/Features/WSM/stage6_corpus_probe/r4_corpus_probe_v1.json

The script MUST fail rather than overwrite an existing output path.

Record the report SHA256 in PROGRESS.

## Frozen representation set

For each checkpoint, run the frozen model in `eval()` and `torch.no_grad()`.

Extract exactly these representations on TRAIN and DEV:

1. `audio_projected`
   - `ModelOutput.aux["features_audio"]`
   - dimension 160

2. `video_projected`
   - `ModelOutput.aux["features_video"]`
   - dimension 160

3. `projected_av`
   - concatenate `features_audio` and `features_video`
   - dimension 320

4. `task_fused`
   - flatten/concatenate both task vectors from `aux["task_features"]`
   - dimension 320

5. `gates`
   - flatten `aux["task_modality_weights"]`
   - dimension 4

6. `logits`
   - `ModelOutput.preds`
   - dimension 2

The equal dimension of `projected_av` and `task_fused` is intentional for the fusion-amplification diagnostic.

Do not add other representations after seeing results.

## Extraction invariants

For every seed and split:

- model is frozen and in eval mode;
- no gradients;
- TRAIN extraction is deterministic and not shuffled;
- DEV extraction is deterministic and not shuffled;
- one representation row per canonical segment;
- segment IDs are unique;
- corpus labels align with segment IDs;
- TRAIN and DEV segment-ID sets are disjoint;
- representation tensors are finite;
- dimensions exactly match the contract above;
- sample counts equal the DataModule canonical TRAIN/DEV counts;
- no Test loader iteration.

Sort extracted rows by canonical `segment_id` before fitting probes.

## Deterministic linear probe

Implement the probe inside `scripts/common/run_r4_corpus_probe.py`; do not add a dependency.

For each checkpoint and each of the six representations:

1. fit feature standardization on TRAIN only:
   - mean per dimension;
   - population std per dimension;
   - replace std < `1e-8` with 1;
2. apply TRAIN statistics to TRAIN and DEV;
3. convert probe tensors to CPU float64;
4. linear classifier: one affine logit;
5. initialize weight and bias to exactly zero;
6. use class-balanced weighted BCE:
   - each TRAIN class contributes equal total weight;
7. fixed L2 penalty:
   - coefficient `1e-4`;
   - weights only, not bias;
8. optimizer exactly:
   - `torch.optim.LBFGS`;
   - `lr=1.0`;
   - `max_iter=250`;
   - `tolerance_grad=1e-10`;
   - `tolerance_change=1e-12`;
   - `history_size=50`;
   - `line_search_fn="strong_wolfe"`;
9. no hyperparameter search;
10. no early stopping on DEV;
11. DEV is evaluation only.

Record convergence/final TRAIN objective.

## Shuffled-label sanity control

For each checkpoint use one deterministic TRAIN-label permutation:

    generator seed = 9201 + checkpoint_seed

Use `torch.randperm` over the sorted TRAIN corpus-label vector.

The shuffled vector must preserve exact class counts and must not equal the original vector.

Fit the same probe procedure for every representation using shuffled TRAIN labels.

Evaluate against the TRUE DEV corpus labels.

Do not shuffle DEV labels.

This control is diagnostic only and is not used to tune the probe.

## Probe metrics

For true-label and shuffled-label probes record on DEV:

- AUROC;
- balanced accuracy at fixed probability threshold 0.5;
- Brier score.

Implement AUROC locally/deterministically without sklearn.

For every representation report per-seed values and three-seed mean/sample std.

Also report per-seed:

    true AUROC - shuffled-label AUROC

## Gate-distribution audit

Without fitting another model, for each R4 checkpoint and DEV corpus report task-specific audio gate weight:

- mean;
- sample std;
- q25;
- median;
- q75.

For each task report the difference:

    mean_audio_weight(depression_corpus) - mean_audio_weight(parkinson_corpus)

Do this for depression query and Parkinson query separately.

This is descriptive only.

## Frozen interpretation rules

### A. Strong corpus-decodability flag

Record exactly:

    CORPUS IDENTITY IS STRONGLY LINEARLY DECODEABLE FROM R4 FUSED REPRESENTATIONS

only if BOTH conditions hold for `task_fused`:

1. true-label DEV corpus AUROC >= 0.90 on at least 2 of 3 R4 seeds;
2. `true AUROC - shuffled-label AUROC >= 0.20` on at least 2 of 3 R4 seeds.

Otherwise record exactly:

    STRONG CORPUS-ID DECODABILITY FLAG NOT TRIGGERED

### B. Fusion-amplification flag

Compute same-seed:

    task_fused AUROC - projected_av AUROC

Record exactly:

    R4 TASK-CONDITIONED FUSION AMPLIFIES LINEAR CORPUS DECODABILITY

only if BOTH:

1. the delta is >= +0.05 on at least 2 of 3 seeds;
2. the three-seed mean delta is >= +0.05.

Otherwise record exactly:

    FUSION AMPLIFICATION OF CORPUS DECODABILITY NOT SUPPORTED

### Interpretation boundary

Even if either flag triggers:

- do not say corpus identity causes the disease predictions;
- do not say corpus identity replaces disease signal;
- do not demote/promote R4;
- do not alter configs;
- do not propose domain mitigation as an implemented change;
- do not use Test;
- do not claim significance from three seeds.

Manager will decide whether a later domain ablation is required.

## Mandatory pre-execution firewall commit

Before running the full three-checkpoint probe:

1. implement the script;
2. verify the three checkpoint paths and SHA256;
3. instantiate each config/DataModule/model;
4. load the correct checkpoint into the correct model;
5. run a tiny TRAIN+DEV extraction smoke for each seed;
6. prove all six representations exist, are finite, and have exact dimensions;
7. prove corpus labels come from `sample_meta[*]["corpus"]`;
8. prove no Test loader was iterated;
9. run a tiny synthetic unit check of the local AUROC implementation:
   - perfect ordering -> AUROC 1.0;
   - reversed ordering -> AUROC 0.0;
   - tied constant scores -> AUROC 0.5;
10. run a tiny synthetic probe optimization sanity check;
11. `git diff --check`;
12. `git diff origin/main -- src` must be empty;
13. append firewall evidence to PROGRESS;
14. commit and push one firewall commit.

Do not run the full probe before the firewall commit exists on origin.

## Exact execution

After the firewall commit, run exactly once:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python       scripts/common/run_r4_corpus_probe.py       --output /media/maxim/Programs/Features/WSM/stage6_corpus_probe/r4_corpus_probe_v1.json

No second run for metric improvement.

If execution fails before producing a valid output, repair only a deterministic implementation/runtime defect, document it, and do not change the frozen probe method.

## Required PROGRESS evidence

Record:

- task/branch;
- script path;
- exact three config/checkpoint paths and checkpoint SHA256;
- firewall commit SHA;
- checkpoint-load verification;
- TRAIN/DEV counts and corpus counts;
- representation dimensions;
- no-Test proof;
- probe method and frozen hyperparameters;
- external JSON path/SHA256;
- per-seed/per-representation true and shuffled metrics;
- three-seed mean/std metrics;
- task_fused true-minus-shuffled AUROC deltas;
- task_fused-minus-projected_av AUROC deltas;
- both exact frozen interpretation strings;
- gate distributions/deltas by corpus;
- explicit statement that corpus identity is structurally coupled to observed-task identity;
- explicit no causal shortcut claim;
- no main-model training;
- no source/config changes;
- no Test use.

## Final scope checks

Run:

    git diff --check
    git diff origin/main -- src
    git status --short
    git diff --stat origin/main...HEAD
    git log -12 --oneline --decorate

Only the new probe script and PROGRESS may differ from origin/main.

## Acceptance criteria

TASK-006B passes only if:

- branch exactly `codex/task-006b` from current manager main;
- only script + PROGRESS change;
- all three frozen R4 checkpoints are verified by SHA;
- no R4/main-model retraining occurs;
- TRAIN-only fit / DEV-only evaluation for probes;
- no Test loader iteration;
- all six frozen representations are audited;
- deterministic true-label and shuffled-label probes complete for all seeds/representations;
- AUROC implementation sanity checks pass;
- probe method is not tuned;
- gate distributions are reported;
- interpretation strings are applied exactly;
- report is external and hashed;
- branch pushed;
- main/master untouched.

Passing TASK-006B closes Stage-6 corpus-probe item 9 only. It authorizes no next task.

## Required handoff

Respond in English using exactly:

1. Outcome
2. Changed files
3. Verification
4. Plan status
5. Blockers and risks
6. Next atomic step

Explicitly include:

- branch `codex/task-006b`;
- firewall SHA;
- final evidence SHA;
- pushed status;
- main/master untouched;
- source/config diff empty;
- checkpoint paths/SHA;
- output JSON path/SHA;
- no main-model training;
- no Test loader iteration;
- per-representation three-seed AUROC summary;
- task_fused per-seed AUROC and shuffled-control deltas;
- task_fused-minus-projected_av deltas;
- both exact frozen interpretation strings;
- gate-distribution summary;
- no causal shortcut claim;
- Stage 6 active;
- Stage-5 optimization closed;
- no Stage7/Text/Final Test.

For section 6 write only:

    Manager review of TASK-006B; do not start another task.

Stop.
