# TASK-007C: Separate Frozen-Checkpoint Final Test Evaluation

## Authority and branch

This task follows **MANAGER-DECISION-080**.

Required branch:

    codex/task-007c

Start from current manager-updated `origin/main`.

This is the only authorized Final Test task.

## Frozen pre-Test state

All 15 checkpoints are frozen and verified.

Methods/roles are frozen BEFORE Test:

1. **primary paper candidate:** equal-parameter shared fusion;
2. **secondary multimodal reference:** full R4 trial012;
3. **baseline:** frozen temporal audio.

Seeds for every method:

    42, 43, 44, 45, 46

Do not change these roles based on Test results.

Authoritative DEV dossier:

    docs/STAGE7_FINAL_DEV_EVIDENCE_EN.md

Exact DEV completion:

    STAGE-7 FIVE-SEED DEV CHECKPOINT FREEZE COMPLETE

Frozen five-seed DEV Mean:

- audio: `0.765145`, 95% CI `[0.746230,0.784059]`;
- full R4: `0.790057`, 95% CI `[0.763530,0.816583]`;
- shared fusion: `0.790914`, 95% CI `[0.766258,0.815571]`.

Frozen paired DEV Mean CIs:

- full R4 - audio: `[0.001169,0.048655]`;
- shared - audio: `[0.009556,0.041984]`;
- shared - full R4: `[-0.009704,0.011419]`.

No Test result may cause a model/config/checkpoint/threshold/role change.

## Final Test protocols

Evaluate exactly:

- `test_none`;
- `test_soft`;
- `test_hard`.

Canonical total membership must be:

- TEST_NONE: `1364`;
- TEST_SOFT: `1208`;
- TEST_HARD: `1014`.

Use the canonical per-task observed-label mask. Unknown labels remain ignored.

Primary metrics for every checkpoint/protocol:

- depression UAR;
- depression MF1;
- depression Score = (UAR + MF1)/2;
- Parkinson UAR;
- Parkinson MF1;
- Parkinson Score = (UAR + MF1)/2;
- Mean_Score = (D Score + P Score)/2.

Classification threshold remains frozen:
- binary-logit models: raw logit >= 0;
- historical two-class audio42 path: class argmax, equivalently class1-class0 margin >= 0.

No threshold fitting.
No recalibration.
No Test calibration analysis is required or authorized.

## Allowed tracked files

Only:

- `scripts/common/evaluate_stage7_final_test.py`
- `docs/STAGE7_FINAL_TEST_EVIDENCE_EN.md`
- `docs/PROGRESS_EN.md`

No `src/*` changes.
No config changes.
No cache changes.
No dependency changes.
No checkpoint changes.
No prior evidence-file edits.

## External machine-readable report

Write exactly:

    /media/maxim/Programs/Features/WSM/stage7_final_test/final_test_v1.json

The evaluator MUST refuse overwrite.

The report must contain aggregate metric/count/checkpoint/config/run traceability only.

Do NOT store:
- raw logits;
- raw probabilities;
- per-sample predictions;
- labels;
- raw sample metadata beyond aggregate protocol counts.

Record report SHA256 in the final docs.

## Evaluation-only script contract

Create:

    scripts/common/evaluate_stage7_final_test.py

The script must:

1. encode/read the frozen 15-checkpoint ledger from committed Stage-7 evidence without changing any identity;
2. verify every checkpoint exists and SHA256 matches before evaluation;
3. verify referenced configs exist and remain unchanged;
4. load each model using the exact frozen architecture/config semantics;
5. special-case historical audio42 ONLY through the already accepted `FrozenAudioTemporalAdapter` compatibility path;
6. use current standard audio model for audio seeds43-46;
7. use frozen R4/shared model semantics for fusion checkpoints;
8. use canonical Test protocol membership;
9. compute only the frozen UAR/MF1/Score/Mean metrics;
10. write one atomic JSON report only after all required evaluations succeed;
11. refuse overwrite;
12. expose `--help`;
13. support a DEV-only preflight mode that never constructs/iterates Test datasets when feasible;
14. never train, optimize, fit a threshold, or mutate model state.

No Test result-dependent branch is allowed in the evaluator.

## Mandatory pre-Test firewall

Before iterating ANY Test loader:

1. implement the evaluator;
2. run `--help`;
3. run static/synthetic checks:
   - metric formula;
   - raw-logit threshold at 0;
   - two-class margin/argmax equivalence;
   - sparse observed-mask behavior;
   - Student-t summary arithmetic;
   - overwrite refusal;
4. verify evaluator has exactly 15 frozen checkpoint entries, 3 methods × 5 seeds;
5. verify all 15 checkpoint SHA256 values;
6. verify all config paths;
7. verify pseudo-cache SHA where relevant;
8. verify no `src`, config, cache, dependency diff;
9. perform DEV-only reproduction using the evaluator for ALL 15 checkpoints;
10. require every reproduced D/P/Mean Score to match `docs/STAGE7_FINAL_DEV_EVIDENCE_EN.md` within absolute tolerance `0.0005`;
11. verify audio42 raw-logit/legacy-argmax equivalence as established in TASK-007B;
12. verify no Test dataset/loader iteration occurred during preflight;
13. append firewall evidence to PROGRESS;
14. `git diff --check`;
15. commit and PUSH the firewall.

No Test loader may be iterated before the firewall commit is visible on origin.

If DEV reproduction fails for any checkpoint, STOP. Do not inspect Test and do not fix source/config in TASK-007C.

## Exactly one final Test invocation

After the pushed firewall, run exactly once:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python \
      scripts/common/evaluate_stage7_final_test.py \
      --data-root /media/maxim/Databases/WSM_NEW \
      --audio-feature-cache-root /media/maxim/Databases/WSM_NEW/features \
      --video-cache-root /media/maxim/Programs/Features/WSM/video_depart_v1_fullframe_fallback/cache \
      --pseudo-cache-path /media/maxim/Programs/Features/WSM/ramps_r2_semantic_v1/train_missing_targets.pt \
      --output /media/maxim/Programs/Features/WSM/stage7_final_test/final_test_v1.json

No second invocation for metric improvement.

If the invocation fails after Test evaluation begins:
- do not edit model/config/evaluator logic;
- preserve any failure evidence without exposing partial metrics as final;
- stop and return to manager.

A process/environment retry with the identical committed evaluator is NOT automatically authorized; manager review is required.

## Frozen final statistics

Use the same pre-Test-frozen multiplier:

    2.7764451051977987

For EACH protocol independently (`test_none`, `test_soft`, `test_hard`):

### Per-method five-seed summaries

For D Score, P Score, Mean_Score report:

- five seed values;
- arithmetic mean;
- sample std, ddof=1;
- min;
- max;
- range;
- two-sided 95% Student-t CI:
  `mean ± 2.7764451051977987 * s / sqrt(5)`.

### Paired same-seed comparisons

Compute:

1. full R4 - audio;
2. shared - audio;
3. shared - full R4.

For D/P/Mean report:

- five seed deltas;
- mean delta;
- sample std;
- same 95% Student-t CI.

For Mean additionally report wins/ties/losses.

Do not change the method roles regardless of Test outcomes.

Do not add another inferential test after seeing results.

## DEV-to-Test generalization summary

For each method/protocol report descriptively:

    Test five-seed Mean - frozen DEV five-seed Mean

using the already frozen DEV means.

This is descriptive only.

Do not tune or select from it.

## Required final evidence document

Create:

    docs/STAGE7_FINAL_TEST_EVIDENCE_EN.md

It must include:

- evaluator firewall SHA;
- evaluator script SHA256;
- external report path/SHA256;
- commit/runtime/package identity;
- all 15 checkpoint/config/run identities or an exact pointer to the frozen DEV ledger;
- protocol membership counts;
- per-seed D/P UAR/MF1/Score and Mean for all 15 checkpoints × 3 protocols;
- per-protocol five-seed summary tables;
- per-protocol paired delta/CI tables;
- DEV-to-Test Mean deltas;
- frozen pre-Test role assignment;
- explicit no post-Test revision;
- explicit no training/recalibration/threshold fitting;
- the prior historical Test-line visibility exception, clearly separated from this standardized final evaluation.

Record exactly on success:

    STAGE-7 FINAL TEST EVALUATION COMPLETE

If anything is incomplete:

    STAGE-7 FINAL TEST EVALUATION INCOMPLETE

and state the blocker.

## Interpretation boundaries

Final Test is reporting/generalization evidence, not a new selection set.

Preserve:

- shared fusion remains the preselected primary paper candidate even if another method has a higher Test point estimate;
- full R4 remains the secondary multimodal reference;
- audio remains baseline;
- do not claim shared statistically outperforms full R4 unless a PREDECLARED paired 95% CI for the relevant frozen metric excludes zero;
- do not claim missing-label correctness/comorbidity;
- Stage-6 unsupported mechanism claims remain unsupported;
- T1/T2 Stage-3 conclusions remain unchanged.

## PROGRESS update

Record:

- branch;
- firewall SHA;
- final SHA;
- evaluator script SHA;
- external report SHA;
- exact one Test invocation;
- protocol counts;
- per-method five-seed Test summaries;
- paired summaries;
- DEV-to-Test deltas;
- no role/config/checkpoint/threshold revision;
- exact completion string;
- Stage 7 pending manager closure.

## Final scope checks

Run:

    git diff --check
    git diff origin/main -- src
    git diff origin/main -- configs
    git diff origin/main -- pyproject.toml
    git status --short
    git diff --stat origin/main...HEAD
    git log -12 --oneline --decorate

Only the three authorized files may differ.

## Acceptance criteria

TASK-007C passes only if:

- branch exactly `codex/task-007c`;
- evaluator firewall pushed before ANY Test iteration;
- DEV reproduction all 15 passes tolerance;
- all 15 checkpoint SHAs remain exact;
- exactly one standardized final Test invocation occurs;
- TEST_NONE/SOFT/HARD membership is correct;
- frozen metrics/statistics/pairs are applied exactly;
- no model/config/checkpoint/threshold/role change;
- no training;
- no Test-driven retry/tuning;
- machine-readable report and human-readable report are traceable;
- branch pushed;
- main/master untouched.

Passing TASK-007C does not authorize another experiment. Manager review will determine Stage-7/project closure documentation only.

## Required handoff

Respond in English using exactly:

1. Outcome
2. Changed files
3. Verification
4. Plan status
5. Blockers and risks
6. Next atomic step

Include branch, firewall/final SHA, pushed status, main/master untouched, evaluator/report SHA, exact Test invocation count, protocol counts, five-seed Test means/CIs, paired Mean summaries, DEV-to-Test deltas, frozen role statement, no post-Test revision, exact completion string, Stage 7 pending manager closure.

For section 6 write only:

    Manager review of TASK-007C; do not start another task.

Stop.
