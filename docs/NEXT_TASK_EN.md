# TASK-006K: Stage-6 Evidence Synthesis and Claim-Freeze Dossier

## Authority and branch

This task follows **MANAGER-DECISION-072**.

Required branch:

    codex/task-006k

Start from current manager-updated `origin/main`.

Create exactly one new task branch from that main.

This task is **documentation-only**.

No training, probing, model execution, cache build, metric recomputation from Test, source edit, config edit, or model-selection experiment is authorized.

## Goal

Produce one authoritative, compact, traceable Stage-6 dossier that answers:

1. Which Stage-6 claims are supported?
2. Which are not supported?
3. Which findings are diagnostic-only?
4. Has every required Stage-6 plan item received its required evidence?
5. Are there any genuine remaining core A+V evidence gaps?
6. What do the results imply for candidate complexity and scientific claims?
7. What remains required before Stage 7 can be unlocked?

Do NOT promote or select a final method. Manager retains that decision.

## Required reading

Read in this order:

1. `AGENTS.md`
2. `docs/README.md`
3. `docs/PROJECT_REQUIREMENTS.md`
4. `docs/PROGRESS_EN.md`
5. `docs/plan/STAGE_6.md`
6. `docs/plan/STAGE_3.md`
7. `docs/plan/STAGE_7.md`
8. `docs/NEXT_TASK_EN.md`

Read source/config files only when needed to resolve an evidence identity already recorded in PROGRESS. Do not execute training or evaluation.

The Stage-5 archive may be read only when needed to verify frozen references already cited by Stage-6 evidence.

## Allowed tracked files

Only:

- `docs/STAGE6_CLAIM_LEDGER_EN.md`
- `docs/PROGRESS_EN.md`

No other tracked file may change.

## Mandatory evidence policy

GitHub-committed Stage-6 PROGRESS evidence is authoritative.

Do not invent missing metrics.
Do not infer significance.
Do not reinterpret Test monitoring.
Do not reopen failed gates.
Do not convert “NOT SUPPORTED” into “harmful”, “useless”, or “proven unnecessary”.

Use exact frozen claim strings where they exist.

## Required claim ledger

Create `docs/STAGE6_CLAIM_LEDGER_EN.md`.

It must contain a table with at least:

- Stage-6 plan item;
- accepted task / PR;
- comparator/control;
- seed count;
- key DEV evidence;
- exact frozen claim/result;
- status: SUPPORTED / NOT SUPPORTED / DIAGNOSTIC ONLY;
- interpretation boundary.

At minimum include all of the following accepted results.

### Sparse MTL

Exact result:

    SPARSE MTL JOINT-TRAINING CONTRIBUTION NOT SUPPORTED

Accepted paired task-isolated three-seed:
- D `0.717846`
- P `0.881495`
- Mean `0.799671`

Full-minus-isolated:
- D `+0.017410`
- P `-0.042935`
- Mean `-0.012762`

Record TASK-006J procedural exception:
- total invocations = 7;
- first D42 failed in TRAIN epoch1 before DEV/Test evaluation;
- corrective firewall preceded the accepted six completed runs;
- accepted completed run count = 6;
- no metric-driven retry/tuning.

### Task-aware fusion

Exact result:

    TASK-AWARE FUSION CONTRIBUTION NOT SUPPORTED

Shared equal-parameter three-seed:
- D `0.7407873333`
- P `0.8467100000`
- Mean `0.7935663333`

Full-minus-shared:
- D `-0.0055310000`
- P `-0.0081493333`
- Mean `-0.0066575154`

Both variants: `295239` trainable parameters.

### Direct pseudo supervision

Exact result:

    DIRECT PSEUDO-SUPERVISION CONTRIBUTION SUPPORTED

No-direct-pseudo three-seed:
- D `0.7134376667`
- P `0.8263193333`
- Mean `0.7698780000`

Full-minus-ablation:
- D `+0.0218186667`
- P `+0.0122413333`
- Mean `+0.0170308179`

Boundary: direct pseudo BCE term only.

### Uncertainty / reliability

Exact result:

    UNCERTAINTY/RELIABILITY CONTRIBUTION SUPPORTED

Uniform-reliability three-seed:
- D `0.7352583333`
- P `0.8345853333`
- Mean `0.7849220000`

Full-minus-ablation:
- D `-0.0000020000`
- P `+0.0039753333`
- Mean `+0.0019868179`

Boundary: small effect; combined graded reliability in pseudo BCE weighting + RA-controller reliability signal, not separately identified.

### Semantic evidence

Exact result:

    SEMANTIC-EVIDENCE CONTRIBUTION VIA DEPRESSION PSEUDO ACCEPTANCE NOT SUPPORTED

No-semantic-depression three-seed:
- D `0.714447`
- P `0.868369`
- Mean `0.791408`

Full-minus-ablation:
- D `+0.020809`
- P `-0.029808`
- Mean `-0.004499`

Boundary: tests the semantic-enabled depression pseudo-acceptance path only.

### RA-STCH balancing

Record both exact results:

    RA-STCH BALANCING CONTRIBUTION OVER EQUAL NOT SUPPORTED

    RA-STCH ADVANTAGE OVER SIMPLER BALANCING NOT SUPPORTED

Fixed-composition RA three-seed:
- D `0.6931503333`
- P `0.8519640000`
- Mean `0.7725573333`

Equal Mean `0.7791726667`.
Corrected Progress Mean `0.7847510000`.
RA minus best-simple Mean by seed:
- `-0.004654`
- `-0.010530`
- `-0.024654`

Boundary: this isolates balancing/controller contribution; it does not automatically demote the separately optimized trial-012 composition.

### Modality removal

Video exact result:

    VIDEO MODALITY CONTRIBUTION NOT SUPPORTED

No-video three-seed:
- D `0.749645`
- P `0.783216`
- Mean `0.766431`

Full-minus-no-video:
- D `-0.014389`
- P `+0.055344`
- Mean `+0.020478`

Boundary: frozen claim fails due D non-regression criterion; do not say video is universally useless.

Online audio exact result:

    ONLINE AUDIO INPUT MODALITY CONTRIBUTION SUPPORTED

No-audio three-seed:
- D `0.653375`
- P `0.792030`
- Mean `0.722703`

Full-minus-no-audio:
- D `+0.081881`
- P `+0.046531`
- Mean `+0.064206`

Boundary: online student audio branch only; frozen pseudo cache still contains historical audio-derived teacher information.

### Shuffled/mismatched pseudo negative control

Exact accepted result:

    SAMPLE-SPECIFIC PSEUDO ALIGNMENT CLAIM SUPPORTED BY THIS NEGATIVE CONTROL

Matched Equal Mean `0.7791726667`.
Exact shuffled FIX2 Mean `0.7371180000`.
Matched-minus-shuffled Mean `+0.0420546667`.
Matched > shuffled on 3/3 seeds.

Boundary: no pseudo-label correctness, missing-label recovery, comorbidity, or significance claim.

### Corpus probe

Record exact flags:

    STRONG CORPUS-ID DECODABILITY FLAG NOT TRIGGERED

    FUSION AMPLIFICATION OF CORPUS DECODABILITY NOT SUPPORTED

Three-seed true AUROC means:
- audio `0.791292`
- video `0.705505`
- projected A+V `0.750650`
- task-fused `0.759832`
- gates `0.551605`
- logits `0.575197`

Task-fused minus projected-A+V AUROC deltas:
- `+0.024588`
- `-0.000165`
- `+0.003123`
- mean `+0.009182`

Boundary: corpus identity is structurally coupled to observed-task identity; no causal shortcut claim.

## Stage-6 completeness matrix

Map every current `docs/plan/STAGE_6.md` item 1–10 to accepted evidence.

Required conclusions to check, not assume:

1. sparse MTL — TASK-006J;
2. task-aware fusion — TASK-006I;
3. direct pseudo — TASK-006D;
4. uncertainty/reliability — TASK-006E;
5. semantic evidence — TASK-006F;
6. RA-STCH — TASK-006C;
7. remove each modality — TASK-006G/H;
8. shuffled/mismatched pseudo negative control — TASK-006A/FIX2;
9. corpus probe — TASK-006B;
10. equal-parameter control if size grows materially — TASK-006I provides an explicit equal-parameter architectural control; additionally state whether the conditional requirement is otherwise triggered.

Also assess required diagnostics:
- DEV UAR/MF1/Score;
- ECE/Brier;
- pseudo coverage/class balance;
- task gradient norms/cosines;
- negative-transfer deltas;
- gate distributions.

If all are represented by accepted evidence, record exactly:

    CORE STAGE-6 A+V EVIDENCE MATRIX COMPLETE

Otherwise record exactly:

    CORE STAGE-6 A+V EVIDENCE MATRIX INCOMPLETE

and name the specific missing item. Do not create a new experiment.

## Candidate implications section

This section must be descriptive, not a promotion decision.

Record the frozen full trial-012 reference:
- D `0.7352563333`
- P `0.8385606667`
- Mean `0.7869088179`
- Mean std `0.0294384420`.

Record matched temporal audio:
- Mean `0.7733169588`
- Mean std `0.0138512531`.

Record that full R4 > matched audio Mean on 3/3 seeds but audio is more seed-stable.

Also record materially relevant Stage-6 comparator facts:
- shared-fusion equal-parameter Mean `0.7935663333`, above full on 3/3 seeds;
- no-semantic-depression Mean `0.791408`, above full aggregate but with a D/P trade-off;
- paired task-isolated Mean `0.799671`, but it is a two-model comparator and not equal total deployment size;
- no-video Mean `0.766431`, below full aggregate but with higher D;
- no-audio Mean `0.722703`, substantially below full.

Required interpretation:
- Stage-6 evidence does **not** justify claiming that every component of full trial-012 is necessary;
- multiple complexity/mechanism claims are not supported;
- direct pseudo supervision, graded reliability, sample-specific pseudo alignment, and the online audio branch have positive support under their exact gates;
- no final method promotion/demotion or Stage-7 candidate selection is made in TASK-006K.

## Deferred Stage-3 / Stage-7 boundary

PROJECT_REQUIREMENTS requires deferred Text/Description to be completed before paper-ready freeze.

Therefore the dossier must state:

- even if core Stage-6 A+V evidence is complete, Stage 7 remains LOCKED;
- Final Test remains unauthorized;
- the next programme-level phase after manager acceptance should be the deferred Stage-3 Text/Description study, unless the manager identifies a genuine Stage-6 documentation/evidence gap;
- no new A+V tuning should be proposed.

Do not edit Stage-3 or Stage-7 plans in this task.

## PROGRESS update

Append a compact TASK-006K synthesis entry that includes:

- branch;
- claim-ledger path;
- exact completeness string;
- supported claim list;
- not-supported claim list;
- diagnostic-only findings;
- TASK-006J procedural exception;
- candidate-implication boundaries;
- Stage-3 requirement;
- Stage-7 lock;
- statement that no run/source/config/Test decision occurred.

Do not mark Stage 6 COMPLETE yourself.
Do not activate Stage 3 yourself.
Manager will do that after review.

## Final scope checks

Run:

    git diff --check
    git status --short
    git diff --stat origin/main...HEAD
    git log -12 --oneline --decorate

Only `docs/STAGE6_CLAIM_LEDGER_EN.md` and `docs/PROGRESS_EN.md` may differ.

## Acceptance criteria

TASK-006K passes only if:

- branch exactly `codex/task-006k`;
- no source/config/script changes;
- no training/probe/cache generation;
- no Test-based analysis beyond restating already accepted monitoring boundaries;
- all Stage-6 claim strings and key values match accepted evidence;
- TASK-006J exception is represented accurately as 7 total invocations / 6 accepted completed runs;
- completeness matrix covers all Stage-6 items and required diagnostics;
- candidate implications are descriptive only;
- Stage 3 remains deferred pending manager activation;
- Stage 7/Final Test remain locked;
- branch pushed;
- main/master untouched.

Passing TASK-006K authorizes no next task automatically.

## Required handoff

Respond in English using exactly:

1. Outcome
2. Changed files
3. Verification
4. Plan status
5. Blockers and risks
6. Next atomic step

Include:
- branch;
- final commit SHA;
- pushed status;
- main/master untouched;
- claim-ledger path;
- exact completeness string;
- supported/not-supported/diagnostic-only counts and lists;
- TASK-006J procedural exception;
- no run/source/config changes;
- Stage 6 still pending manager closure;
- Stage 3 still deferred pending manager activation;
- Stage 7 and Final Test locked.

For section 6 write only:

    Manager review of TASK-006K; do not start another task.

Stop.
