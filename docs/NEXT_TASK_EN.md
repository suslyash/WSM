# TASK-003E: Stage-3 Text/Description Synthesis and Closure Dossier

## Authority and branch

This task follows **MANAGER-DECISION-077**.

Required branch:

    codex/task-003e

Start from current manager-updated `origin/main`.

This task is **documentation-only**.

No training, inference, generation, cache build, model download, probing, Test analysis, source/config/script change, or hyperparameter/model selection is authorized.

## Goal

Create one authoritative Stage-3 dossier that answers:

1. What transcript/data contract was established?
2. What did T1 demonstrate as a standalone text baseline?
3. Why was T2 stopped before implementation/training?
4. Has the Stage-3 two-family budget been completely and honestly exercised?
5. What Stage-3 claims are supported, unsupported, blocked, or diagnostic-only?
6. Which frozen methods remain relevant for a separate Stage-7 candidate-freeze decision?

Do NOT select the final Stage-7 methods yourself.

## Required reading

Read:

1. `AGENTS.md`
2. `docs/README.md`
3. `docs/PROJECT_REQUIREMENTS.md`
4. `docs/PROGRESS_EN.md`
5. `docs/plan/STAGE_3.md`
6. `docs/plan/STAGE_7.md`
7. `docs/STAGE3_TRANSCRIPT_AUDIT_EN.md`
8. `docs/STAGE3_T2_DESCRIPTION_PREFLIGHT_EN.md`
9. `docs/STAGE6_CLAIM_LEDGER_EN.md`
10. `docs/NEXT_TASK_EN.md`

Use existing committed evidence only.

## Allowed tracked files

Only:

- `docs/STAGE3_CLAIM_LEDGER_EN.md`
- `docs/PROGRESS_EN.md`

No other tracked file may change.

## Mandatory Stage-3 evidence ledger

Create `docs/STAGE3_CLAIM_LEDGER_EN.md`.

### A. Transcript contract

Record exact accepted TASK-003A conclusions:

    SEGMENT TEXT ALIGNMENT NOT ESTABLISHED
    T1 TEXT UNIT SHOULD BE VIDEO-LEVEL TRANSCRIPT
    T1 TRANSCRIPT DATA CONTRACT READY

Authoritative report SHA256:

    4254d81281522015d0633d6534e1616e925a2bf0130f67a37b541a2aacc2ed78

Key facts:

- 8,622 canonical segments;
- 755 unique corpus/split/video groups;
- 754 available plain-text transcript files;
- 1 missing TRAIN transcript;
- no decode errors;
- no cross-split exact transcript-hash duplicates;
- three within-TRAIN exact duplicate hashes;
- no explicit source timing metadata;
- no parseable transcript timestamps;
- TEST lexical/language audit was disabled in the corrected authoritative report.

Boundary:
there is no scientifically established segment-level transcript alignment. T1 uses video-level transcript units; segment broadcast is evaluation-only.

### B. T1 frozen family identity

Record:

- encoder `FacebookAI/xlm-roberta-base`;
- requested revision `e73636d`;
- resolved commit `e73636d4f797dec63c3081bb6ed5c7b0bb3f2089`;
- model weight SHA256 `6fd4797bc397c3b8b55d6bb5740366b57e6a3ce91c04c77f22aafc0c128e6feb`;
- cache index SHA256 `4f276b60e8e423a1eab2d69a02782daa274fb183ae028a525f16df3adcf83ea8`;
- 553 unique TRAIN video-level units: D287/P266;
- downstream trainable parameter count `447746`;
- one frozen family only; no encoder search.

### C. T1 evidence

Exact completion:

    T1 THREE-SEED TEXT BASELINE COMPLETE

Per-seed primary DEV D/P/Mean:

- seed42: `0.721153/0.899381/0.810267`;
- seed43: `0.562911/0.825768/0.694339`;
- seed44: `0.596355/0.809025/0.702690`.

Three-seed mean:

- D `0.6268063333`;
- P `0.8447246667`;
- Mean `0.7357653333`.

Sample std:

- D `0.0834002123`;
- P `0.0480683689`;
- Mean `0.0646553057`.

Range:

- D `0.158242`;
- P `0.090356`;
- Mean `0.115928`.

Contextual deltas:

versus matched audio:
- D `-0.1035314632`;
- P `+0.0284285455`;
- Mean `-0.0375516255`.

versus full R4:
- D `-0.1084500000`;
- P `+0.0061640000`;
- Mean `-0.0511434846`.

Interpretation boundary:

- T1 is a standalone text-stream baseline;
- it does NOT show that text improves A+V;
- P is strong relative to audio/R4 context while D is materially weaker and seed-sensitive;
- no three-seed significance claim;
- unique-video DEV diagnostics were secondary and lower than primary segment-weighted metrics on all seeds;
- Test monitoring did not drive selection.

### D. T2 observable-description preflight

Exact result:

    T2 DESCRIPTION GENERATION CONTRACT BLOCKED

Frozen generator:

- `Qwen/Qwen3-VL-8B-Instruct`;
- revision `1dd1e02d981403da25ed73d43430e4ef598cb94b`;
- prompt SHA256 `118b93ba090af2b4ce755d9da1476e073cab0624760e9aa4c347faad8750ee4d`;
- deterministic uniform `num_frames=8` after documented CUDA-memory correction;
- report SHA256 `767d687e2f857938fa882555250d7a4be4159556e49537377967d82c36407930`.

Manual audit:

- nonempty `24/24`;
- visually grounded/mostly observable `24/24`;
- major unsupported/hallucinated `0/24`;
- diagnosis/disease/health-state inference `0/24`;
- demographic/identity inference `4/24`;
- causal/medication inference `0/24`;
- dataset/task/label leakage `0/24`;
- concise enough `18/24`.

Failed frozen conditions:
- zero demographic/identity inference;
- at least 22/24 concise.

Boundary:

- T2 did not reach cache construction, model implementation, training, DEV performance, or Test generation;
- therefore there is no T2 performance claim;
- no prompt/model remediation is authorized inside Stage 3 because that would exceed the frozen second-family preflight/search contract.

## Stage-3 completeness matrix

Map the active Stage-3 plan and PROJECT_REQUIREMENTS to evidence.

At minimum verify:

1. transcript coverage/encoding/language/leakage audit — TASK-003A;
2. no invented segment alignment — TASK-003A/T1 contract;
3. T1 pretrained transcript family implemented — TASK-003B;
4. T1 three-seed standalone evidence — TASK-003C;
5. second allowed family T2 attempted as transcript+observable-description concept — TASK-003D;
6. description prompt/model revision fingerprinted — TASK-003D;
7. diagnosis-free prompt/manual audit — TASK-003D;
8. no broad text/description search — family/model/prompt budgets respected;
9. no Test-driven text/description selection;
10. description sanity check occurred but failed its frozen gate, so implementation correctly stopped.

If all required Stage-3 work is represented, record exactly:

    STAGE-3 TEXT/DESCRIPTION EVIDENCE MATRIX COMPLETE

Otherwise record:

    STAGE-3 TEXT/DESCRIPTION EVIDENCE MATRIX INCOMPLETE

and name the exact missing item. Do not propose an experiment.

## Family-budget conclusion

Record exactly:

    STAGE-3 TWO-FAMILY BUDGET EXHAUSTED

if:

- T1 is the sole implemented pretrained transcript family;
- T2 is the sole second-family observable-description attempt;
- no third encoder/generator/prompt-remediation family was tried.

Clarify that a blocked T2 preflight is a valid bounded-study outcome and does not require forcing T2 training.

## Pre-Stage-7 candidate inventory

Create a descriptive inventory, not a ranking or final selection.

Include at least:

### Frozen temporal audio
- D/P/Mean `0.7303377965/0.8162961212/0.7733169588`;
- Mean std `0.0138512531`;
- strong stability reference.

### Full R4 trial012
- D/P/Mean `0.7352563333/0.8385606667/0.7869088179`;
- Mean std `0.0294384420`;
- beats matched audio Mean on 3/3 seeds;
- multiple Stage-6 component claims are unsupported, so complexity necessity is not established.

### Equal-parameter shared-fusion Stage-6 comparator
- D/P/Mean `0.7407873333/0.8467100000/0.7935663333`;
- full-minus-shared Mean `-0.0066575154`;
- same `295239` trainable parameters as full;
- this is a relevant simplified single-model candidate for later manager freeze, but TASK-003E does not select it.

### No-semantic-depression comparator
- D/P/Mean `0.714447/0.868369/0.791408`;
- full-minus-ablation D/P/Mean `+0.020809/-0.029808/-0.004499`;
- higher aggregate Mean than full but material D/P trade-off;
- relevant contextual candidate, not automatically final.

### Paired task-isolated comparator
- D/P/paired Mean `0.717846/0.881495/0.799671`;
- two separately trained models;
- NOT equal total deployment size;
- diagnostic/upper-control candidate, not a like-for-like single-model finalist.

### T1 text-only
- D/P/Mean `0.6268063333/0.8447246667/0.7357653333`;
- standalone contextual baseline;
- not evidence that text should be added to A+V.

### T2
- blocked before implementation/training;
- ineligible for Stage-7 performance comparison under current evidence.

Do not label a winner.
Do not choose the Stage-7 set.
Do not assign ranks/scores/tiers.

## Stage boundary

The dossier must state:

- Stage 3 remains ACTIVE until manager accepts TASK-003E;
- Stage 7 remains LOCKED;
- Final Test remains unauthorized;
- no further Stage-3 text/description family is justified under the frozen programme budget;
- after manager acceptance, the next programme-level action should be a separate Stage-7 candidate/config/seed freeze task, not immediate training.

Do not edit Stage-3 or Stage-7 plan files.

## PROGRESS update

Append a compact TASK-003E evidence section with:

- branch;
- ledger path;
- exact Stage-3 completeness string;
- exact two-family budget string;
- T1 exact completion and aggregate evidence;
- T2 exact blocked result and failed gate counts;
- pre-Stage-7 candidate inventory boundary;
- no run/source/config/cache/Test analysis;
- Stage 3 pending manager closure;
- Stage 7/Final Test locked.

Do not mark Stage 3 COMPLETE yourself.
Do not activate Stage 7 yourself.

## Final scope checks

Run:

    git diff --check
    git status --short
    git diff --stat origin/main...HEAD
    git log -10 --oneline --decorate

Only `docs/STAGE3_CLAIM_LEDGER_EN.md` and `docs/PROGRESS_EN.md` may differ.

## Acceptance criteria

TASK-003E passes only if:

- branch exactly `codex/task-003e`;
- documentation-only scope;
- all Stage-3 values and hashes match accepted evidence;
- T1 and T2 interpretation boundaries are preserved;
- completeness matrix is explicit;
- family-budget exhaustion is explicit;
- candidate inventory is descriptive only;
- no final method selection/ranking;
- no Stage-7 activation;
- branch pushed;
- main/master untouched.

Passing TASK-003E authorizes no Stage-7 run automatically.

## Required handoff

Respond in English using exactly:

1. Outcome
2. Changed files
3. Verification
4. Plan status
5. Blockers and risks
6. Next atomic step

Include branch, final commit SHA, pushed status, main/master untouched, ledger path, exact Stage-3 completeness string, exact family-budget string, T1/T2 status summary, candidate-inventory summary, no compute/source/config/cache/Test analysis, Stage 3 pending manager closure, Stage 7/Final Test locked.

For section 6 write only:

    Manager review of TASK-003E; do not start another task.

Stop.
