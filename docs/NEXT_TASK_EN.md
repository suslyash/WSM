# TASK-003D: T2 Observable-Description Generator / Source / Prompt Preflight

## Authority and branch

This task follows **MANAGER-DECISION-076**.

Required branch:

    codex/task-003d

Start from current manager-updated `origin/main`.

This is a Stage-3 T2 preflight only.

No production description cache and no T2 training are authorized.

## Why T2 remains in scope

Stage-3 allows at most two text/description families:

- T1: transcript encoder;
- T2: T1 + cached observable description/semantic feature + simple T/D gating.

T1 is complete as a standalone three-seed baseline. T2 is the second and final allowed family, not a hyperparameter/model sweep.

## Frozen description generator

Use exactly:

    Qwen/Qwen3-VL-8B-Instruct

Requested revision:

    1dd1e02d981403da25ed73d43430e4ef598cb94b

Do not try another VLM in this task.

Do not change dependencies.

The task must determine whether the current project environment can load and run this frozen revision. If not, record a blocker and stop; do not silently upgrade packages or switch models.

Record:

- resolved Hugging Face commit;
- config/tokenizer/processor identities;
- safetensors index SHA256;
- every model shard SHA256;
- package versions;
- device;
- dtype;
- peak allocated/reserved GPU memory when applicable;
- wall-clock generation time per audited sample.

## Frozen input source

The semantic description source is **visual-only canonical segment video**.

For every canonical segment row, the candidate source is exactly its existing canonical `segment_path` from `build_manifest`.

Do not use:

- transcript text;
- audio waveform;
- disease labels;
- task labels;
- pseudo labels;
- corpus name;
- split name;
- video ID/segment ID in the prompt;
- prior model predictions.

The VLM may receive only the segment video plus the frozen neutral prompt.

Audit all canonical segment paths structurally and record:

- total rows;
- existing/missing files by split/corpus;
- file extensions;
- zero-byte files;
- decode/open failures under the chosen processor path.

Do not manually inspect TEST video content.

## Frozen prompt

Use exactly this user prompt for every generated description:

    Describe only directly observable visual behavior in this short video clip.
    Focus on facial movement or expression, gaze and head motion, hand or body movement,
    posture, interaction or engagement, and recording/view conditions when they are visible.
    Do not infer or mention any diagnosis, disease, health condition, neurological or psychiatric
    state, medication, cause, identity, age, sex or gender, race or ethnicity, or dataset,
    corpus, task, label, score, or prediction.
    Do not guess unobservable facts. Use concise neutral factual language.
    If an aspect is not visible, omit it.

No system prompt may add disease/task context.

If the model framework requires a generic system role, it must be a neutral non-medical assistant identity and must be recorded verbatim.

## Frozen generation settings

Use deterministic generation:

- `do_sample = false`;
- `max_new_tokens = 120`;
- one output sequence;
- no beam search;
- no temperature/top-p tuning;
- no prompt variants.

Use the model/processor's documented video preprocessing defaults except for any deterministic frame/FPS argument that is strictly required by the current API. If such an argument is required, record and freeze it before sample generation; do not search alternatives.

## Deterministic TRAIN/DEV manual-audit sample

Generate descriptions only for a deterministic sample of at most 24 canonical segment rows.

Strata:

- depression TRAIN;
- depression DEV;
- Parkinson TRAIN;
- Parkinson DEV.

Target 6 rows per stratum.

Selection:

- stable SHA256 ordering of `corpus|split|video_id|segment_file`;
- take the first 6 unique segment rows in each stratum;
- no label/class balancing;
- no performance information.

TEST is excluded completely from generation in TASK-003D.

## External preflight report

Write exactly:

    /media/maxim/Programs/Features/WSM/stage3_t2_preflight/qwen3vl8b_observable_preflight_v1.json

Refuse overwrite.

The external report may contain generated descriptions for the 24 TRAIN/DEV audit items because manual review is required, but no raw description text may be committed to GitHub.

Record report SHA256 in PROGRESS and the human-readable audit doc.

## Manual description audit

For every sampled output, manually record:

- nonempty: yes/no;
- visually grounded / mostly observable: yes/no;
- major unsupported/hallucinated fact: yes/no;
- diagnosis/disease/health-state inference: yes/no;
- demographic/identity inference: yes/no;
- causal/medication inference: yes/no;
- dataset/task/label leakage: yes/no;
- concise enough for semantic encoding: yes/no.

Do not use disease labels to judge whether a description is “correct”.

The reviewer may look at the sampled TRAIN/DEV video clip and generated description only.

Do not inspect Test video content.

## Frozen preflight gate

Record exactly:

    T2 DESCRIPTION GENERATION CONTRACT READY

only if all are true:

1. all canonical segment source paths are deterministically mapped and structural availability is recorded;
2. the frozen Qwen3-VL revision loads and generates under the current environment with no dependency change;
3. all 24 sampled descriptions are nonempty;
4. zero sampled descriptions contain diagnosis/disease/health-state inference;
5. zero sampled descriptions contain demographic/identity inference;
6. zero sampled descriptions contain causal/medication inference;
7. zero sampled descriptions contain dataset/task/label leakage;
8. at least 22/24 are judged visually grounded / mostly observable;
9. at least 22/24 are concise enough for semantic encoding;
10. no TEST content was generated or manually inspected.

Otherwise record exactly:

    T2 DESCRIPTION GENERATION CONTRACT BLOCKED

and name every failed condition.

Do not switch model or prompt in the same task if blocked.

## Reproducibility / deterministic check

Before accepting the sample generation contract:

- choose the lexicographically first sampled item;
- run the exact frozen generation twice from a fresh generation call without changing model/prompt/settings;
- require exact generated-text equality;
- record hashes, not the repeated text, in PROGRESS.

If exact equality fails, record BLOCKED.

## Allowed tracked files

Only:

- `scripts/description/preflight_qwen3vl_observable.py`
- `docs/STAGE3_T2_DESCRIPTION_PREFLIGHT_EN.md`
- `docs/PROGRESS_EN.md`

No `src/*` changes.
No configs.
No dependencies.
No T1 cache changes.
No description production cache.

## Pre-execution firewall

Before loading the full VLM or generating sampled descriptions:

1. implement the preflight script;
2. run `--help`;
3. synthetic/test-mode checks prove:
   - prompt is exactly frozen;
   - no label/task/corpus/split/video/segment metadata enters prompt;
   - sample selection is deterministic TRAIN/DEV only;
   - TEST is excluded from generation;
   - report refuses overwrite;
4. structural source-path audit may run without model generation;
5. verify no source/config/dependency diffs;
6. `git diff --check`;
7. append firewall evidence to PROGRESS;
8. commit and PUSH firewall.

No VLM sample generation before firewall is visible on origin.

## Execute preflight exactly once

After firewall push:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python \
      scripts/description/preflight_qwen3vl_observable.py \
      --data-root /media/maxim/Databases/WSM_NEW \
      --output /media/maxim/Programs/Features/WSM/stage3_t2_preflight/qwen3vl8b_observable_preflight_v1.json

A pure runtime/API defect may be corrected transparently with a pushed corrective firewall before one replacement preflight. No prompt/model/source changes are allowed as a “runtime correction”.

## Human-readable document

Create `docs/STAGE3_T2_DESCRIPTION_PREFLIGHT_EN.md` with:

- exact model/revision/resolved commit;
- prompt and generation settings;
- canonical structural source audit;
- sample IDs only, not labels and not generated text;
- manual audit flags/counts;
- reproducibility result;
- memory/runtime profile;
- exact READY/BLOCKED string;
- explicit no TEST generation/manual inspection;
- explicit no training/model-performance metric.

Do not include generated descriptions verbatim.

## PROGRESS evidence

Record:

- branch;
- firewall/final SHA;
- external report path/SHA;
- model/revision/resolved commit;
- model shard hashes;
- environment/package identity;
- exact prompt hash and generation settings;
- source structural counts;
- deterministic sample IDs;
- manual audit aggregate counts;
- repeat-generation hash equality;
- memory/time profile;
- exact preflight gate string;
- no Test content generation/inspection;
- no performance metrics;
- no training;
- Stage 3 active;
- Stage 7/Final Test locked.

## Final scope checks

Run:

    git diff --check
    git diff origin/main -- src
    git diff origin/main -- configs
    git diff origin/main -- pyproject.toml
    git status --short
    git diff --stat origin/main...HEAD
    git log -10 --oneline --decorate

Only the three authorized tracked files may differ.

## Acceptance criteria

TASK-003D passes only if:

- branch exactly `codex/task-003d`;
- one frozen Qwen3-VL model/revision only;
- one exact prompt only;
- visual canonical segment source only;
- no labels/task/transcript/audio enter generation;
- deterministic TRAIN/DEV sample only;
- Test generation/manual inspection = false;
- external report traceable;
- manual audit gate applied exactly;
- no source/config/dependency/model-training changes;
- branch pushed;
- main/master untouched.

Passing TASK-003D authorizes no full description cache or T2 training automatically.

## Required handoff

Respond in English using exactly:

1. Outcome
2. Changed files
3. Verification
4. Plan status
5. Blockers and risks
6. Next atomic step

Include branch, firewall/final SHA, pushed status, report path/SHA, model/revision/resolved commit, shard hashes, source structural audit, sample count, manual audit gate counts, deterministic-repeat result, memory/runtime profile, exact READY/BLOCKED string, Test generation/manual inspection false, no training/performance metrics, Stage 3 active, Stage 7/Final Test locked.

For section 6 write only:

    Manager review of TASK-003D; do not start another task.

Stop.
