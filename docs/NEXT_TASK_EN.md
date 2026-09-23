# TASK-001B: Build the Canonical Partial-Label Segment Manifest

## Role

You are the implementing Codex. Execute only this task, update docs/PROGRESS_EN.md, commit the implementation on the required task branch, push that branch to origin, then stop. Follow AGENTS.md.

The required branch is:

    codex/task-001b

Do not implement a datamodule, masked loss, text alignment, description extraction, or any later Stage 1 task in this cycle.

## Goal

Implement the reproducible canonical Stage 1 segment-manifest builder using the source contract established by TASK-001A.

The manifest must represent the two disease targets honestly:

- observed disease label = raw 0/1 target;
- unobserved disease label = null/unknown, never 0;
- observed_depression / observed_parkinson explicitly define supervision;
- no pseudo-labels.

This task creates the canonical manifest contract and builder only. Stage 1 remains partial because authoritative speaker identity is unresolved and no segment-level text alignment exists.

## Manager Decisions for Unresolved TASK-001A Findings

These decisions are authoritative for this task:

1. speaker_id:
   - keep the required speaker_id column;
   - set it to null for every row because no authoritative source was found;
   - never substitute video_id or infer speaker identity;
   - record speaker_independence_verified=false in the audit;
   - do not claim the Stage 1 split gate is complete.

2. text_available:
   - the video-level .txt source discovered in TASK-001A is not sufficient evidence of segment-level text availability;
   - set text_available=false for every canonical segment until a later task establishes segment alignment;
   - retain video-level transcript coverage only in the audit metadata, not as true segment availability.

3. description_available:
   - set description_available=false for every row until an authoritative generated/cached description source exists.

4. audio_available and video_available:
   - derive them only from the file-existence rules audited in TASK-001A.

5. split:
   - preserve the existing deterministic video-level priority rule test > dev > train;
   - do not create a new split;
   - do not use Test labels, predictions, or metrics for any model/selection decision.

## Required Reading

Read before editing:

1. AGENTS.md;
2. docs/PROJECT_REQUIREMENTS.md, especially Sections 2, 9, 10, 11, and 13;
3. Stage 1 in docs/PLAN.md;
4. docs/PROGRESS_EN.md, especially TASK-001A;
5. docs/NEXT_TASK_EN.md;
6. scripts/common/audit_manifest_sources.py;
7. src/common/utils/segment_index.py.

## Allowed Files

- src/common/data/__init__.py;
- src/common/data/wsm_manifest.py;
- scripts/common/build_wsm_manifest.py;
- docs/PROGRESS_EN.md.

Creating src/common/data is allowed if it does not exist.

No other tracked file may be modified.

Generated manifest/audit outputs are verification artifacts and MUST NOT be committed.

## Forbidden Actions

- any change under src/audio;
- modifying src/common/utils/segment_index.py;
- modifying any existing audio datamodule;
- creating a new Chimera datamodule in this task;
- changing labels, diagnosis values, BAD_SEGMENTS, or split priority;
- replacing unknown disease labels with 0 or any numeric sentinel that could be interpreted as negative;
- inferring speaker_id;
- marking text_available=true from video-level transcript existence alone;
- creating descriptions or semantic features;
- pseudo-labeling;
- training, tuning, checkpoint selection, threshold selection, or Test evaluation;
- inspecting Test predictions or Test performance metrics;
- video/text/description/fusion model development;
- dependency installation;
- broad refactors;
- editing docs/NEXT_TASK_EN.md;
- pushing implementation directly to main/master;
- opening or merging a PR.

## Implementation Requirements

### 1. Reusable manifest builder

Implement src/common/data/wsm_manifest.py with a deterministic reusable builder for the canonical segment manifest.

The canonical output columns, in this exact order, are:

    segment_id
    video_id
    speaker_id
    corpus
    split
    y_depression
    y_parkinson
    observed_depression
    observed_parkinson
    audio_available
    video_available
    text_available
    description_available

Requirements:

- use the same raw source CSVs, delimiters, BAD_SEGMENTS exclusions, and existing test > dev > train video-level priority semantics audited in TASK-001A;
- fail on missing/malformed required metadata instead of silently repairing it;
- corpus values are exactly depression or parkinson;
- split values are exactly train, dev, or test;
- segment_id is deterministic and collision-free from the audited composite corpus + video_id + segment_file; use one documented stable string encoding and assert uniqueness;
- speaker_id must be a null value for every row in this task;
- depression rows:
  - y_depression = raw diagnosis 0/1;
  - y_parkinson = null;
  - observed_depression = true;
  - observed_parkinson = false;
- Parkinson rows:
  - y_depression = null;
  - y_parkinson = raw diagnosis 0/1;
  - observed_depression = false;
  - observed_parkinson = true;
- audio_available is true only when the audited segment WAV exists and is non-empty;
- video_available is true only when the audited segment video exists and is non-empty;
- text_available is false for every row in this task;
- description_available is false for every row in this task.

Use nullable data representation that preserves unknown disease labels through CSV round-trip without converting them to 0.

### 2. Manifest audit

The reusable module must also produce a machine-readable audit containing at least:

- schema/version identifier;
- row counts per corpus/split;
- observed positive/negative counts per disease and split;
- unknown count per disease and split;
- null speaker_id count;
- speaker_independence_verified=false with the TASK-001A reason;
- segment_id uniqueness/collision count;
- video_id pairwise split-overlap counts after the existing split rule;
- audio/video availability counts;
- text_available count = 0 plus the TASK-001A video-level transcript source coverage 8549/8622 as source-only evidence;
- description_available count = 0;
- canonical manifest SHA-256 fingerprint over stable serialized manifest content;
- flags confirming no pseudo-labels, no Test predictions inspected, no Test metrics inspected, and no model selection performed.

### 3. CLI

Implement scripts/common/build_wsm_manifest.py as a thin CLI around the reusable module.

It must accept:

    --data-root
    --manifest-output
    --audit-output

It must refuse to overwrite either output unless an explicit --overwrite flag is supplied.

It must refuse to write output inside src/, configs/, or docs/.

It must not modify the dataset source files.

### 4. No Chimera registration yet

This task does not add a config-selectable component, so no registry/plugin change is required.

Do not implement a datamodule here.

## Acceptance Criteria

- canonical builder and CLI exist only in the allowed paths;
- builder produces exactly the 13 required columns in the required order;
- retained manifest has exactly 8622 rows under the currently audited dataset state;
- row counts are train=6325, dev=933, test=1364;
- segment_id is unique for all 8622 retained rows;
- post-rule video_id overlap is zero across train/dev/test;
- all speaker_id values are null and audit explicitly says speaker_independence_verified=false;
- depression rows never contain a known y_parkinson value;
- Parkinson rows never contain a known y_depression value;
- unknown labels survive a CSV write/read round-trip as missing/null and are never 0-filled;
- observed_* masks exactly match corpus ownership;
- audio_available and video_available match real audited file existence;
- text_available is false for all rows;
- description_available is false for all rows;
- audit contains a deterministic SHA-256 manifest fingerprint;
- no Test predictions/performance metrics are inspected;
- no training/model selection occurs;
- no datamodule is created;
- python compilation passes;
- manifest verification assertions pass;
- git diff --check passes;
- git diff -- src/audio is empty;
- task branch is codex/task-001b;
- implementation commit is pushed to origin;
- main/master is not modified by implementing Codex;
- diff against origin/main contains only the four allowed tracked paths.

## Exact Verification Commands

Run from repository root after creating the task branch.

    python3 -m py_compile       src/common/data/__init__.py       src/common/data/wsm_manifest.py       scripts/common/build_wsm_manifest.py

    rm -f /tmp/wsm_stage1_manifest.csv /tmp/wsm_stage1_manifest_audit.json

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python       scripts/common/build_wsm_manifest.py       --data-root /media/maxim/Databases/WSM_NEW       --manifest-output /tmp/wsm_stage1_manifest.csv       --audit-output /tmp/wsm_stage1_manifest_audit.json

    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python - <<'PY'
    import json
    from pathlib import Path
    import pandas as pd

    manifest_path = Path("/tmp/wsm_stage1_manifest.csv")
    audit_path = Path("/tmp/wsm_stage1_manifest_audit.json")

    df = pd.read_csv(manifest_path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))

    required = [
        "segment_id", "video_id", "speaker_id", "corpus", "split",
        "y_depression", "y_parkinson",
        "observed_depression", "observed_parkinson",
        "audio_available", "video_available",
        "text_available", "description_available",
    ]

    assert list(df.columns) == required
    assert len(df) == 8622
    assert df["segment_id"].is_unique
    assert df["speaker_id"].isna().all()

    assert df.groupby("split").size().to_dict() == {
        "dev": 933,
        "test": 1364,
        "train": 6325,
    }

    dep = df["corpus"] == "depression"
    par = df["corpus"] == "parkinson"

    assert df.loc[dep, "y_depression"].notna().all()
    assert df.loc[dep, "y_parkinson"].isna().all()
    assert df.loc[par, "y_depression"].isna().all()
    assert df.loc[par, "y_parkinson"].notna().all()

    assert df.loc[dep, "observed_depression"].astype(bool).all()
    assert (~df.loc[dep, "observed_parkinson"].astype(bool)).all()
    assert (~df.loc[par, "observed_depression"].astype(bool)).all()
    assert df.loc[par, "observed_parkinson"].astype(bool).all()

    assert not df["text_available"].astype(bool).any()
    assert not df["description_available"].astype(bool).any()

    assert audit["speaker_independence_verified"] is False
    assert audit["segment_identity"]["collision_count"] == 0
    assert audit["split_overlap"]["any_video_overlap"] is False
    assert audit["manifest_fingerprint"]["algorithm"] == "sha256"
    assert len(audit["manifest_fingerprint"]["value"]) == 64

    assert audit["test_usage"]["model_predictions_inspected"] is False
    assert audit["test_usage"]["performance_metrics_inspected"] is False
    assert audit["test_usage"]["selection_or_tuning_performed"] is False
    assert audit["pseudo_labels_created"] is False

    print("canonical manifest assertions passed")
    PY

    git diff --check

    git diff -- src/audio

    git status --short

Before committing, inspect:

    git diff --       src/common/data/__init__.py       src/common/data/wsm_manifest.py       scripts/common/build_wsm_manifest.py       docs/PROGRESS_EN.md

Then commit and push according to AGENTS.md.

After commit and push, verify:

    git status --short

    git rev-parse --abbrev-ref HEAD

    git rev-parse HEAD

    git diff --stat origin/main...HEAD

    git diff --name-only origin/main...HEAD

The final diff name list must contain only:

    src/common/data/__init__.py
    src/common/data/wsm_manifest.py
    scripts/common/build_wsm_manifest.py
    docs/PROGRESS_EN.md

## Required PROGRESS_EN Update

Append TASK-001B facts without erasing prior evidence.

Record:

- exact branch name;
- implementation commit SHA;
- push result;
- manifest row/split counts;
- segment-id uniqueness;
- video split-overlap result;
- speaker_id null/unresolved status and speaker_independence_verified=false;
- disease observed/unknown counts;
- modality availability counts;
- manifest fingerprint;
- exact verification commands/results;
- confirmation that unknown disease labels remained null and were never mapped to negative;
- confirmation src/audio stayed unchanged;
- confirmation no Test predictions/metrics were inspected;
- confirmation no training/model selection ran;
- confirmation no datamodule was created;
- Stage 1 remains partial;
- recommended next atomic step only.

## Required Handoff

Respond in English using exactly these headings:

1. Outcome
2. Changed files
3. Verification
4. Plan status
5. Blockers and risks
6. Next atomic step

In the handoff explicitly state:

- branch: codex/task-001b;
- implementation commit SHA;
- whether the branch was pushed to origin;
- that implementing Codex did not modify main/master;
- diff summary against origin/main;
- src/audio unchanged;
- Test predictions/metrics not inspected.

Stop after this task. Do not implement the next Stage 1 task.
