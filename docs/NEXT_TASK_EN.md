# TASK-001E: Establish the Authoritative Speaker-Map Contract and Stage 1 Leakage Gate

## Role

You are the implementing Codex. Execute only this task, update docs/PROGRESS_EN.md, commit the implementation on the required task branch, push that branch to origin, then stop. Follow AGENTS.md.

The required branch is:

    codex/task-001e

Do not begin Stage 2, model development, training, pseudo-labeling, or text/description work in this cycle.

## Goal

Implement the explicit, reusable contract required to verify speaker-independent splits without inventing speaker identities.

TASK-001A established that no authoritative speaker identifier exists in the currently audited raw segment CSVs or adjacent JSON metadata. Therefore this task must NOT infer speaker_id.

Instead, implement:

1. a strict authoritative speaker-map schema and validator;
2. a deterministic leakage audit against the canonical manifest;
3. a machine-readable unresolved gate report when no authoritative map is supplied.

The result must make Stage 1 gate status mechanically checkable:

- video split independence is already verified;
- speaker split independence is verified only when a valid authoritative speaker map covers the canonical manifest;
- without such a map, Stage 1 remains blocked/partial.

## Required Reading

Read before editing:

1. AGENTS.md;
2. docs/PROJECT_REQUIREMENTS.md, especially Sections 2, 9, 10, 11, and 13;
3. Stage 1 in docs/PLAN.md;
4. docs/PROGRESS_EN.md, especially TASK-001A through TASK-001D;
5. docs/NEXT_TASK_EN.md;
6. src/common/data/wsm_manifest.py;
7. scripts/common/audit_manifest_sources.py;
8. scripts/common/build_wsm_manifest.py.

## Allowed Files

- src/common/data/wsm_speaker_map.py;
- scripts/common/audit_speaker_map.py;
- docs/PROGRESS_EN.md.

No other tracked file may be modified.

## Forbidden Actions

- any change under src/audio;
- changing src/common/data/wsm_manifest.py in this task;
- changing the existing canonical manifest schema;
- assigning speaker_id from video_id;
- inferring speaker identity from path strings, row order, corpus, diagnosis, labels, filenames, transcript text, face similarity, voice similarity, channel names, or any other heuristic;
- scraping or downloading external personal/profile data;
- generating or committing a fabricated speaker map;
- changing existing splits;
- relaxing the requirement for speaker independence;
- marking Stage 1 complete when speaker coverage is unresolved;
- training, tuning, model selection, threshold selection, or Test evaluation;
- reading Test predictions or Test performance metrics;
- pseudo-labeling or reliability logic;
- dependency installation;
- broad refactors;
- editing docs/NEXT_TASK_EN.md;
- pushing implementation directly to main/master;
- opening or merging a PR.

Structural Test metadata may be used only for IDs/splits/leakage auditing, never for model decisions.

## Authoritative Speaker-Map Schema

Support a CSV with exactly these columns:

    corpus,video_id,speaker_id,source_reference

Requirements:

- corpus must be exactly depression or parkinson;
- video_id must be non-empty;
- speaker_id must be non-empty;
- source_reference must be non-empty and identify the authoritative source supplied by the dataset owner or documented metadata source;
- one (corpus, video_id) pair may map to exactly one speaker_id;
- duplicate identical rows may be rejected rather than silently deduplicated;
- conflicting mappings must fail clearly;
- the validator must never accept an implicit fallback speaker_id.

The map is external evidence. Do not commit any real speaker map in this task.

## Implementation Requirements

### 1. Reusable validator

Implement src/common/data/wsm_speaker_map.py with reusable functions to:

- load and strictly validate the speaker-map CSV;
- build a mapping keyed by (corpus, video_id);
- compare that mapping with canonical manifest rows produced by build_manifest(data_root);
- report coverage:
  - canonical unique (corpus, video_id) pairs;
  - mapped pairs;
  - unmapped pairs;
  - extra speaker-map pairs not present in the canonical manifest;
- verify every canonical row for the same (corpus, video_id) resolves to the same speaker_id;
- compute speaker sets for train/dev/test;
- compute pairwise speaker overlap train/dev, train/test, dev/test;
- compute pairwise video overlap as a regression check;
- set speaker_independence_verified=true only when:
  - canonical coverage is complete;
  - there are zero speaker conflicts;
  - there is zero speaker overlap across splits;
  - there is zero video overlap across splits.

### 2. Unresolved mode

The module/CLI must also support the current state where no speaker map is available.

When no map is supplied, the audit must produce a machine-readable result with at least:

    status = "unresolved"
    speaker_independence_verified = false
    reason = <explicit TASK-001A evidence summary>
    video_independence_verified = true/false from actual canonical rows
    required_map_schema = [...]
    next_required_evidence = "authoritative speaker map"

It must not invent mappings.

### 3. CLI

Implement scripts/common/audit_speaker_map.py.

Required arguments:

    --data-root
    --output

Optional:

    --speaker-map

Behavior:

- without --speaker-map: emit unresolved JSON and exit 0, because unresolved is a known project state, not a program crash;
- with --speaker-map: validate the map and emit resolved/failed gate evidence;
- malformed/conflicting supplied maps must exit nonzero;
- refuse to overwrite --output unless --overwrite is given;
- refuse output inside the dataset root, src/, configs/, or docs/;
- never modify dataset files.

### 4. Machine-readable report

The JSON report must contain at least:

- schema/version;
- status: unresolved, verified, or failed;
- canonical manifest fingerprint from the current builder audit;
- canonical row count and unique video-pair count;
- required speaker-map columns;
- map path when supplied;
- coverage counts;
- conflict counts/details;
- video split-overlap counts;
- speaker split-overlap counts when a map is supplied;
- video_independence_verified;
- speaker_independence_verified;
- stage1_split_gate_passed;
- test_usage:
  - model_predictions_inspected=false;
  - performance_metrics_inspected=false;
  - selection_or_tuning_performed=false.

stage1_split_gate_passed may be true only when both video_independence_verified and speaker_independence_verified are true.

## Acceptance Criteria

- reusable speaker-map validator exists;
- CLI unresolved mode runs successfully with current dataset and produces speaker_independence_verified=false;
- unresolved mode does not fabricate any speaker IDs;
- unresolved report includes required schema and next required evidence;
- canonical video split overlap remains zero;
- a synthetic complete non-leaking speaker map can verify successfully;
- a synthetic leaking speaker map is detected and stage1_split_gate_passed=false;
- a synthetic incomplete speaker map is detected and speaker_independence_verified=false;
- a synthetic conflicting map fails clearly;
- no real speaker map or personal/profile data is committed;
- no Test predictions/performance metrics are inspected;
- no training/model selection occurs;
- no src/audio change;
- python compilation passes;
- git diff --check passes;
- task branch is codex/task-001e;
- implementation is committed and pushed to origin;
- main/master is not modified by implementing Codex;
- diff against origin/main contains only the three allowed tracked paths;
- docs/PROGRESS_EN.md records that Stage 1 remains partial/blocked unless an authoritative complete non-leaking map is actually supplied and verified.

## Exact Verification Commands

Run from repository root after creating codex/task-001e from current origin/main.

    python3 -m py_compile       src/common/data/wsm_speaker_map.py       scripts/common/audit_speaker_map.py

    rm -f /tmp/wsm_speaker_gate_unresolved.json

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python       scripts/common/audit_speaker_map.py       --data-root /media/maxim/Databases/WSM_NEW       --output /tmp/wsm_speaker_gate_unresolved.json

    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python - <<'PY'
    import json
    from pathlib import Path

    data = json.loads(Path("/tmp/wsm_speaker_gate_unresolved.json").read_text(encoding="utf-8"))
    assert data["status"] == "unresolved"
    assert data["speaker_independence_verified"] is False
    assert data["stage1_split_gate_passed"] is False
    assert data["video_independence_verified"] is True
    assert data["required_speaker_map_columns"] == [
        "corpus", "video_id", "speaker_id", "source_reference"
    ]
    assert data["next_required_evidence"] == "authoritative speaker map"
    assert data["test_usage"]["model_predictions_inspected"] is False
    assert data["test_usage"]["performance_metrics_inspected"] is False
    assert data["test_usage"]["selection_or_tuning_performed"] is False
    print("unresolved speaker gate assertions passed")
    PY

Run a synthetic contract smoke that does not use or commit real identities:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python - <<'PY'
    import csv
    import json
    import tempfile
    from pathlib import Path

    from common.data.wsm_manifest import build_manifest
    from common.data.wsm_speaker_map import audit_speaker_map

    rows, _ = build_manifest("/media/maxim/Databases/WSM_NEW")
    pairs = {}
    for row in rows:
        pairs.setdefault((row["corpus"], row["video_id"]), row["split"])

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        good = root / "good.csv"
        with good.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["corpus", "video_id", "speaker_id", "source_reference"])
            for idx, ((corpus, video_id), split) in enumerate(sorted(pairs.items())):
                writer.writerow([corpus, video_id, f"{split}_speaker_{idx}", "synthetic-contract-test"])
        good_report = audit_speaker_map(rows, good)
        assert good_report["status"] == "verified"
        assert good_report["speaker_independence_verified"] is True
        assert good_report["stage1_split_gate_passed"] is True

        leaking = root / "leaking.csv"
        with leaking.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["corpus", "video_id", "speaker_id", "source_reference"])
            first_by_split = {}
            for idx, ((corpus, video_id), split) in enumerate(sorted(pairs.items())):
                speaker = f"{split}_speaker_{idx}"
                first_by_split.setdefault(split, (corpus, video_id))
                writer.writerow([corpus, video_id, speaker, "synthetic-contract-test"])

        # Rewrite one DEV and one TRAIN identity to the same synthetic speaker.
        raw = list(csv.reader(leaking.open(encoding="utf-8")))
        header, body = raw[0], raw[1:]
        train_i = next(i for i, row in enumerate(body) if pairs[(row[0], row[1])] == "train")
        dev_i = next(i for i, row in enumerate(body) if pairs[(row[0], row[1])] == "dev")
        body[train_i][2] = "shared_synthetic_speaker"
        body[dev_i][2] = "shared_synthetic_speaker"
        with leaking.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(body)

        leaking_report = audit_speaker_map(rows, leaking)
        assert leaking_report["speaker_independence_verified"] is False
        assert leaking_report["stage1_split_gate_passed"] is False
        assert leaking_report["speaker_split_overlap"]["train_vs_dev"]["count"] >= 1

        incomplete = root / "incomplete.csv"
        with incomplete.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["corpus", "video_id", "speaker_id", "source_reference"])
            items = sorted(pairs.items())
            for idx, ((corpus, video_id), split) in enumerate(items[:-1]):
                writer.writerow([corpus, video_id, f"{split}_speaker_{idx}", "synthetic-contract-test"])
        incomplete_report = audit_speaker_map(rows, incomplete)
        assert incomplete_report["speaker_independence_verified"] is False
        assert incomplete_report["coverage"]["unmapped_pairs"] == 1

    print("synthetic speaker-map contract smoke passed")
    PY

    git diff --check

    git diff -- src/audio

    git status --short

Before committing, inspect:

    git diff --       src/common/data/wsm_speaker_map.py       scripts/common/audit_speaker_map.py       docs/PROGRESS_EN.md

Then commit and push according to AGENTS.md.

After commit and push, verify:

    git status --short

    git rev-parse --abbrev-ref HEAD

    git rev-parse HEAD

    git diff --stat origin/main...HEAD

    git diff --name-only origin/main...HEAD

The final diff name list must contain only:

    src/common/data/wsm_speaker_map.py
    scripts/common/audit_speaker_map.py
    docs/PROGRESS_EN.md

## Required PROGRESS_EN Update

Append TASK-001E facts without erasing prior evidence.

Record:

- exact branch name;
- implementation commit SHA;
- push result;
- unresolved current speaker-gate result;
- canonical video-independence result;
- required authoritative map schema;
- synthetic verified/leaking/incomplete contract smoke results;
- explicit statement that synthetic maps are tests only and are not project evidence;
- exact verification commands/results;
- confirmation src/audio stayed unchanged;
- confirmation no Test predictions/metrics were inspected;
- confirmation no training/model selection ran;
- Stage 1 remains partial/blocked unless real authoritative speaker evidence is supplied;
- recommended next atomic step only.

## Required Handoff

Respond in English using exactly these headings:

1. Outcome
2. Changed files
3. Verification
4. Plan status
5. Blockers and risks
6. Next atomic step

Explicitly state:

- branch: codex/task-001e;
- implementation commit SHA;
- whether the branch was pushed to origin;
- that implementing Codex did not modify main/master;
- diff summary against origin/main;
- src/audio unchanged;
- Test predictions/metrics not inspected;
- whether any real authoritative speaker map was available;
- current stage1_split_gate_passed value.

Stop after this task. Do not infer speakers or begin Stage 2.
