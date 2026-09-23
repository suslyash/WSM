# TASK-001A: Audit Manifest Sources and Freeze the Stage 1 Data Contract

## Role

You are the implementing Codex. Execute only this task, update PROGRESS_EN.md, then stop. Follow AGENTS.md. Do not begin manifest implementation or any later Stage 1 task in the same cycle.

## Goal

Start PLAN Stage 1 with a reproducible source audit for the canonical segment manifest. Determine the authoritative raw source for each required manifest field and the structural split-identity evidence before any production manifest or new datamodule is implemented.

This task is an audit/contract task only. It must not guess a speaker identity, fabricate modality availability, or change training behavior.

## Required Reading

Read these files before editing:

1. docs/PROJECT_REQUIREMENTS.md, especially Sections 2, 9, 10, and 13;
2. Stage 1 in docs/PLAN.md;
3. docs/PROGRESS_EN.md;
4. docs/PROJECT_INIT_STRUCTURE.md, especially the segment-index and datamodule audit;
5. src/common/utils/segment_index.py;
6. src/audio/data/wsm_audio_segment_datamodule.py.

Inspect the actual local dataset metadata under /media/maxim/Databases/WSM_NEW only as needed for this structural audit.

## Allowed Files

- scripts/common/audit_manifest_sources.py;
- docs/PROGRESS_EN.md.

Creating the scripts/common directory is allowed if it does not exist.

No other tracked file may be modified.

## Forbidden Actions

- any change under src/audio;
- any change under src/common, src/fusion, src/video, src/text, or src/description;
- creating the production canonical manifest;
- creating or modifying a Chimera datamodule, dataset, loss, model, metric, callback, config, or registry entry;
- changing existing split assignments;
- treating an unknown disease label as 0/negative;
- inferring speaker_id from video_id, path text, row order, corpus, or any other heuristic;
- fabricating audio/video/text/description availability when no real file/source exists;
- training, tuning, checkpoint selection, threshold selection, or Test evaluation;
- reading or reporting Test model predictions or Test performance metrics;
- repairing the legacy audio Test-in-validation behavior in this task;
- broad refactors or cleanup outside this audit.

Structural inspection of Test metadata is allowed only for schema, IDs, split-overlap detection, filenames, and file-existence auditing. It must not be used for model or hyperparameter decisions.

## Implementation Requirements

Implement scripts/common/audit_manifest_sources.py as a deterministic, read-only audit utility.

The audit must:

1. inspect the raw metadata used by the existing depression and Parkinson segment indexes for train/dev/test;
2. record the exact source files and exact raw column names found for each corpus/split;
3. report row counts and unique video_id counts per corpus/split without changing any split;
4. determine whether an explicit authoritative speaker identifier exists in the raw metadata or an adjacent documented metadata source;
5. if an authoritative speaker identifier is not available, report speaker_id as unresolved and do not substitute video_id or invent a mapping;
6. verify whether the deterministic candidate segment identity (corpus + video_id + segment_file) is unique, and report any collisions;
7. report video_id overlap across train/dev/test after the existing split rules and, only if an authoritative speaker identifier exists, speaker_id overlap across train/dev/test;
8. identify the real observable source/path rule, if any, for each availability field:
   - audio_available;
   - video_available;
   - text_available;
   - description_available.
   If a modality has no existing authoritative source yet, report it as unavailable/unresolved rather than synthesizing one;
9. emit a machine-readable JSON audit to a user-specified --output path;
10. include in that JSON a proposed raw-to-canonical field mapping for exactly these required manifest columns:

       segment_id, video_id, speaker_id, corpus, split,
       y_depression, y_parkinson,
       observed_depression, observed_parkinson,
       audio_available, video_available,
       text_available, description_available

11. encode the label contract in the audit mapping:
    - depression-corpus rows: observed_depression=true and observed_parkinson=false;
    - Parkinson-corpus rows: observed_depression=false and observed_parkinson=true;
    - the unobserved disease target is unknown/null, never 0;
    - no pseudo-label is created;
12. make no writes under the dataset root;
13. fail clearly on malformed or contradictory metadata instead of silently repairing it.

Do not claim the Stage 1 gate is complete. This task establishes source evidence for the later manifest implementation.

## Acceptance Criteria

- scripts/common/audit_manifest_sources.py exists and is read-only with respect to the dataset;
- the audit runs successfully on /media/maxim/Databases/WSM_NEW, or reports a concrete source blocker without guessing;
- generated JSON contains corpus/split source files, raw columns, row/video counts, segment-identity uniqueness, split-overlap results, speaker-source status, modality-source status, and the full proposed canonical field mapping;
- the proposed label mapping explicitly preserves the unknown disease target as null/unknown with observed_* false;
- speaker_id is supported by an explicit source or is explicitly marked unresolved; video_id is never silently reused as speaker_id;
- no Test prediction/performance metric is inspected or reported;
- no production manifest or datamodule is created;
- python compilation passes;
- git diff --check passes;
- git diff -- src/audio is empty;
- the final tracked diff contains only scripts/common/audit_manifest_sources.py and docs/PROGRESS_EN.md;
- docs/PROGRESS_EN.md records exact commands, results, discovered source mappings, unresolved fields/blockers, and the recommended next atomic step;
- Stage 1 remains partial/not complete after this task.

## Exact Verification Commands

Run exactly these checks from the repository root:

    python3 -m py_compile scripts/common/audit_manifest_sources.py

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python       scripts/common/audit_manifest_sources.py       --data-root /media/maxim/Databases/WSM_NEW       --output /tmp/wsm_stage1_manifest_source_audit.json

    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python - <<'PY'
    import json
    from pathlib import Path

    path = Path("/tmp/wsm_stage1_manifest_source_audit.json")
    data = json.loads(path.read_text(encoding="utf-8"))

    required = [
        "segment_id", "video_id", "speaker_id", "corpus", "split",
        "y_depression", "y_parkinson",
        "observed_depression", "observed_parkinson",
        "audio_available", "video_available",
        "text_available", "description_available",
    ]

    mapping = data["canonical_field_mapping"]
    assert list(mapping) == required or set(mapping) == set(required)

    label_contract = data["label_contract"]
    assert label_contract["depression"]["observed_depression"] is True
    assert label_contract["depression"]["observed_parkinson"] is False
    assert label_contract["parkinson"]["observed_depression"] is False
    assert label_contract["parkinson"]["observed_parkinson"] is True
    assert label_contract["depression"]["y_parkinson"] is None
    assert label_contract["parkinson"]["y_depression"] is None

    speaker = data["speaker_source"]
    assert speaker["status"] in {"resolved", "unresolved"}
    if speaker["status"] == "unresolved":
        assert not speaker.get("fallback_to_video_id", False)

    assert data["test_usage"]["model_predictions_inspected"] is False
    assert data["test_usage"]["performance_metrics_inspected"] is False
    print("manifest source audit assertions passed")
    PY

    git diff --check

    git diff -- src/audio

    git status --short

## Required Handoff

Update docs/PROGRESS_EN.md without erasing prior evidence.

Respond in English using exactly the AGENTS.md manager-handoff headings:

1. Outcome
2. Changed files
3. Verification
4. Plan status
5. Blockers and risks
6. Next atomic step

Include:

- exact commands and concise results;
- the resolved/unresolved source for speaker_id;
- the resolved/unresolved source for each modality availability field;
- whether segment identity was unique;
- video and, if available, speaker split-overlap findings;
- confirmation that unknown disease labels were not mapped to negative;
- confirmation that src/audio stayed unchanged;
- confirmation that no Test predictions/metrics were inspected;
- confirmation that no production manifest/datamodule was created.

Stop after this task. Do not implement the canonical manifest or any following Stage 1 task.
