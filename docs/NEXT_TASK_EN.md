# TASK-001F: Integrate Verified Speaker Evidence into the Canonical Manifest Consumer

## Role

You are the implementing Codex. Execute only this task, update docs/PROGRESS_EN.md, commit on the required task branch, push to origin, then stop. Follow AGENTS.md.

Required branch:

    codex/task-001f

Do not begin Stage 2, model development, training, pseudo-labeling, text/description work, or split redesign.

## Goal

Complete the Stage 1 speaker-evidence integration path created by TASK-001E.

Current behavior is intentionally incomplete: WSMManifestDataModule raises if speaker_independence_verified=true and cannot consume a valid authoritative speaker map end-to-end.

Implement a minimal optional authoritative speaker-map path so that:

- default behavior with no speaker map remains unchanged: speaker_id=None and speaker_independence_verified=false;
- when a supplied authoritative map passes TASK-001E audit, each manifest sample receives its authoritative speaker_id;
- the DataModule exposes speaker_independence_verified=true only after the real/supplied map passes the leakage gate;
- failing, incomplete, conflicting, or leaking maps are rejected;
- split assignments, labels, masks, and modality availability remain unchanged.

No real speaker map exists in the repository today, so Stage 1 remains partial after this task unless the dataset owner later supplies one.

## Required Reading

1. AGENTS.md
2. docs/PROJECT_REQUIREMENTS.md Sections 2, 9, 10, 11, 13
3. Stage 1 in docs/PLAN.md
4. docs/PROGRESS_EN.md through TASK-001E
5. docs/NEXT_TASK_EN.md
6. src/common/data/wsm_manifest.py
7. src/common/data/wsm_speaker_map.py
8. src/fusion/data/wsm_manifest_datamodule.py
9. scripts/common/build_wsm_manifest.py

## Allowed Files

- src/common/data/wsm_speaker_map.py
- src/fusion/data/wsm_manifest_datamodule.py
- scripts/common/build_wsm_manifest.py
- docs/PROGRESS_EN.md

No other tracked file may be modified.

## Forbidden Actions

- any change under src/audio;
- changing canonical split assignments;
- changing disease labels or observed masks;
- inferring speaker identity;
- accepting video_id as speaker_id;
- generating or committing a real/fabricated speaker map;
- relaxing TASK-001E validation;
- changing text/description availability;
- training or Test evaluation;
- reading Test predictions/performance metrics;
- Stage 2 work;
- editing docs/NEXT_TASK_EN.md;
- pushing to main/master;
- opening or merging a PR.

## Implementation Requirements

1. Extend src/common/data/wsm_speaker_map.py with a reusable function that, given canonical manifest rows and a supplied speaker-map path:
   - runs the existing strict audit;
   - requires stage1_split_gate_passed=true;
   - returns new rows with speaker_id populated from the validated mapping;
   - does not mutate the input rows;
   - fails if any canonical pair is unmapped;
   - preserves every field except speaker_id.

2. Update WSMManifestDataModule:
   - add optional speaker_map_path;
   - when absent, preserve current unresolved behavior exactly;
   - when present, validate/apply it before creating train/dev/test datasets;
   - remove the current hard failure on speaker_independence_verified=true;
   - set self.speaker_independence_verified from the verified gate result;
   - expose data.speaker_independence_verified accurately in describe_context;
   - expose data.stage1_split_gate_passed accurately;
   - keep DEV/Test separation unchanged.

3. Update scripts/common/build_wsm_manifest.py:
   - add optional --speaker-map;
   - when supplied, validate/apply speaker IDs before writing the manifest;
   - include a speaker_gate object in the audit output;
   - preserve the current no-map output behavior;
   - do not silently change the no-map manifest fingerprint/output.

4. Do not modify src/common/data/wsm_manifest.py.

## Acceptance Criteria

- default no-map path still produces 8622 rows with all speaker_id null and speaker_independence_verified=false;
- a synthetic complete non-leaking map populates all speaker IDs and yields speaker_independence_verified=true;
- the same synthetic verified map is accepted by WSMManifestDataModule;
- leaking/incomplete/conflicting synthetic maps are rejected;
- train/dev/test remain 6325/933/1364;
- DEV/Test remain separate;
- labels, observed masks, modality availability, segment IDs, and split values are byte/semantic-equivalent before vs after speaker enrichment except speaker_id;
- no real speaker map is committed;
- no Test predictions/metrics, training, tuning, or model selection;
- src/audio unchanged;
- python compilation passes;
- git diff --check passes;
- branch codex/task-001f is committed and pushed;
- diff against origin/main contains only the four allowed paths;
- Stage 1 remains partial unless a real authoritative speaker map is actually supplied.

## Exact Verification Commands

Run from repository root.

    python3 -m py_compile       src/common/data/wsm_speaker_map.py       src/fusion/data/wsm_manifest_datamodule.py       scripts/common/build_wsm_manifest.py

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python - <<'PY'
    import csv
    import tempfile
    from pathlib import Path

    from common.data.wsm_manifest import build_manifest
    from common.data.wsm_speaker_map import apply_verified_speaker_map, SpeakerMapError
    from fusion.data.wsm_manifest_datamodule import WSMManifestDataModule

    rows, _ = build_manifest("/media/maxim/Databases/WSM_NEW")
    assert len(rows) == 8622
    assert all(row["speaker_id"] is None for row in rows)

    pairs = {}
    for row in rows:
        pairs.setdefault((row["corpus"], row["video_id"]), row["split"])

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "speaker.csv"
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["corpus","video_id","speaker_id","source_reference"])
            for i, ((corpus, video_id), split) in enumerate(sorted(pairs.items())):
                w.writerow([corpus, video_id, f"{split}_speaker_{i}", "synthetic-contract-test"])

        enriched, gate = apply_verified_speaker_map(rows, path)
        assert gate["stage1_split_gate_passed"] is True
        assert gate["speaker_independence_verified"] is True
        assert all(row["speaker_id"] for row in enriched)

        for before, after in zip(rows, enriched):
            for key in before:
                if key != "speaker_id":
                    assert before[key] == after[key]

        dm = WSMManifestDataModule(
            data_root="/media/maxim/Databases/WSM_NEW",
            speaker_map_path=str(path),
        )
        assert dm.speaker_independence_verified is True
        assert len(dm.train_dataset) == 6325
        assert len(dm.val_dataset) == 933
        assert len(dm.test_dataset) == 1364
        assert dm.train_dataset[0]["speaker_id"] is not None

    dm_unresolved = WSMManifestDataModule(data_root="/media/maxim/Databases/WSM_NEW")
    assert dm_unresolved.speaker_independence_verified is False
    assert dm_unresolved.train_dataset[0]["speaker_id"] is None

    print("speaker integration smoke passed")
    PY

    git diff --check
    git diff -- src/audio
    git status --short

Before committing inspect only:

    git diff --       src/common/data/wsm_speaker_map.py       src/fusion/data/wsm_manifest_datamodule.py       scripts/common/build_wsm_manifest.py       docs/PROGRESS_EN.md

After commit/push:

    git status --short
    git rev-parse --abbrev-ref HEAD
    git rev-parse HEAD
    git diff --stat origin/main...HEAD
    git diff --name-only origin/main...HEAD

## Required PROGRESS_EN Update

Record exact branch, implementation SHA, push result, no-map behavior, synthetic verified integration result, DEV/Test counts/separation, rejection of invalid speaker maps, src/audio unchanged, no Test metrics, no training/model selection, and that Stage 1 remains partial until real authoritative speaker evidence is supplied.

## Required Handoff

Respond in English using exactly:

1. Outcome
2. Changed files
3. Verification
4. Plan status
5. Blockers and risks
6. Next atomic step

Explicitly include branch codex/task-001f, implementation SHA, push status, main/master untouched, diff summary, src/audio unchanged, Test metrics not inspected, and whether any real authoritative speaker map was available.

Stop after this task. Do not begin Stage 2.
