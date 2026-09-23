# TASK-001C: Implement the Canonical Manifest Consumer and Separate DEV/Test DataModule

## Role

You are the implementing Codex. Execute only this task, update docs/PROGRESS_EN.md, commit the implementation on the required task branch, push that branch to origin, then stop. Follow AGENTS.md.

The required branch is:

    codex/task-001c

Do not begin video/text/description modeling, masked-loss implementation, training, or any later Stage 1/2 task in this cycle.

## Goal

Implement the first Chimera-compatible consumer of the canonical Stage 1 manifest.

The new datamodule must:

- consume the canonical partial-label manifest contract from src/common/data/wsm_manifest.py;
- expose train, DEV, and Test as separate datasets;
- never inject Test into validation;
- preserve two independent disease targets with explicit observed-task masks;
- expose modality-availability masks without fabricating unavailable text/description streams;
- remain data/contract-only: no model, loss, training config, or feature extraction in this task.

Stage 1 must remain partial after this task because authoritative speaker identity is still unresolved.

## Required Reading

Read before editing:

1. AGENTS.md;
2. docs/PROJECT_REQUIREMENTS.md, especially Sections 2, 3, 4, 9, 10, 11, and 13;
3. Stage 1 in docs/PLAN.md;
4. docs/PROGRESS_EN.md, especially TASK-001A and TASK-001B;
5. docs/NEXT_TASK_EN.md;
6. src/common/data/wsm_manifest.py;
7. src/audio/data/wsm_audio_segment_datamodule.py, for Chimera DataModule conventions only;
8. src/chimera_plugin.py.

## Allowed Files

- src/fusion/data/__init__.py;
- src/fusion/data/wsm_manifest_datamodule.py;
- src/chimera_plugin.py;
- docs/PROGRESS_EN.md.

Creating src/fusion/data is allowed if it does not exist.

No other tracked file may be modified.

## Forbidden Actions

- any change under src/audio;
- changing src/common/data/wsm_manifest.py;
- changing source labels, BAD_SEGMENTS, split priority, or manifest semantics;
- inferring speaker_id or claiming speaker independence is verified;
- treating an unknown disease label as 0/negative;
- creating pseudo-labels;
- making Test part of val_dataset or any validation loader;
- reading Test predictions or Test performance metrics;
- training, tuning, checkpoint selection, threshold selection, or model selection;
- adding model, loss, metric, callback, optimizer, or training config code;
- feature extraction or cache generation;
- enabling text_available or description_available beyond the canonical manifest;
- video/text/description/fusion model development;
- dependency installation;
- broad refactors;
- editing docs/NEXT_TASK_EN.md;
- pushing implementation directly to main/master;
- opening or merging a PR.

## Implementation Requirements

### 1. Canonical manifest dataset

Implement a lightweight dataset in src/fusion/data/wsm_manifest_datamodule.py that consumes canonical manifest rows only.

Each sample must expose, at minimum:

- segment_id;
- video_id;
- speaker_id;
- corpus;
- split;
- targets: float tensor/array of shape [2] ordered [depression, parkinson];
- observed_mask: bool tensor/array of shape [2] ordered [depression, parkinson];
- modality_available: bool tensor/array of shape [4] ordered [audio, video, text, description].

Unknown targets must remain semantically unknown. For tensor batching, it is acceptable to use a neutral numeric placeholder only if and only if observed_mask=false for that element and the implementation explicitly prevents interpreting the placeholder as supervision. Prefer NaN if compatible with the collate/batch contract.

The dataset must assert:

- observed labels are 0/1;
- unobserved labels are missing in the canonical row;
- observed_mask exactly matches non-null target ownership;
- no sample has both observed_mask values false;
- modality availability comes directly from the canonical manifest.

### 2. Chimera DataModule

Implement a DataModule registered as:

    wsm_manifest_datamodule

It must accept at least:

- data_root;
- optional manifest_path;
- optional manifest_audit_path.

Behavior:

- if manifest_path is supplied, load and validate that canonical CSV;
- otherwise build the canonical manifest in memory with build_manifest(data_root);
- train_dataset contains only split=train;
- val_dataset contains DEV only;
- test_dataset contains Test only;
- Test MUST NOT be inserted into val_dataset;
- no TEST_NONE/SOFT/HARD filtering in this new datamodule;
- no Test label-derived selection behavior.

Use the project's existing DataModule style, but do not copy the legacy audio Test-in-validation behavior.

### 3. Collation

Provide a deterministic collate function suitable for a smoke batch.

The batch must contain:

- metadata lists for IDs/corpus/split;
- targets with shape [B, 2];
- observed_mask with shape [B, 2];
- modality_available with shape [B, 4].

The collate function does not need its own registry key unless it is config-selectable.

### 4. Context description

describe_context must expose at least:

- data.num_tasks = 2;
- data.task_names = ["depression", "parkinson"];
- data.modality_names = ["audio", "video", "text", "description"];
- data.manifest_schema/version when available;
- data.speaker_independence_verified = false;
- train/dev/test row counts.

Do not expose any Test performance value.

### 5. Plugin registration

Update src/chimera_plugin.py to import the new registration module explicitly.

A missing required project module must not be hidden as an optional dependency warning.

Do not alter unrelated plugin registrations.

## Acceptance Criteria

- DATAMODULES contains wsm_manifest_datamodule after plugin registration;
- plugin import emits no project-module warning;
- train/dev/test row counts are exactly 6325 / 933 / 1364 under the current audited dataset state;
- val_dataset contains DEV only;
- test_dataset contains Test only;
- no Test dataset/key/object is present in val_dataset;
- one train smoke batch has targets shape [B,2], observed_mask [B,2], modality_available [B,4];
- every observed target in the smoke batch is finite and 0/1;
- unobserved positions are never treated as supervised negatives;
- depression-corpus samples have mask [true,false];
- Parkinson-corpus samples have mask [false,true];
- text and description availability remain false under the current manifest;
- speaker_id remains unresolved/null and speaker_independence_verified=false;
- no Test predictions/performance metrics are inspected;
- no training/model selection occurs;
- no src/audio change;
- python compilation passes;
- registry smoke passes;
- dataset/collate smoke passes;
- git diff --check passes;
- task branch is codex/task-001c;
- implementation is committed and pushed to origin;
- main/master is not modified by implementing Codex;
- diff against origin/main contains only the four allowed tracked paths.

## Exact Verification Commands

Run from repository root after creating codex/task-001c from the current origin/main.

    python3 -m py_compile       src/fusion/data/__init__.py       src/fusion/data/wsm_manifest_datamodule.py       src/chimera_plugin.py

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python - <<'PY'
    import warnings

    from chimera_ml.core.registry import DATAMODULES
    import chimera_plugin

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        chimera_plugin.register()

    project_warnings = [
        str(item.message)
        for item in caught
        if "Failed to import" in str(item.message)
        and "fusion.data.wsm_manifest_datamodule" in str(item.message)
    ]
    assert not project_warnings, project_warnings
    assert "wsm_manifest_datamodule" in DATAMODULES
    print("registry assertions passed")
    PY

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python - <<'PY'
    import math

    import torch

    from fusion.data.wsm_manifest_datamodule import WSMManifestDataModule

    dm = WSMManifestDataModule(data_root="/media/maxim/Databases/WSM_NEW")

    assert len(dm.train_dataset) == 6325
    assert len(dm.val_dataset) == 933
    assert len(dm.test_dataset) == 1364

    assert all(dm.val_dataset[i]["split"] == "dev" for i in range(min(32, len(dm.val_dataset))))
    assert all(dm.test_dataset[i]["split"] == "test" for i in range(min(32, len(dm.test_dataset))))

    batch = dm.collate_fn([dm.train_dataset[i] for i in range(8)])
    assert tuple(batch["targets"].shape) == (8, 2)
    assert tuple(batch["observed_mask"].shape) == (8, 2)
    assert tuple(batch["modality_available"].shape) == (8, 4)

    for target, mask in zip(batch["targets"], batch["observed_mask"]):
        assert bool(mask.any())
        for value, observed in zip(target.tolist(), mask.tolist()):
            if observed:
                assert value in (0.0, 1.0)
                assert math.isfinite(value)

    for sample in [dm.train_dataset[i] for i in range(min(128, len(dm.train_dataset)))]:
        expected = [True, False] if sample["corpus"] == "depression" else [False, True]
        assert sample["observed_mask"].tolist() == expected
        assert sample["modality_available"].tolist()[2:] == [False, False]
        assert sample["speaker_id"] is None

    assert dm.speaker_independence_verified is False
    print("manifest datamodule smoke passed")
    PY

    git diff --check

    git diff -- src/audio

    git status --short

Before committing, inspect:

    git diff --       src/fusion/data/__init__.py       src/fusion/data/wsm_manifest_datamodule.py       src/chimera_plugin.py       docs/PROGRESS_EN.md

Then commit and push according to AGENTS.md.

After commit and push, verify:

    git status --short

    git rev-parse --abbrev-ref HEAD

    git rev-parse HEAD

    git diff --stat origin/main...HEAD

    git diff --name-only origin/main...HEAD

The final diff name list must contain only:

    src/fusion/data/__init__.py
    src/fusion/data/wsm_manifest_datamodule.py
    src/chimera_plugin.py
    docs/PROGRESS_EN.md

## Required PROGRESS_EN Update

Append TASK-001C facts without erasing prior evidence.

Record:

- exact branch name;
- implementation commit SHA;
- push result;
- registration key;
- train/dev/test counts;
- proof that val contains DEV only and Test is separate;
- batch target/mask/modality shapes;
- exact unknown-target representation and why it cannot become supervised negative;
- speaker_id unresolved/null and speaker_independence_verified=false;
- exact verification commands/results;
- confirmation src/audio stayed unchanged;
- confirmation no Test predictions/metrics were inspected;
- confirmation no training/model selection ran;
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

Explicitly state:

- branch: codex/task-001c;
- implementation commit SHA;
- whether the branch was pushed to origin;
- that implementing Codex did not modify main/master;
- diff summary against origin/main;
- src/audio unchanged;
- Test predictions/metrics not inspected.

Stop after this task. Do not implement a loss, training config, or later Stage 1/2 task.
