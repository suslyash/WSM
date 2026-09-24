# TASK-004A: Implement the Canonical Sparse Audio+Video Fusion DataModule

## Role

You are the implementing Codex. Execute only this task, update docs/PROGRESS_EN.md, commit on the required task branch, push to origin, then stop. Follow AGENTS.md.

Required branch:

    codex/task-004a

Do not implement or train a fusion model in this task.
Do not modify src/audio.
Do not start text/description work.

## Goal

Start the now-active Stage 4 Fusion research with a correct audio+video data contract.

Implement and register a canonical A+V fusion DataModule that joins:

- frozen audio features from the accepted WavLM-base-plus layer 9 / pool 4 cache;
- accepted full-coverage DEPART-compatible video features;
- canonical sparse two-task targets.

The output must be ready for future F0/F1 fusion models and MUST NOT use the legacy task_id-selected single-task classification contract.

## Active Research Order

Text/description is explicitly deferred until after the fusion/RAMPS/core-ablation research cycle.

This task uses only:

    audio + video

No text or description input is required.

## Required Reading

1. AGENTS.md
2. docs/PROJECT_REQUIREMENTS.md Sections 2-10, 13-14
3. docs/PLAN.md Stage 4
4. docs/PROGRESS_EN.md through MANAGER-DECISION-016
5. docs/NEXT_TASK_EN.md
6. configs/wsm_mm_pd_dep_v1/audio/00_frozen_baseline.yaml
7. src/audio/data/wsm_audio_segment_dataset.py for cache layout/feature payload semantics only
8. src/video/data/wsm_video_cache_datamodule.py
9. src/common/data/wsm_manifest.py
10. src/common/utils/segment_index.py
11. src/fusion/data/wsm_manifest_datamodule.py

## Allowed Tracked Files

- src/fusion/data/wsm_av_fusion_datamodule.py
- src/fusion/data/__init__.py
- src/chimera_plugin.py
- docs/PROGRESS_EN.md

No other tracked file may be modified.

## Registry Key

Register:

    wsm_av_fusion_datamodule

Do not alter existing DataModule registry keys.

## Fixed Inputs

Data root:

    /media/maxim/Databases/WSM_NEW

Frozen audio feature cache root:

    /media/maxim/Databases/WSM_NEW/features

Frozen audio feature contract:

- extractor type: transformers_ssl
- model: microsoft/wavlm-base-plus
- layer: 9
- temporal_pool: 4
- cache path convention identical to WSMAudioSegmentDataset
- payload keys:
  - audio_temporal
  - audio_cls

Accepted video cache root:

    /media/maxim/Programs/Features/WSM/video_depart_v1_fullframe_fallback/cache

Video feature contract:

- feature dim: 512
- temporal T in [1,60]
- all real temporal positions valid
- full canonical coverage

Expected canonical counts:

    train      = 6325
    dev        = 933
    test_none  = 1364
    test_soft  = 1208
    test_hard  = 1014

## Join Identity

Join modalities by the canonical segment identity corresponding to:

    corpus/task
    video_id
    segment_file

The canonical segment_id remains authoritative.

Use the same canonical/raw mapping already established in Stage 1/2.

Requirements:

- exactly one audio feature payload per canonical row;
- exactly one video artifact per canonical row;
- no duplicate identity;
- no silent row dropping;
- no positional/index-based join;
- no corpus/task inference from filenames beyond the established canonical/raw index.

## Audio Cache Policy

Do NOT modify or regenerate src/audio.

Do not silently run a new audio feature extractor in this task.

The DataModule must consume the already frozen cache and fail clearly if a required audio cache file is missing.

For each audio payload validate at least:

- audio_temporal is Tensor [Ta,Da], Ta>=1;
- audio_cls is Tensor [Da];
- dimensions agree;
- tensors are finite after the existing frozen cache contract;
- Da is constant across the dataset;
- payload identity/path corresponds to the expected canonical row.

Expose the detected frozen audio feature dimension via context.

## Video Cache Policy

Consume the accepted full-coverage cache.

You MAY reuse public constants/contracts from video.data.wsm_video_cache_datamodule, but do not mutate video code.

Validate for every selected row:

- one cache index record exists;
- status is extracted/reused;
- artifact segment_id matches;
- features [Tv,512], 1<=Tv<=60;
- valid_mask bool [Tv] and all true;
- accepted CLIP/YOLO/preprocessing identities remain valid;
- no unavailable row.

If using shared validation logic requires a private helper, prefer a small local validation adapter over depending on an unstable private symbol.

## Dataset Sample Contract

Each sample must expose:

    inputs["audio"]      -> float Tensor [Ta,Da]
    inputs["audio_cls"]  -> float Tensor [Da]
    inputs["video"]      -> float Tensor [Tv,512]

    targets              -> float Tensor [2]
    observed_mask        -> bool Tensor [2]
    audio_mask           -> bool Tensor [Ta]
    video_mask           -> bool Tensor [Tv]

Metadata must include at least:

- segment_id
- video_id
- corpus
- split
- segment_file
- evaluation_protocol when applicable
- audio_cache_path
- video_cache_path
- video_cache_fingerprint
- audio_temporal_length
- video_temporal_length

Unknown disease target MUST remain NaN with observed_mask=false.

Do not emit task_id as a model input.

## Collate Contract

Implement a fusion-local collate function returning chimera_ml.core.batch.Batch.

Pad audio and video independently to their real batch maxima:

    audio -> [B,max_Ta,Da]
    video -> [B,max_Tv,512]

Create:

    audio_mask -> [B,max_Ta] bool
    video_mask -> [B,max_Tv] bool
    observed_mask -> [B,2] bool
    modality_available -> [B,2] bool ordered [audio,video]

For this complete A+V dataset, modality_available must currently be all true.

Padded positions must be exact zero with mask=false.

Return:

    inputs={
        "audio": ...,
        "audio_cls": ...,
        "video": ...,
    }

    targets=[B,2]

    masks={
        "audio_mask": ...,
        "video_mask": ...,
        "observed_mask": ...,
        "modality_available": ...,
    }

Do NOT include:

- task_ids
- corpus id as numeric model input
- split id
- Test protocol id
- labels inside inputs

## DataModule Contract

Constructor parameters must include at least:

    data_root
    audio_feature_cache_root
    video_cache_root

Plus normal DataLoader parameters:

    batch_size
    num_workers
    pin_memory
    persistent_workers
    shuffle_train
    drop_last_train

Datasets:

    train_dataset
    val_dataset
    test_dataset={
        "test_none": ...,
        "test_soft": ...,
        "test_hard": ...,
    }

Expected lengths:

    len(train_dataset) == 6325
    len(val_dataset) == 933
    len(test_none) == 1364
    len(test_soft) == 1208
    len(test_hard) == 1014

No unavailable rows are allowed under the currently accepted full audio/video cache contract.

## Epoch-Level Evaluation Streams

Override val_dataloader() to return exactly:

    {
        "dev": ...,
        "test_none": ...,
        "test_soft": ...,
        "test_hard": ...,
    }

DEV remains val_dataset only.

Test datasets remain separate and are not merged into val_dataset.

All evaluation loaders:

- shuffle=false
- drop_last=false
- fusion collate

## Context Contract

describe_context must expose at least:

- data.num_tasks = 2
- data.task_names = ["depression","parkinson"]
- data.modality_names = ["audio","video"]
- data.audio_feature_dim
- data.video_feature_dim = 512
- data.audio_video_fusion = true
- data.train_rows = 6325
- data.dev_rows = 933
- data.test_none_rows = 1364
- data.test_soft_rows = 1208
- data.test_hard_rows = 1014
- data.audio_unavailable = 0
- data.video_unavailable = 0
- data.test_protocols = ["test_none","test_soft","test_hard"]

Do not expose Test metric values.

## Required Audit

At DataModule construction or via a reusable audit method, record/verify:

- canonical count by split;
- unique join count;
- audio cache files found;
- video cache records found;
- audio feature dimension;
- video feature dimension;
- zero missing audio;
- zero missing video;
- zero duplicate joins;
- exact Test raw membership counts;
- no dropped rows.

The audit may be kept as an in-memory dict attribute such as:

    dm.audit

No new tracked report file is required in this task.

## Acceptance Criteria

Registry/data:

- DATAMODULES contains wsm_av_fusion_datamodule;
- plugin imports it without project-module warning;
- train/dev/test counts exactly match 6325/933/1364/1208/1014;
- all 8622 canonical unique rows have both audio and video;
- no missing audio cache file;
- no missing video cache artifact;
- no duplicate join;
- unknown labels stay NaN+masked;
- no task_id model input.

Batch:

- audio [B,max_Ta,Da];
- audio_cls [B,Da];
- video [B,max_Tv,512];
- audio_mask/video_mask bool;
- observed_mask [B,2];
- modality_available [B,2] all true;
- exact-zero padding;
- batch metadata preserves canonical identity;
- mixed variable-length audio/video batch collates successfully.

Streams:

- val_dataloader keys exactly dev/test_none/test_soft/test_hard;
- DEV/Test remain separate;
- no Test selector logic is introduced.

Safety:

- no fusion model implementation;
- no training config;
- no training run;
- no Test metric computation;
- no text/description work;
- src/audio unchanged;
- video source unchanged;
- python compilation passes;
- registry smoke passes;
- git diff --check passes;
- branch codex/task-004a committed and pushed;
- main/master untouched;
- tracked diff contains only the four allowed paths.

## Exact Verification Commands

Compile:

    python3 -m py_compile \
      src/fusion/data/wsm_av_fusion_datamodule.py \
      src/fusion/data/__init__.py \
      src/chimera_plugin.py

Registry smoke:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python - <<'PY'
    import warnings
    from chimera_ml.core.registry import DATAMODULES
    import chimera_plugin

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        chimera_plugin.register()

    project_warnings = [
        str(x.message)
        for x in caught
        if "Failed to import" in str(x.message)
        and "fusion.data.wsm_av_fusion_datamodule" in str(x.message)
    ]
    assert not project_warnings, project_warnings
    assert "wsm_av_fusion_datamodule" in DATAMODULES.keys()
    print("A+V fusion DataModule registry smoke passed")
    PY

Full DataModule contract smoke:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python - <<'PY'
    import torch

    from fusion.data.wsm_av_fusion_datamodule import WSMAVFusionDataModule

    dm = WSMAVFusionDataModule(
        data_root="/media/maxim/Databases/WSM_NEW",
        audio_feature_cache_root="/media/maxim/Databases/WSM_NEW/features",
        video_cache_root="/media/maxim/Programs/Features/WSM/video_depart_v1_fullframe_fallback/cache",
        batch_size=4,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )

    assert len(dm.train_dataset) == 6325
    assert len(dm.val_dataset) == 933
    assert len(dm.test_dataset["test_none"]) == 1364
    assert len(dm.test_dataset["test_soft"]) == 1208
    assert len(dm.test_dataset["test_hard"]) == 1014

    assert dm.audit["canonical_total"] == 8622
    assert dm.audit["joined_total"] == 8622
    assert dm.audit["missing_audio"] == 0
    assert dm.audit["missing_video"] == 0
    assert dm.audit["duplicate_join"] == 0

    loaders = dm.val_dataloader()
    assert list(loaders.keys()) == ["dev", "test_none", "test_soft", "test_hard"]

    # Build a real mixed batch.
    batch = next(iter(dm.train_dataloader()))

    audio = batch.inputs["audio"]
    audio_cls = batch.inputs["audio_cls"]
    video = batch.inputs["video"]
    audio_mask = batch.get_masks("audio_mask")
    video_mask = batch.get_masks("video_mask")
    observed = batch.get_masks("observed_mask")
    available = batch.get_masks("modality_available")

    assert audio.ndim == 3
    assert audio_cls.ndim == 2
    assert video.ndim == 3 and video.shape[-1] == 512
    assert audio.shape[0] == video.shape[0] == 4
    assert audio_cls.shape[0] == 4
    assert tuple(audio_mask.shape) == tuple(audio.shape[:2])
    assert tuple(video_mask.shape) == tuple(video.shape[:2])
    assert tuple(observed.shape) == (4,2)
    assert tuple(available.shape) == (4,2)
    assert available.dtype == torch.bool and bool(available.all())
    assert tuple(batch.targets.shape) == (4,2)

    # Padding must be exact zero.
    assert torch.equal(audio[~audio_mask], torch.zeros_like(audio[~audio_mask]))
    assert torch.equal(video[~video_mask], torch.zeros_like(video[~video_mask]))

    # Unknown targets remain masked NaN.
    assert torch.isnan(batch.targets[~observed]).all()

    # No legacy task-id model input.
    assert "task_ids" not in batch.inputs
    assert "task_id" not in batch.inputs

    print(
        "A+V fusion DataModule smoke passed",
        audio.shape,
        video.shape,
        dm.audio_feature_dim,
    )
    PY

Add a deterministic identity audit proving that sampled rows map to the same canonical segment across audio and video paths, including at least one row from each corpus and each split/protocol.

Then:

    git diff --check
    git diff -- src/audio
    git diff -- src/video
    git status --short

Before commit inspect only:

    git diff -- \
      src/fusion/data/wsm_av_fusion_datamodule.py \
      src/fusion/data/__init__.py \
      src/chimera_plugin.py \
      docs/PROGRESS_EN.md

After commit/push:

    git status --short
    git rev-parse --abbrev-ref HEAD
    git rev-parse HEAD
    git diff --stat origin/main...HEAD
    git diff --name-only origin/main...HEAD

## Required PROGRESS_EN Update

Record:

- branch;
- implementation commit SHA;
- push result;
- registry key;
- exact frozen audio cache contract/path;
- exact video cache contract/path;
- canonical/join counts;
- detected audio feature dim;
- video feature dim;
- missing/duplicate counts;
- train/dev/Test protocol dataset counts;
- batch shapes from a real batch;
- sparse target/observed-mask verification;
- no-task-id verification;
- four evaluation stream keys;
- confirmation no model/config/training/Test metrics;
- confirmation text/description remains deferred;
- src/audio unchanged;
- src/video unchanged;
- Stage 4 status;
- recommended next atomic step only.

## Required Handoff

Respond in English using exactly:

1. Outcome
2. Changed files
3. Verification
4. Plan status
5. Blockers and risks
6. Next atomic step

Explicitly include:

- branch codex/task-004a;
- implementation commit SHA;
- pushed-to-origin status;
- main/master untouched;
- registry key;
- audio feature dim;
- train/dev/test counts;
- join completeness;
- real batch audio/video shapes;
- no task_id input;
- no training/Test metrics;
- text/description deferred;
- src/audio unchanged;
- src/video unchanged.

Stop after this task.
