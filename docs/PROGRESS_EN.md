# WSM Project Progress

## 1. Current State

Plan initialized: 2026-09-23.

Current stage: **Stage 2 in progress**.

Final Test authorized: **no**.

## 2. Stage Status

| Stage | Status | Gate | Evidence |
|---|---|---|---|
| Paper/project analysis | complete | Baselines, structure, requirements, plan | BASELINES.md, PROJECT_INIT_STRUCTURE.md, PROJECT_REQUIREMENTS.md, PLAN.md |
| 0. Reproducible base | complete | Canonical frozen audio config validates and registry/smoke gates pass | Verification records TASK-000-A and TASK-000B below |
| 1. Manifest/partial-label contract | complete | Canonical manifest, separate DEV/Test consumer, masked sparse loss, zero video leakage, and owner-accepted speaker-independent split contract | TASK-001A through TASK-001E plus manager decision below |
| 2. Video | partial | Deterministic V1 preprocessing/cache contract implemented; no model winner selected | TASK-002A |
| 3. Text/description | not started | At most two families; prompt audit | None |
| 4. Fusion baselines | not started | Comparable F0/F1/F2 | None |
| 5. RAMPS | not started | Direct reliable missing-head gradient | None |
| 6. Ablations | not started | Claims backed by multi-seed evidence | None |
| 7. Final evaluation | locked | Config freeze and manager authorization | None |

## 3. Frozen Historical Audio Reference

- run: wsm_audio_models-e0ce-006;
- selected epoch: 4;
- DEV/Mean_Score: 0.787827;
- DEV depression Score: 0.747918;
- DEV Parkinson Score: 0.827735;
- historical TEST_NONE/SOFT/HARD Mean_Score: 0.809486 / 0.815156 / 0.828135;
- WavLM-base-plus layer 9, pool 4;
- temporal Transformer: hidden 192, 3 layers, 4 heads, 128 steps.

Historical Test values came from an existing summary; they were not used for a new selection.

## 4. Open Blockers

1. The existing AV YAML is legacy/non-runnable: wsm_segment_datamodule and wsm_avsync_loss are absent from the registry.
2. The AV YAML also lacks snapshot_callback and early_stopping_callback and monitors dev/mean_macro_f1 instead of dev/mean_score.
3. The current audio datamodule still includes Test loaders in each validation epoch; this task did not change it.
4. Authoritative speaker_id remains unavailable. The dataset owner explicitly accepts the existing train/dev/test partition as speaker-independent; this is an owner-provided assumption, not a measured identity audit.
5. Canonical DEV/Test separation and observed-label-only masked loss are implemented. Stage 1 is closed under the owner-provided split assumption.

## 5. Execution Log

### DOC-001 — Analysis and research plan

Status: complete.

- Audited the four supplied papers, SOTA review, source, configs, saved summaries, and snapshot.
- Added project requirements, staged RAMPS plan, manager contract, progress ledger, and atomic task file.
- Corrected root .gitignore so project docs/configs/scripts can be tracked.
- No Python source, training, or new Test evaluation was performed in that step.

### TASK-000-A — Verify user-restored audio provider

Status: partial.

Observed changes:

- src/fusion/models/av_sync_mamba_segment.py restored from the historical snapshot with additional autocast handling;
- src/fusion package initializers restored;
- src/chimera_plugin.py explicitly imports fusion.models.av_sync_mamba_segment;
- src/audio has no diff.

Verification:

- Commands used:
  - env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -c <plugin/registry assertions>;
  - env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -c <synthetic forward/loss/backward>;
  - .venv/bin/chimera-ml validate-config -c configs/audio_experiments/wsm_audio_mamba_multitask.yaml;
  - .venv/bin/chimera-ml validate-config -c configs/audio_experiments/wsm_avsyncmamba_multitask.yaml;
  - .venv/bin/chimera-ml registry list;
  - python3 -m py_compile on the restored provider, audio model, and plugin;
  - git diff --check and git diff -- src/audio.
- plugin registration emitted no warnings;
- MODELS contains audio_mamba_segment_model and av_sync_mamba_segment_model;
- DATAMODULES contains wsm_audio_segment_datamodule;
- LOSSES contains wsm_audio_loss;
- required audio callbacks are registered;
- AudioMambaSegmentModel instantiated with the Transformer path;
- synthetic CPU forward/loss/backward passed: preds shape (4, 2), loss 0.7533376813, all gradients finite;
- both supplied YAML files pass chimera-ml validate-config structural validation;
- git diff --check passed before documentation translation;
- git diff -- src/audio was empty;
- no training and no new Test evaluation ran.

Assessment:

- the broken audio import/registry blocker is fixed;
- the Stage 0 gate is not complete because the canonical audio config is absent;
- the legacy AV config must not be treated as runnable or as the new fusion baseline.

Recommended next task: TASK-000B in NEXT_TASK.md.


### TASK-000B — Canonicalize the frozen audio baseline

Status: complete.

Changed files:

- configs/wsm_mm_pd_dep_v1/audio/00_frozen_baseline.yaml;
- scripts/run_audio_experiments.sh (default BASE now points to the canonical config);
- docs/PROGRESS_EN.md.

Implementation:

- Copied the verified source config from configs/audio_experiments/wsm_audio_mamba_multitask.yaml.
- Set experiment_name to wsm_mm_pd_dep_v1 and run_name to frozen_audio_wavlm_l9_pool4.
- Preserved WavLM-base-plus layer 9, temporal_pool 4, Transformer hidden_dim 192 / 3 layers / 4 heads / sequence_steps 128 / dropout 0.25, label_smoothing 0.02, focal_gamma 1.0, aux_weight 0.10, AdamW lr 1e-4 / weight_decay 0.01, and seed 42.
- Used the registered wsm_segment_metrics_callback and retained checkpoint, snapshot, summary, early stopping, console-file, and MLflow instrumentation.
- checkpoint_callback and early_stopping_callback both monitor dev/mean_score in max mode.
- The supplied AV config was not modified; it remains legacy/non-runnable as documented above.

Exact verification commands/results:

- .venv/bin/chimera-ml validate-config -c configs/wsm_mm_pd_dep_v1/audio/00_frozen_baseline.yaml — passed: Config is valid.
- PYTHONPATH=src .venv/bin/python -c '<config/callback/logger/registry assertions and synthetic smoke>' — passed: all referenced audio registry keys exist; required callbacks/loggers and monitor values asserted; synthetic CPU forward/loss/backward passed; no training/Test evaluation.
- Plugin registration in the same command emitted no project-module warning. It emitted only the existing PyTorch nested-tensor warning.
- git diff --check — passed.
- git diff -- src/audio — empty.
- git status --short and full diff audit — only the canonical config, launcher default, and this progress entry are in scope.

No training ran. No Test metrics were inspected or used.

Recommended next atomic task: begin Stage 1 manifest/partial-label contract only after manager approval; do not advance within this task.


### TASK-001A - Audit manifest sources and freeze the Stage 1 data contract

Status: partial complete for this audit task; Stage 1 remains partial and no production manifest was created.

Changed files:

- scripts/common/audit_manifest_sources.py;
- docs/PROGRESS_EN.md.

Audit source evidence:

- Depression sources:
  - train: /media/maxim/Databases/WSM_NEW/depression/train_labels_segments_min_filtered.csv; comma-delimited; columns video_id, diagnosis, segment_file; 3715 raw rows, 291 unique video IDs.
  - dev: /media/maxim/Databases/WSM_NEW/depression/dev_labels_segments_min_filtered.csv; comma-delimited; columns video_id, diagnosis, segment_file; 624 raw rows, 621 after the existing three bad-segment exclusions, 56 unique video IDs.
  - test: /media/maxim/Databases/WSM_NEW/depression/test_labels_segments_min_filtered_mishas.csv; semicolon-delimited; columns video_id, diagnosis, segment_file, soft_filter, hard_filter, diagnosis_2; 827 rows, 55 unique video IDs.
- Parkinson sources:
  - train: /media/maxim/Databases/WSM_NEW/parkinson/train_labels_segments_min_filtered.csv; comma-delimited; columns video_id, diagnosis, segment_file; 2736 rows, 272 unique video IDs.
  - dev: /media/maxim/Databases/WSM_NEW/parkinson/dev_labels_segments_min_filtered.csv; comma-delimited; columns video_id, diagnosis, segment_file; 312 rows, 44 unique video IDs.
  - test: /media/maxim/Databases/WSM_NEW/parkinson/test_labels_segments_min_filtered_mishas.csv; semicolon-delimited; columns video_id, diagnosis, segment_file, soft_filter, hard_filter, diagnosis_2; 537 rows, 46 unique video IDs.

Identity and split evidence:

- Candidate segment identity is corpus + video_id + segment_file. There were 8748 candidate rows, 8748 unique candidates, and zero collisions.
- Raw video-ID overlap before the existing priority rule: train/dev 7, train/test 2, dev/test 0.
- After the existing segment-index priority rule test > dev > train by video_id: train/dev/test video overlap is zero for every pair; retained row counts are train 6325, dev 933, test 1364.
- No authoritative speaker identifier was found in the six raw index schemas or 776 adjacent JSON metadata files. speaker_id is unresolved; fallback_to_video_id is false.

Modality source evidence:

- audio_available is resolved by root/corpus/{split}_labels/{video_id}/segments/{segment_stem}.wav; 8622/8622 retained rows exist.
- video_available is resolved by root/corpus/{split}_labels/{video_id}/segments/{segment_file}; 8622/8622 retained rows exist.
- text_available is resolved only as an adjacent video-level transcript at root/corpus/{split}_labels/{video_id}/{video_id}.txt; 8549/8622 exist. Segment-level text alignment is not established.
- description_available is unresolved/unavailable: no authoritative description or semantic-feature source exists.
- The proposed canonical mapping contains exactly the required 13 fields. Depression rows map raw diagnosis to y_depression and set y_parkinson to null with observed_depression=true and observed_parkinson=false. Parkinson rows do the converse. Unknown labels remain null/unknown, never negative; no pseudo-labels are created.

Exact verification commands and results:

- python3 -m py_compile scripts/common/audit_manifest_sources.py - passed.
- PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python scripts/common/audit_manifest_sources.py --data-root /media/maxim/Databases/WSM_NEW --output /tmp/wsm_stage1_manifest_source_audit.json - passed; wrote the JSON audit to /tmp only and made no dataset-root writes.
- The required JSON assertions - passed: manifest fields, label contract, unresolved speaker fallback protection, and false Test prediction/performance inspection flags.
- git diff --check - passed.
- git diff -- src/audio - empty.
- No model predictions or performance metrics were inspected. No production manifest or datamodule was created.

Recommended next atomic task: implement the canonical Stage 1 manifest from this audited source contract only after manager review of the unresolved speaker_id and segment-level text/description availability decisions.


### TASK-001B - Build the canonical partial-label segment manifest

Status: implementation complete; Stage 1 remains partial.

Branch: codex/task-001b.
Implementation commit SHA: 598e154809a893df9c64e56c6f2dd936801a2f4c.
Final branch HEAD: 15bf749e00c63063af49cd6a2bb6eb548313b2a8.
Push result: successful: origin/codex/task-001b created and pushed.
Manager integration: PR #1 merged to main as a2f20fb9b7b9658ae0a402533058563abc11bffb.

Changed files:

- src/common/data/__init__.py;
- src/common/data/wsm_manifest.py;
- scripts/common/build_wsm_manifest.py;
- docs/PROGRESS_EN.md.

Implementation facts:

- Reusable builder uses the audited CSV sources, delimiters, BAD_SEGMENTS exclusions, and unchanged test > dev > train video-level priority rule.
- Canonical output has exactly 13 columns in the required order.
- segment_id is a deterministic JSON-array encoding of [corpus, video_id, segment_file], with uniqueness asserted.
- speaker_id is null for every row; speaker_independence_verified=false; video_id is never used as a speaker fallback.
- Depression rows have only y_depression observed; Parkinson rows have only y_parkinson observed. Unobserved disease values are blank/null in CSV, never 0; no pseudo-labels are created.
- audio_available and video_available use only non-empty audited segment WAV/MP4 files.
- text_available and description_available are false for every row. Video-level transcript coverage is retained only as audit evidence.
- CLI refuses existing outputs without --overwrite and refuses writes inside the dataset root, src/, configs/, or docs/.
- No datamodule, registry entry, or plugin change was made.

Manifest and audit results:

- 8622 total rows; train=6325, dev=933, test=1364.
- Candidate segment identity is unique: 8622 unique IDs, zero collisions.
- Post-rule video split overlap is zero for train/dev, train/test, and dev/test.
- speaker_id null count is 8622; speaker_independence_verified=false.
- Audio availability: 8622 true, 0 false.
- Video availability: 8622 true, 0 false.
- Text availability: 0 true, 8622 false; video-level transcript source-only coverage is 8549/8622.
- Description availability: 0 true, 8622 false.
- Disease counts by corpus/split are recorded in the machine-readable audit. Observed depression counts are depression train 1431 positive / 2229 negative, dev 315 / 306, test 335 / 492. Observed Parkinson counts are Parkinson train 1058 positive / 1607 negative, dev 105 / 207, test 133 / 404. Unknown counts are Parkinson target 3660 / 621 / 827 on depression train/dev/test and depression target 2665 / 312 / 537 on Parkinson train/dev/test.
- Manifest SHA-256: e236e534eae3049b41ab134ebee7379c87a47bb8136102cf36d3c6df5c6c99bc.

Exact verification commands and results:

- python3 -m py_compile src/common/data/__init__.py src/common/data/wsm_manifest.py scripts/common/build_wsm_manifest.py - passed.
- rm -f /tmp/wsm_stage1_manifest.csv /tmp/wsm_stage1_manifest_audit.json - passed.
- PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python scripts/common/build_wsm_manifest.py --data-root /media/maxim/Databases/WSM_NEW --manifest-output /tmp/wsm_stage1_manifest.csv --audit-output /tmp/wsm_stage1_manifest_audit.json - passed; wrote 8622 rows and audit JSON.
- Required pandas/JSON manifest assertions - passed, including CSV null round-trip, schema/order, row/split counts, masks, availability, fingerprint, and no Test/model-selection flags.
- git diff --check - passed.
- git diff -- src/audio - empty.
- No Test predictions or performance metrics were inspected. No training or model selection ran. No datamodule was created. Dataset source files were not modified.

Stage 1 remains partial because authoritative speaker identity and segment-level text alignment remain unresolved.

Recommended next atomic task: audit and implement the separate DEV/Test manifest-consumer contract or datamodule only after manager approval; do not infer speaker identity or segment text alignment.


### TASK-001C - Implement the canonical manifest consumer and separate DEV/Test DataModule

Status: implementation complete; Stage 1 remains partial.

Branch: codex/task-001c.
Implementation commit SHA: 22d31c682bb49747c806eb99366908a82ad7c39c.
Final branch HEAD: 7aea0afad9df047a1050c81f7b0d1a48a2b318a8.
Push result: successful: origin/codex/task-001c created and pushed.
Manager integration: PR #2 merged to main as c4b00f78987c010442fd08a0c1771f174a3c762a.

Changed files:

- src/fusion/data/__init__.py;
- src/fusion/data/wsm_manifest_datamodule.py;
- src/chimera_plugin.py;
- docs/PROGRESS_EN.md.

Implementation facts:

- Added WSMManifestDataset and WSMManifestDataModule consuming the canonical manifest builder or a supplied canonical CSV.
- Registered the DataModule as wsm_manifest_datamodule and imported it explicitly from src/chimera_plugin.py.
- train_dataset contains only train rows; val_dataset contains only DEV rows; test_dataset contains only Test rows. Test is never injected into validation and no TEST_NONE/SOFT/HARD filtering exists.
- Each sample exposes segment/video/speaker/corpus/split metadata, targets [depression, parkinson], observed_mask [depression, parkinson], and modality_available [audio, video, text, description].
- Unknown targets are represented as NaN in tensor batches and are always paired with observed_mask=false; observed targets are validated as finite 0/1. This prevents unknown disease labels from becoming supervised negatives.
- Canonical text and description availability remain false. speaker_id remains null and speaker_independence_verified=false.
- No model, loss, metric, callback, optimizer, training config, feature extraction, or unrelated registry component was added.

Verification results:

- train/dev/test rows: 6325 / 933 / 1364.
- val_dataset contains only split=dev; test_dataset contains only split=test; no Test object is inserted into validation.
- Smoke batch shapes: targets [8, 2], observed_mask [8, 2], modality_available [8, 4].
- Depression samples use mask [true, false]; Parkinson samples use [false, true].
- Unobserved target positions are NaN, not 0, and observed positions are finite 0/1.
- Text and description masks are false; speaker_id is null.

Exact verification commands and results:

- python3 -m py_compile src/fusion/data/__init__.py src/fusion/data/wsm_manifest_datamodule.py src/chimera_plugin.py - passed.
- Required registry command reached an installed Chimera API incompatibility: Registry does not implement Python membership for the requested assertion. The corrected equivalent using DATAMODULES.keys() passed; no project-module warning for fusion.data.wsm_manifest_datamodule was emitted.
- PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python corrected registry smoke - passed.
- PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python dataset/collate smoke - passed.
- git diff --check - passed.
- git diff -- src/audio - empty.
- No Test predictions or performance metrics were inspected. No training, tuning, checkpoint selection, threshold selection, or model selection ran.
- No source manifest files were modified and no later modeling/loss task was started.

Stage 1 remains partial because authoritative speaker identity and segment-level text alignment remain unresolved.

Recommended next atomic task: implement the masked sparse loss/data contract consumer only after manager approval; do not add pseudo-labeling, text alignment, or model training in that task.


### TASK-001D - Implement the observed-label-only masked sparse loss contract

Status: implementation complete; Stage 1 remains partial.

Branch: codex/task-001d.
Implementation commit SHA: 02df3e8524964bb88edfa0b438048202e429434c.
Final branch HEAD: 90094215b3a6aafe98b5a6dceb24623f1a9de2b9.
Push result: successful: origin/codex/task-001d created and pushed.
Manager integration: PR #3 merged to main as d5bc6c5b128a5ca99070b04744dd1adc30a34390.

Changed files:

- src/common/loss/__init__.py;
- src/common/loss/wsm_masked_sparse_loss.py;
- src/chimera_plugin.py;
- docs/PROGRESS_EN.md.

Implementation facts:

- Added BaseLoss-compatible wsm_masked_sparse_loss registered in LOSSES.
- The loss computes independent binary BCE-with-logits only on observed elements:
  L_obs = sum(mask * BCEWithLogits(logit, target)) / (sum(mask) + eps).
- Logits, targets, and observed_mask must all have shape [B, 2]. Observed targets are validated as finite binary 0/1.
- Masked target positions are selected out before target validation and BCE evaluation, so NaN unknown placeholders cannot propagate into the scalar loss.
- Masked-out logits receive no BCE operation and therefore receive exactly zero gradient; no class weighting, corpus weighting, focal term, smoothing, task balancing, pseudo-labeling, or reliability logic was added.
- The callable supports both a plain manifest-collate dict and the installed Chimera Batch API, including Batch.get_masks("observed_mask").
- Zero observed elements raise a clear ValueError rather than returning zero.

Exact verification commands and results:

- python3 -m py_compile src/common/loss/__init__.py src/common/loss/wsm_masked_sparse_loss.py src/chimera_plugin.py - passed.
- PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python registry smoke - passed; LOSSES.keys() contains wsm_masked_sparse_loss and no project-module warning was emitted.
- PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python masked loss forward/backward smoke - passed; finite scalar and observed gradients, exactly zero masked gradients, masked-placeholder invariance, observed-target sensitivity, and zero-observed failure.
- PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python ModelOutput/Batch compatibility smoke - passed.
- git diff --check - passed.
- git diff -- src/audio - empty.
- No Test predictions or performance metrics were inspected. No training, checkpoint selection, threshold selection, or model selection ran.
- No manifest, datamodule, model, or training configuration was changed.

Stage 1 remains partial because authoritative speaker identity is unresolved and the Stage 1 gate is not manager-closed.

Recommended next atomic task: implement the next manager-approved baseline integration contract only; do not add pseudo-labeling, reliability weighting, or model/training changes in that task.


### TASK-001E - Establish the authoritative speaker-map contract and Stage 1 leakage gate

Status: implementation complete; Stage 1 remains partial/blocked.

Branch: codex/task-001e.
Implementation commit SHA: fefc24b2c01b5cf95fb80f67a3a6b8a6ccf6aa44.
Final branch HEAD: 749f4230391466825e88bd843d93de05de6c2e5d.
Push result: successful: origin/codex/task-001e created and pushed.
Manager integration: PR #4 merged to main as 38bab90d7325ea58349cc57244d1fe1d526cbd25.

Changed files:

- src/common/data/wsm_speaker_map.py;
- scripts/common/audit_speaker_map.py;
- docs/PROGRESS_EN.md.

Implementation facts:

- Added strict authoritative speaker-map CSV validation with exactly these columns: corpus, video_id, speaker_id, source_reference.
- Empty fields, invalid corpus values, duplicate/conflicting (corpus, video_id) pairs, and malformed maps fail clearly.
- No fallback to video_id or any speaker inference is implemented.
- The reusable audit reports canonical coverage, unmapped/extra pairs, conflicts, video overlap, speaker overlap, manifest fingerprint, and the combined Stage 1 gate.
- Without a supplied map, the CLI emits status=unresolved, speaker_independence_verified=false, video_independence_verified=true from the canonical rows, stage1_split_gate_passed=false, and next_required_evidence=authoritative speaker map.
- A supplied map verifies speaker independence only when coverage is complete, conflicts are zero, speaker overlap is zero, and video overlap remains zero.

Current unresolved gate result:

- Canonical video split independence remains verified with zero train/dev, train/test, and dev/test video overlap.
- No real authoritative speaker map was available or committed.
- Current machine-readable gate status: unresolved.
- speaker_independence_verified=false.
- stage1_split_gate_passed=false.
- Synthetic maps were used only for contract tests and are not project evidence.

Exact verification commands and results:

- python3 -m py_compile src/common/data/wsm_speaker_map.py scripts/common/audit_speaker_map.py - passed.
- rm -f /tmp/wsm_speaker_gate_unresolved.json - passed.
- PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python scripts/common/audit_speaker_map.py --data-root /media/maxim/Databases/WSM_NEW --output /tmp/wsm_speaker_gate_unresolved.json - passed; emitted unresolved JSON without modifying the dataset.
- Required unresolved JSON assertions - passed: unresolved status, false speaker gate, true video independence, required schema, next evidence, and all false Test/model-selection flags.
- Synthetic speaker-map contract smoke - passed:
  - complete non-leaking synthetic map: status=verified and both independence checks/gate true;
  - leaking synthetic map: train/dev speaker overlap detected and gate false;
  - incomplete synthetic map: unmapped pair detected and speaker gate false;
  - conflicting synthetic map: validator raised SpeakerMapError.
- git diff --check - passed.
- git diff -- src/audio - empty.
- No Test predictions or performance metrics were inspected. No training, tuning, threshold selection, or model selection ran.

Stage 1 remains partial/blocked until a real authoritative speaker map is supplied and passes the leakage gate. The synthetic identities are not authoritative evidence.

Recommended next atomic task: manager review or provision of the authoritative speaker map; do not infer speakers or begin Stage 2.


### MANAGER-DECISION-001 — Accept current split as speaker-independent by owner contract

Status: accepted.

- The dataset owner explicitly stated that no speaker_id is available and directed the project to use the existing train/dev/test split as already speaker-independent.
- speaker_id therefore remains null/unavailable in the canonical manifest. No speaker identity is inferred.
- TASK-001E's speaker-map tooling remains available as optional future audit infrastructure but is no longer a Stage 1 gate requirement.
- TASK-001F is superseded before implementation and must not be executed.
- Video-level split independence remains empirically verified with zero overlap after the existing split rule.
- Speaker independence is an owner-provided dataset assumption and must be described as such in research artifacts; it is not independently measured evidence.
- Stage 1 gate is accepted as complete under this explicit owner decision.
- Final Test remains locked.

Recommended next atomic task: begin Stage 2 with reproducible video input/preprocessing and cache-contract audit only; do not train or compare video models yet.

### TASK-002A - Define the reproducible V1 video preprocessing and cache contract

Status: implementation complete; no training or Test evaluation ran.

Changed files:

- src/video/__init__.py;
- src/video/features/__init__.py;
- src/video/features/clip_video_features.py;
- scripts/video/extract_clip_video_features.py;
- docs/PROGRESS_EN.md.

Implementation facts:

- Added deterministic uniform full-segment sampling with inclusive endpoints and default target_frames=32.
- Added frozen, eval-mode CLIP extraction with default openai/clip-vit-base-patch32 and revision main; model loading is local-files-only by default and failures are reported rather than silently downloaded or converted into fake features.
- The extractor accepts RGB frames only at the processor boundary, returns temporal features [T,D] and a boolean valid_mask [T], and has explicit zero/unreadable-frame failure results.
- Cache fingerprints include manifest fingerprint, segment identity, source path, model/revision, preprocessing version, sampling method, processor identity, target frame count, and available package versions.
- Cache artifacts contain segment_id, source_path, features, valid_mask, model/revision, preprocessing metadata, and cache_fingerprint. Failed extractions cannot be serialized as successful artifacts.
- The CLI defaults to building the canonical manifest from data-root when --manifest-path is omitted, processes only video_available rows, records the limit and fingerprints, refuses protected src/configs/docs output locations, and emits machine-readable success/failure reports.
- No registry, model config, training, label, split, corpus, diagnosis, or task metadata is passed to the encoder.

Exact verification commands and results:

- PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m py_compile src/video/features/clip_video_features.py scripts/video/extract_clip_video_features.py - passed.
- PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 deterministic helper smoke - passed for 1-frame input, 100-to-32 sampling, endpoint coverage, and fingerprint sensitivity.
- .venv/bin/python synthetic extractor smoke - passed; synthetic forward produced [32,5] features and boolean [32] validity, and unreadable input returned failure with no features or mask.
- .venv/bin/python scripts/video/extract_clip_video_features.py --data-root /media/maxim/Databases/WSM_NEW --manifest-path /tmp/wsm_stage1_manifest.csv --cache-root /tmp/wsm_video_cache_002a --report-output /tmp/wsm_video_report_002a.json --limit 1 - passed structurally; report recorded limit=1, success_count=0, failure_count=1 because the real extraction path was unavailable, with no fake artifact created.
- CLI --help exposed all required arguments.
- Protected-output refusal smoke - passed.
- git diff --check - passed.
- git diff -- src/audio - empty; src/audio remained unchanged.
- No training, checkpoint selection, or Test predictions/metrics were run or inspected.
- Ruff was unavailable at .venv/bin/ruff; no dependency was installed.

Metrics/artifacts: one machine-readable structural report at /tmp/wsm_video_report_002a.json; no cache artifact was written for the failed extraction. The canonical manifest fingerprint recorded by the CLI was e236e534eae3049b41ab134ebee7379c87a47bb8136102cf36d3c6df5c6c99bc.

Deviations/blockers: the structural CLI run did not produce a feature cache because the selected real row failed extraction; this is an explicit failure report, not a successful baseline. No video model comparison was attempted.

Recommended next atomic task: manager review and, if accepted, proceed to the next Stage 2 video task without training or Test evaluation in this task.

### TASK-002B - Integrate the DEPART YOLOv8 body-ROI stage into the V1 temporal CLIP cache

Status: implementation complete; Stage 2 remains partial.

Branch: codex/task-002b.
Implementation commit: 1fa62cf.
Push result: pending final push.

Changed files:

- src/video/features/yolov8_body_roi.py;
- src/video/features/clip_video_features.py;
- src/video/features/__init__.py;
- scripts/video/extract_clip_video_features.py;
- pyproject.toml;
- docs/PROGRESS_EN.md.

Implementation facts:

- Added a local-only YOLOv8 single-human-body adapter. It requires an explicitly supplied local weights file, computes its SHA-256, never downloads weights, and supports injected detector objects for smoke tests.
- Implemented deterministic detection selection: confidence threshold, highest confidence, larger area, then lexicographic coordinates; boxes are clipped to image bounds and invalid-area boxes are rejected.
- Changed the V1 default to 60 uniformly sampled temporal positions. Normal CLI execution requires --yolo-weights and uses confidence=0.5, IoU=0.5, imgsz=640. Raw-frame extraction remains explicit --raw-frame-debug only.
- ROI mode preserves all temporal positions and chronological order. Valid detections are encoded from body crops; invalid positions are zero-padded only after feature dimension is known and receive valid_mask=false. All-missing detections fail with no successful artifact.
- Cache fingerprints now include detector identity, weights SHA-256, confidence, IoU, image size, and ROI policy version in addition to the existing manifest/model/sampling fields.
- Successful artifacts include detector metadata, sampled indices, selected boxes/confidences, valid mask, valid detection count, and detection coverage. Reports separate no-body-detected from video/read/model-load failures.
- Added ultralytics to pyproject.toml as a declared dependency only; no installation was run.

Exact verification commands and results:

- python3 -m py_compile src/video/features/yolov8_body_roi.py src/video/features/clip_video_features.py src/video/features/__init__.py scripts/video/extract_clip_video_features.py - passed.
- PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python deterministic body-selection smoke - passed; same-confidence larger box selected and sub-threshold detection rejected.
- .venv/bin/python injected temporal ROI/CLIP smoke - passed; 60 positions, interleaved valid/invalid detections, [60,4] features, bool [60] mask, exact zero padding, 40/60 coverage, all-invalid failure, and no model download.
- .venv/bin/python fingerprint sensitivity smoke - passed for weights SHA-256, confidence, IoU, image size, target frames, and ROI policy changes.
- PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python scripts/video/extract_clip_video_features.py --help - passed; required YOLO arguments and 60-frame default exposed.
- Missing-weights CLI refusal smoke - passed; missing local weights returned a clear FileNotFoundError and no extraction ran.
- ultralytics runtime was available in .venv; no local detector checkpoint was found and no weights were downloaded.
- git diff --check - passed.
- git diff -- src/audio - empty; src/audio remained unchanged.
- No full-dataset feature extraction, training, model selection, or Test predictions/metrics ran.

Deviations/blockers: real YOLO weights were unavailable, so verification used injected detector/encoder objects only. The implementation is ready for a later manager-authorized cache extraction once a checkpoint path is provisioned.

Recommended next atomic task: manager review/provision of the authoritative YOLOv8 body checkpoint, followed by a limited cache extraction audit; do not run full extraction or model training yet.
