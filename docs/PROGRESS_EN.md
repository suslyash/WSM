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

Manager integration: PR #5 merged to main as fb93b31527bc4f1be5dac35c5194a1f266bb6aec.

Recommended next atomic task: integrate the DEPART YOLOv8 single-class human-body ROI stage into the temporal CLIP preprocessing/cache pipeline before any full-cache extraction or video-model training.


### MANAGER-DECISION-002 — Align Stage 2 V1 with the DEPART body-ROI pipeline

Status: accepted after TASK-002A completed under its original scope.

- TASK-002A is accepted as the reusable deterministic raw-frame temporal CLIP/cache foundation that was originally assigned.
- The dataset owner clarified during TASK-002A execution that the intended DEPART-like V1 must include YOLO-based human-body region extraction before CLIP, while preserving the ordered frame sequence for temporal modeling.
- The DEPART article specifies a YOLOv8 single-class human-body detector applied per sampled frame, followed by body-region cropping/resizing before CLIP visual encoding and Transformer temporal modeling.
- This clarification is not treated as a retroactive TASK-002A failure.
- The next atomic task must add the YOLOv8 body-ROI stage and revise cache fingerprints/artifacts/reporting accordingly before full-dataset feature extraction.
- No full-dataset video cache extraction is authorized yet.
- Final Test remains locked.


### TASK-002B - Integrate the DEPART YOLOv8 body-ROI stage into the V1 temporal CLIP cache

Status: implementation complete; Stage 2 remains partial.

Branch: codex/task-002b.
Implementation commit: 1fa62cf.
Push result: successful; origin/codex/task-002b created and pushed.

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

Manager integration: original PR #6 conflicted after manager-side main updates and was closed unmerged. Accepted TASK-002B was replayed onto current main via manager integration PR #7 and merged as c23df08c46b5ab98c8d530581535d552edd9406f.

Recommended next atomic task: provision the pinned DEPART-referenced YOLOv8 human-body checkpoint and run a limited real cache extraction audit; do not run full extraction or model training yet.

### TASK-002C - Provision the pinned DEPART YOLOv8 checkpoint and run a limited real extraction audit

Status: partial/blocked; no successful real cache artifact was produced because the frozen CLIP processor/model was unavailable locally and this task did not authorize downloading another model.

Branch: codex/task-002c.
Implementation commit: 45a0497.
Push result: successful; origin/codex/task-002c created and pushed.

Changed files:

- scripts/video/audit_depart_real_extraction.py;
- docs/PROGRESS_EN.md.

Pinned checkpoint provenance:

- upstream repository: J3lly-Been/YOLOv8-HumanDetection;
- upstream commit: ce2aae2e821100aee58ce2e7f75994a7e6c2ab9e;
- upstream path: best.pt;
- Git blob SHA: afa44d4fcd0ff54691912bf8960d4fbb98ae1278;
- local path: /media/maxim/Programs/Models/WSM/depart_yolov8/best.pt;
- local size: 6259289 bytes;
- local SHA-256: a6aead7bf0eccb35bd56731bfaa6ea19a4645a66150d2d0b19dd3fb1b116ef43;
- the checkpoint binary is not tracked or committed.

Implementation/audit facts:

- Added a fixed-sample helper that builds the canonical manifest in memory, deterministically selects exactly three video-available train rows and three dev rows, processes zero Test rows, and cannot run the full manifest by default.
- The selected train segment IDs were ["depression","-7UpRmNVJzQ","-7UpRmNVJzQ_001.mp4"], ["depression","-7UpRmNVJzQ","-7UpRmNVJzQ_002.mp4"], and ["depression","-7UpRmNVJzQ","-7UpRmNVJzQ_003.mp4"].
- The selected dev segment IDs were ["depression","3SYkj_mya6A","3SYkj_mya6A_001.mp4"], ["depression","3SYkj_mya6A","3SYkj_mya6A_002.mp4"], and ["depression","3SYkj_mya6A","3SYkj_mya6A_003.mp4"].
- Source frame counts/temporal lengths were train 376/60, 293/60, 243/60 and dev 394/60, 376/60, 147/60. No T<60 blocker appeared.
- All six attempts reached the real pipeline and failed with category video_read_or_model_load because the local-only CLIP image processor for openai/clip-vit-base-patch32 was unavailable. Success count=0, failure count=6, no-body-detected count=0, other extraction failures=6, mean/min/max successful detection coverage unavailable (0.0 report default), and no cache artifacts were written.
- Runtime versions recorded in the report: ultralytics 8.4.157, transformers 5.14.1, torch 2.10.0. CUDA was used and available.
- The first detector initialization emitted an Ultralytics auto-install warning and installed dill despite no explicit install command; the helper was then corrected to set YOLO_AUTOINSTALL=false and the fixed audit was rerun. No model weights other than the authorized pinned detector were downloaded. This environment-side runtime behavior is recorded as a deviation.

Exact verification commands/results:

- python3 -m py_compile scripts/video/audit_depart_real_extraction.py - passed.
- curl from the pinned raw GitHub commit to /media/maxim/Programs/Models/WSM/depart_yolov8/best.pt - downloaded; Git blob verification produced afa44d4fcd0ff54691912bf8960d4fbb98ae1278.
- sha256sum /media/maxim/Programs/Models/WSM/depart_yolov8/best.pt - produced a6aead7bf0eccb35bd56731bfaa6ea19a4645a66150d2d0b19dd3fb1b116ef43.
- CUDA availability check - passed; CUDA=True.
- PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python scripts/video/audit_depart_real_extraction.py --data-root /media/maxim/Databases/WSM_NEW --yolo-weights /media/maxim/Programs/Models/WSM/depart_yolov8/best.pt --cache-root /tmp/wsm_depart_002c/cache --report-output /tmp/wsm_depart_002c/report.json --per-split 3 --target-frames 60 --device cuda - completed the fixed six-row audit and returned exit code 2 because success_count=0.
- Report validation against /tmp/wsm_depart_002c/report.json - passed for selected_count=6, train/dev selection, zero Test usage, pinned Git blob, six temporal lengths of 60, and no short-video blockers.
- git diff --check - passed.
- git diff -- src/audio - empty; src/audio remained unchanged.
- No full extraction, training, model selection, Test rows, Test predictions, or Test metrics ran.

Metrics/artifacts: machine-readable report at /tmp/wsm_depart_002c/report.json; zero successful cache artifacts. Stage 2 remains partial.

Manager integration: PR #8 merged to main as d73b2aeac6692356d586cd5f33d72774fd0239d0. TASK-002C is accepted as partial/blocked evidence, not as a successful real extraction pass.

Recommended next atomic task: provision/cache the approved pinned CLIP model revision through an explicitly authorized model-provisioning task, then rerun only this fixed six-segment audit; do not start full extraction or training.

### TASK-002D - Provision the pinned CLIP revision and rerun the fixed six-segment real audit

Status: blocked; the pinned model provisioned and loads locally, but the unchanged TASK-002B extractor has a concrete Transformers compatibility blocker and no successful artifact was produced.

Branch: codex/task-002d.
Implementation commit: 06fbb97.
Push result: successful; origin/codex/task-002d created and pushed.

Tracked change:

- docs/PROGRESS_EN.md only.

Pinned CLIP provisioning:

- repository: openai/clip-vit-base-patch32;
- revision: b97b0100e55e367c057773c2a614676470b0d575;
- architecture: CLIP ViT-B/32;
- HF_HOME: /media/maxim/Programs/Models/WSM/huggingface;
- resolved snapshot: /media/maxim/Programs/Models/WSM/huggingface/hub/models--openai--clip-vit-base-patch32/snapshots/b97b0100e55e367c057773c2a614676470b0d575;
- cached files: config.json, merges.txt, model.safetensors, preprocessor_config.json, pytorch_model.bin, special_tokens_map.json, tokenizer.json, tokenizer_config.json, vocab.json;
- local-only CLIPProcessor load: passed;
- local-only CLIPModel load: passed; model.eval() and projection dimension 512.
- HF_HUB_CACHE had to be explicitly set to /media/maxim/Programs/Models/WSM/huggingface/hub because the environment default pointed elsewhere; no source semantics changed.

Audit facts:

- Existing pinned YOLO checkpoint SHA-256 recheck passed: a6aead7bf0eccb35bd56731bfaa6ea19a4645a66150d2d0b19dd3fb1b116ef43.
- The exact TASK-002C six rows were attempted: train ["depression","-7UpRmNVJzQ","-7UpRmNVJzQ_001.mp4"], ["depression","-7UpRmNVJzQ","-7UpRmNVJzQ_002.mp4"], ["depression","-7UpRmNVJzQ","-7UpRmNVJzQ_003.mp4"]; dev ["depression","3SYkj_mya6A","3SYkj_mya6A_001.mp4"], ["depression","3SYkj_mya6A","3SYkj_mya6A_002.mp4"], ["depression","3SYkj_mya6A","3SYkj_mya6A_003.mp4"].
- CUDA was used. All six preflight temporal lengths remained 60: train source frames 376, 293, 243; dev source frames 394, 376, 147.
- Test rows processed: zero. Success count=0, failure count=6, no-body-detected count=0, other extraction failures=6, detection coverage unavailable, and zero cache artifacts were produced.
- Exact blocker for every row: AttributeError: 'BaseModelOutputWithPooling' object has no attribute 'ndim'. The unchanged extractor fallback calls vision_model(...).pooler_output, but the installed pinned Transformers runtime returns a BaseModelOutputWithPooling object there. TASK-002D forbids source changes, so this was not repaired.
- No package auto-install was attempted in TASK-002D. YOLO_AUTOINSTALL=false was set.

Exact verification commands/results:

- HF_HOME=/media/maxim/Programs/Models/WSM/huggingface YOLO_AUTOINSTALL=false .venv/bin/python snapshot_download at the pinned revision - passed; resolved snapshot recorded above.
- HF_HOME=/media/maxim/Programs/Models/WSM/huggingface HF_HUB_CACHE=/media/maxim/Programs/Models/WSM/huggingface/hub YOLO_AUTOINSTALL=false .venv/bin/python local-only CLIPProcessor/CLIPModel load - passed.
- sha256sum /media/maxim/Programs/Models/WSM/depart_yolov8/best.pt - passed with the expected SHA-256.
- rm -rf /tmp/wsm_depart_002d followed by the exact fixed six-segment audit command with --model-revision b97b0100e55e367c057773c2a614676470b0d575 --target-frames 60 --device cuda - completed with exit code 2 and report status partial_blocked_no_success.
- Report validation passed for six selected rows, exact IDs, zero Test usage, target_frames=60, no short-video blockers, and unchanged YOLO SHA; success_count>=1 could not pass because the extractor blocker remains.
- git diff --check - passed.
- git diff -- src/audio - empty; src/audio remained unchanged.
- No full extraction, training, model selection, Test rows, Test predictions, or Test metrics ran.

Metrics/artifacts: /tmp/wsm_depart_002d/report.json records the six failures; /tmp/wsm_depart_002d/cache contains no successful cache artifact. Stage 2 remains partial.

Manager integration: PR #9 merged to main as 326da7ba5a4cdcdfceca90e3f6f3b61afef28ce3. TASK-002D is accepted as blocked evidence; CLIP provisioning succeeded but real extraction remains blocked by the identified output-unwrapping incompatibility.

Recommended next atomic task: repair only the Transformers CLIP output-unwrapping compatibility defect, then rerun the exact same fixed six-segment audit.

### TASK-002E - Fix Transformers CLIP output unwrapping and clear the fixed six-segment real audit

Status: implementation complete; Stage 2 remains partial pending later video-model work.

Branch: codex/task-002e.
Implementation commit: bd3e710.
Push result: successful; origin/codex/task-002e created and pushed.

Changed files:

- src/video/features/clip_video_features.py;
- docs/PROGRESS_EN.md.

Compatibility fix:

- Added normalize_clip_image_features with strict precedence: Tensor first, then image_embeds Tensor, then pooler_output Tensor; unsupported output types and non-rank-2 tensors raise clear RuntimeError exceptions.
- Applied the helper to both get_image_features and vision_model paths without changing projection, sampling, detector, temporal masking, cache fingerprint, or artifact semantics.
- In the real pinned Transformers runtime, model.get_image_features exists and returns BaseModelOutputWithPooling; its pooler_output is a Tensor with shape [B,512]. No double projection was introduced.

Pinned runtime/audit facts:

- CLIP repo/revision: openai/clip-vit-base-patch32 at b97b0100e55e367c057773c2a614676470b0d575.
- YOLO SHA-256 remained a6aead7bf0eccb35bd56731bfaa6ea19a4645a66150d2d0b19dd3fb1b116ef43.
- CUDA was used. The exact six segments were attempted: train ["depression","-7UpRmNVJzQ","-7UpRmNVJzQ_001.mp4"], ["depression","-7UpRmNVJzQ","-7UpRmNVJzQ_002.mp4"], ["depression","-7UpRmNVJzQ","-7UpRmNVJzQ_003.mp4"]; dev ["depression","3SYkj_mya6A","3SYkj_mya6A_001.mp4"], ["depression","3SYkj_mya6A","3SYkj_mya6A_002.mp4"], ["depression","3SYkj_mya6A","3SYkj_mya6A_003.mp4"].
- Test rows processed: zero. All six preflight temporal lengths remained 60.
- Success count=6, failure count=0. Detection coverage by segment: 0.9166666667, 1.0, 1.0, 0.9166666667, 0.9166666667, 1.0; mean=0.9583333333, min=0.9166666667, max=1.0.
- All successful artifacts validated with feature shape [60,512], bool valid_mask [60], exact zero invalid positions, finite valid positions, chronological sampled indices, pinned YOLO SHA, detector settings 0.5/0.5/640, pinned CLIP revision, and unique cache fingerprints.
- No package installation or auto-install was attempted. YOLO_AUTOINSTALL=false was set.

Exact verification commands/results:

- python3 -m py_compile src/video/features/clip_video_features.py - passed.
- CLIP output normalization regression smoke for Tensor, image_embeds, pooler_output, and unsupported output - passed.
- Local-only pinned CLIPProcessor/CLIPModel load - passed.
- Real runtime output inspection - passed; get_image_features returned BaseModelOutputWithPooling with Tensor pooler_output [1,512].
- sha256sum /media/maxim/Programs/Models/WSM/depart_yolov8/best.pt - passed with the expected SHA-256.
- Exact fixed six-segment audit with --target-frames 60, pinned CLIP revision, pinned YOLO, CUDA, and YOLO_AUTOINSTALL=false - passed with exit code 0 and success_count=6.
- Report/artifact validation against /tmp/wsm_depart_002e/report.json - passed; report status complete and all_success_artifacts_valid=true.
- git diff --check - passed.
- git diff -- src/audio - empty; src/audio remained unchanged.
- No full extraction, training, model selection, Test rows, Test predictions, or Test metrics ran.

Metrics/artifacts: six successful cache artifacts under /tmp/wsm_depart_002e/cache and machine-readable report at /tmp/wsm_depart_002e/report.json. These are external audit artifacts and are not committed.

Manager integration: PR #10 merged to main as c7b190e73ad2daca9caef75686ee8b75c104cf97. The V1 preprocessing/cache real-data gate is accepted as cleared.

Recommended next atomic task: implement/register the V1 DEPART-like temporal video model contract and verify synthetic forward/loss/backward on [B,60,512] cached-feature-shaped inputs; do not run full extraction or training yet.

### TASK-002F - Implement and register the V1 DEPART-like temporal video model contract

Status: implementation complete; Stage 2 remains partial.

Branch: codex/task-002f.
Implementation commit: 509b595.
Push result: successful; origin/codex/task-002f created and pushed.

Changed files:

- src/video/models/depart_v1.py;
- src/video/models/__init__.py;
- src/chimera_plugin.py;
- docs/PROGRESS_EN.md.

Implementation facts:

- Registered model key: wsm_video_depart_v1_model.
- Implemented the V1 path [B,60,512] -> LayerNorm/Linear/GELU/Dropout projection -> learned positional embeddings -> batch-first, norm-first TransformerEncoder -> LayerNorm -> valid-mask mean pooling -> two independent scalar heads.
- Outputs are direct logits [B,2] ordered [depression, parkinson]. ModelOutput.aux contains pooled features [B,H] and separate depression/parkinson task logits. No sigmoid, task_id selection, or three-class softmax exists.
- Context-aware factory uses data.video_feature_dim when supplied, defaults to 512, and rejects data.num_tasks other than 2.
- Input validation covers rank, floating-point type, dimensions, feature width, sequence length, mask conversion, and all-invalid samples. Masked temporal positions are excluded from Transformer attention and pooling.

Exact verification commands/results:

- python3 -m py_compile src/video/models/__init__.py src/video/models/depart_v1.py src/chimera_plugin.py - passed.
- Chimera registry smoke with warning capture - passed; wsm_video_depart_v1_model registered and no project-module warning emitted.
- Required synthetic forward/loss/backward smoke - passed; preds [4,2], pooled features [4,64], task logits [4], finite masked sparse loss 0.6393991708755493, and finite gradients.
- Mask invariance smoke - passed; changing masked frames to 1e6 did not change eval logits.
- Partial masks worked; all-invalid sample raised ValueError.
- Unknown NaN targets remained masked and were accepted by the observed-label-only loss.
- A benign PyTorch nested-tensor warning was emitted because norm_first=True; it does not affect correctness and no dependency was installed.
- git diff --check - passed.
- git diff -- src/audio - empty; src/audio remained unchanged.
- No cache extraction, training, model selection, Test rows, Test predictions, or Test metrics ran.

Metrics/artifacts: no training or cache artifacts were created. Stage 2 remains partial pending later data/model integration tasks.

Manager integration: PR #11 merged to main as 78aa592b0f8b068b12a75d1072386656d16dbc9f. The V1 registered model contract is accepted.

Recommended next atomic task: add a strict train/dev split filter to the video cache extractor and build the complete V1 cache for TRAIN+DEV only; Test must remain untouched.

### TASK-002G - Build the complete resumable V1 video feature cache for TRAIN+DEV only

Status: implementation complete; Stage 2 remains partial pending video training/integration.

Branch: codex/task-002g.
Implementation commit: final task-branch HEAD; verify with git rev-parse HEAD.
Push result: successful; origin/codex/task-002g created and pushed.

Changed files:

- scripts/video/extract_clip_video_features.py;
- scripts/video/audit_video_cache.py;
- docs/PROGRESS_EN.md.

Implementation facts:

- Added strict --splits parsing/filtering before source-path resolution or model inference. The authorized run used exactly train,dev; Test rows were never read, processed, or indexed.
- Added --resume with exact fingerprint/artifact validation for segment identity, [60,512] features, bool [60] masks, finite valid features, exact-zero invalid features, pinned YOLO SHA, detector settings, and pinned CLIP identity.
- Added atomic cache_index.jsonl updates after each row. Every selected row has exactly one extracted, reused, or failed record.
- Added independent scripts/video/audit_video_cache.py; it performs no YOLO/CLIP extraction and independently validates selected rows, artifacts, fingerprints, failures, coverage, and Test exclusion.

Persistent artifacts:

- cache root: /media/maxim/Programs/Features/WSM/video_depart_v1/cache;
- extraction report: /media/maxim/Programs/Features/WSM/video_depart_v1/extraction_report.json;
- cache index: /media/maxim/Programs/Features/WSM/video_depart_v1/cache/cache_index.jsonl;
- independent audit: /media/maxim/Programs/Features/WSM/video_depart_v1/cache_audit.json.

Full TRAIN+DEV execution results:

- requested splits: train,dev;
- expected counts: train=6325, dev=933, total=7258;
- extracted successes=6999;
- reused successes=51;
- failures=208;
- no-body-detected=96;
- other failures=112: invalid_cache=1 and invalid_extraction_contract=111;
- Test rows processed/indexed=0;
- pinned YOLO SHA-256: a6aead7bf0eccb35bd56731bfaa6ea19a4645a66150d2d0b19dd3fb1b116ef43;
- pinned CLIP revision: b97b0100e55e367c057773c2a614676470b0d575;
- target_frames=60 and detector settings remained confidence=0.5, IoU=0.5, imgsz=640.

Independent audit results:

- complete_for_requested_splits=true;
- missing_record_count=0;
- successful_artifacts_valid=true;
- cache_fingerprints_unique=true;
- success counts: train=6160, dev=890;
- explicit failures: train=165, dev=43;
- valid-artifact coverage: train count=6160, mean=0.8919561688, min=0.0166666667, max=1.0; dev count=890, mean=0.9023970037, min=0.0333333333, max=1.0; overall count=7050, mean=0.8932742317, min=0.0166666667, max=1.0;
- all successful artifacts validated as [60,512] with bool masks, exact-zero invalid positions, finite valid positions, correct detector/CLIP metadata, chronological sampling, and unique fingerprints.

Exact verification commands/results:

- python3 -m py_compile scripts/video/extract_clip_video_features.py scripts/video/audit_video_cache.py - passed.
- CLI --help showed --splits and --resume.
- Two-row train-only resumability smoke - passed: first run extracted 2; identical second run reused 2 with no Test processing.
- Full authorized TRAIN+DEV extractor with --splits train,dev --resume and no --limit - completed; final report selected_count=7258.
- Independent audit helper - passed with exit code 0 and complete=true.
- Required report assertions - passed for expected counts, zero Test rows, complete audit, valid artifacts, unique fingerprints, pinned YOLO SHA, pinned CLIP revision, target_frames, and Test firewall flags.
- No dependency installation occurred; YOLO_AUTOINSTALL=false was set.
- git diff --check - passed.
- git diff -- src/audio - empty; src/audio remained unchanged.
- No training, model selection, Test rows, Test predictions, or Test metrics ran.

Deviation: 111 rows were recorded as invalid_extraction_contract because the existing sampler returns fewer than 60 temporal positions for short videos; the extractor preserved that established sampling semantics and recorded explicit failures. One stale invalid artifact from the interrupted first attempt was recorded as invalid_cache. No fake features were counted as successful.

Recommended next atomic task: manager review of the complete TRAIN+DEV cache and explicit short-video coverage decision; then implement the next video data/model integration task without processing Test.


### MANAGER-DECISION-003 — Accept real variable-length V1 sequences up to 60 frames

Status: accepted after TASK-002G cache audit.

- TASK-002G is accepted and integrated through PR #12 as 96de6a07e65505cc61f568f800254fb14eb9d819.
- The 111 invalid_extraction_contract rows are short source videos whose deterministic sampler returns fewer than 60 real temporal positions.
- These rows must not be expanded by duplicating frames or synthesizing temporal positions.
- The V1 model contract already supports T <= 60 and mask-aware batching, so valid cached sequences with 1 <= T <= 60 are accepted.
- Batch padding, if needed, belongs only in the later cache DataModule/collate and must use video_mask=false for padded positions.
- The 96 no_body_detected rows remain genuine extraction failures; no fake visual feature may be created for them.
- The single stale invalid cache may be overwritten only with an artifact produced under the exact pinned V1 extraction contract.
- Test remains untouched and locked.

Recommended next atomic task: relax only cache structural validation to accept real 1<=T<=60 sequences, rerun TRAIN+DEV in resume mode to recover the short-video rows and stale artifact, independently audit the final coverage, and leave no-body failures explicit.


### TASK-002H - Recover short-video V1 cache rows with variable-length temporal sequences

Status: complete; Stage 2 remains partial pending later video data/model integration.

Branch: codex/task-002h.
Implementation commit: e6d516a36e65bb80125b9f06622ab5bea8444316.
Push result: successful; origin/codex/task-002h created and pushed.

Changed files:

- scripts/video/extract_clip_video_features.py;
- scripts/video/audit_video_cache.py;
- docs/PROGRESS_EN.md.

Implementation facts:

- Relaxed both extractor and independent-audit artifact validation from exact T=60 to real 1 <= T <= 60 with rank [T,512], bool mask [T], matching sampled-index length, chronological sampled indices, finite valid features, exact-zero invalid features, pinned model/revision, detector settings, and preprocessing target_frames=60 metadata.
- Preserved the existing fingerprint inputs and sampling semantics. No duplicated frames, interpolation, or synthetic temporal positions were added.
- Added independent temporal-length distribution, min/max/mean, shorter-than-target count, exact-target count, no-body failure count, and other-failure count to the audit report.

Persistent artifacts:

- extraction report: /media/maxim/Programs/Features/WSM/video_depart_v1/extraction_report_task002h.json;
- independent audit: /media/maxim/Programs/Features/WSM/video_depart_v1/cache_audit_task002h.json;
- cache root/index remained /media/maxim/Programs/Features/WSM/video_depart_v1/cache and cache/cache_index.jsonl.

Full authorized TRAIN+DEV recovery results:

- command used --splits train,dev --resume --overwrite with target_frames=60, CUDA, pinned YOLO checkpoint/SHA, and pinned CLIP revision;
- selected rows=7258;
- reused successes=7051;
- newly extracted successes=111;
- failures=96, all failure_category=no_body_detected;
- Test rows processed/indexed=0;
- no dependency installation occurred; YOLO_AUTOINSTALL=false was set.

Independent audit results:

- complete_for_requested_splits=true;
- expected_total=7258; missing_record_count=0; indexed selected rows=7258;
- successful_artifacts_valid=true; cache_fingerprints_unique=true;
- success counts: train=6255, dev=907; failures: train=70, dev=26;
- failure categories: no_body_detected=96; other failures=0;
- temporal lengths: successful count=7162, min=3, max=60, mean=59.5914549009, shorter_than_60_count=112, exact_target_frames_count=7050;
- all successful artifacts retained target_frames=60 metadata and real chronological sampled indices.

Exact verification commands/results:

- python3 -m py_compile scripts/video/extract_clip_video_features.py scripts/video/audit_video_cache.py — passed.
- Synthetic validator smoke — passed: T=17 and T=60 accepted; T=0 and T=61 rejected.
- Full extractor command with --resume --overwrite and no --limit — passed: extracted=111, reused=7051, failures=96, selected=7258.
- Independent audit helper — passed with complete=true.
- Required JSON assertions — passed for requested splits, expected count, zero Test rows, complete audit, valid artifacts, unique fingerprints, variable temporal lengths, and Test firewall flags.
- git diff --check — passed; git diff -- src/audio — empty; src/audio remained unchanged.
- No training, model selection, Test predictions, or Test metrics ran.

Deviation/blocker: the 96 no_body_detected rows remain genuine extraction failures as required. The recovered short rows use their real sampled lengths, including the measured minimum T=3; later padding belongs to the future DataModule and was not implemented here.

Recommended next atomic task: manager review of the variable-length cache and then implement the V1 video cache DataModule/collate contract with padding masks, without processing Test.


### MANAGER-DECISION-004 — Accept TASK-002H variable-length cache recovery

Status: accepted.

- TASK-002H is integrated through PR #13 as 1374e0a759f3befc498d10a70b279ba72a926d7f.
- Final TRAIN+DEV cache coverage is 7162 successful artifacts and 96 explicit no_body_detected failures.
- Success counts are train=6255 and dev=907; failures are train=70 and dev=26.
- Successful temporal lengths range from T=3 to T=60 with mean 59.5914549009; 112 successful artifacts are shorter than 60.
- The 96 no_body_detected rows remain unavailable for unimodal video training and must not be represented by fake zero videos.
- The next video DataModule must train/validate only on valid cached video artifacts while exposing explicit cache-coverage/unavailable counts in context/metadata.
- Collation must pad real variable-length sequences only at batch time and combine padding with each artifact's existing valid_mask.
- Test remains untouched and locked.

Recommended next atomic task: implement/register the V1 video cache DataModule and collate contract for TRAIN+DEV only, with variable-length padding masks and masked sparse targets; do not create a training config or run training yet.


### TASK-002I - Implement the V1 video cache DataModule and variable-length collate contract

Status: complete; Stage 2 remains partial pending later video training/integration.

Branch: codex/task-002i.
Implementation commit: 4b37a998820585245f83b0a396c16c3d2ca22ce0.
Push result: successful; origin/codex/task-002i created and pushed.

Changed files:

- src/video/data/__init__.py;
- src/video/data/wsm_video_cache_datamodule.py;
- src/chimera_plugin.py;
- docs/PROGRESS_EN.md.

Implementation facts:

- Registered wsm_video_depart_v1_datamodule and added an explicit plugin import with no project-module warning.
- The DataModule consumes only canonical TRAIN/DEV rows and joins them to cache_index.jsonl by segment_id. Test cache rows are not selected or loaded; test_dataset is None and test_rows_loaded=0.
- Every included artifact is validated for real 1 <= T <= 60 features [T,512], bool artifact valid_mask [T], finite valid features, exact-zero invalid features, matching segment_id/fingerprint, pinned CLIP identity/revision, pinned YOLO SHA/settings, target_frames=60 metadata, and chronological sampled indices.
- Failed no_body_detected records are excluded and counted as unavailable; no zero-video or synthetic samples are fabricated.
- The video-local collate pads each batch only to its real batch maximum T, preserves internal detector-missed mask positions, sets padding mask=false, and keeps padded features exact zero.
- Samples preserve independent two-task targets: unknown targets remain NaN with observed_mask=false.

Cache and context results:

- cache root/index: /media/maxim/Programs/Features/WSM/video_depart_v1/cache and cache_index.jsonl;
- train_dataset length=6255; val_dataset length=907;
- unavailable counts: train=70, dev=26;
- valid cache total=7162; explicit failures=96;
- context contract exposed video_feature_dim=512, video_sequence_steps=60, variable_length=true, and test_rows_loaded=0.

Exact verification commands/results:

- python3 -m py_compile src/video/data/__init__.py src/video/data/wsm_video_cache_datamodule.py src/chimera_plugin.py — passed.
- Chimera registry smoke — passed: wsm_video_depart_v1_datamodule registered; no project-module warning.
- Real cache/DataModule smoke — passed: mixed short/full batch [2,60,512], bool video_mask [2,60], targets [2,2], observed_mask [2,2], V1 forward, masked sparse loss, finite loss=0.6383081674575806, finite backward gradients.
- Context contract smoke — passed for all required fields and accepted counts.
- Test firewall assertions — passed: Test rows loaded=0; train samples are train split and validation samples are dev split.
- Internal detector-mask, ownership-mask, and masked-NaN assertions — passed.
- git diff --check — passed; git diff -- src/audio — empty; src/audio remained unchanged.
- No training, Test processing, Test predictions, or Test metrics ran.

Deviation/blocker: the 96 no_body_detected rows remain explicitly unavailable as required. Later batching pads variable-length sequences, but no training DataModule integration or training configuration was created.

Recommended next atomic task: manager review, then implement the next V1 video training/integration gate without processing Test.


### MANAGER-DECISION-005 — Accept TASK-002I DataModule and require masked DEV metrics before training

Status: accepted.

- TASK-002I is integrated through PR #14 as 066eab49f80280ef5f51c03560e737ab0a1fcbb7.
- The registered wsm_video_depart_v1_datamodule is accepted with train=6255, dev=907, unavailable train/dev=70/26, and test_rows_loaded=0.
- Variable-length collate semantics are accepted: pad to batch maximum T, padding mask=false, preserve artifact-internal valid_mask, and keep padded features exact zero.
- Real-cache V1 model plus wsm_masked_sparse_loss forward/loss/backward is accepted.
- Before any training YAML is authorized, the native sparse two-task DEV metrics must compute UAR/MF1/Score from observed labels only and expose the sole selector key dev/mean_score.
- Test remains locked and must not participate in metric selection.

Recommended next atomic task: implement masked two-task DEV metrics/callback integration for [B,2] logits and observed_mask, with exact selector dev/mean_score; no training config or training run yet.


### MANAGER-DECISION-006 — Metric parity across DEV and Test protocols

Status: accepted before TASK-002J implementation.

- The metric/reporting implementation must support identical masked two-task UAR/MF1/Score/Mean_Score semantics for dev, test_none, test_soft, and test_hard.
- dev/mean_score remains the sole model-selection/checkpoint/early-stopping signal.
- test_none/test_soft/test_hard metrics are required for comparative monitoring against baselines/other systems and for final reporting only.
- Test metrics must never select epochs, thresholds, hyperparameters, architectures, modalities, or ablations.
- TASK-002J is superseded in-place by the updated NEXT_TASK_EN wording before implementation.


### MANAGER-DECISION-007 — Video DataModule must expose separate Test protocols

Status: accepted before training integration.

- The V1 video DataModule must ultimately support explicit test_none, test_soft, and test_hard evaluation datasets/protocols in addition to train and dev.
- DEV remains the only validation/model-selection split and the only source of the selector dev/mean_score.
- Test datasets must not be merged into val_dataset and must not execute every validation epoch.
- Test evaluation is a separate non-selective pass used for comparative monitoring against other systems and final reporting.
- The already accepted TASK-002I train/dev DataModule remains valid as the training-side foundation; Test protocol support will be added in a separate atomic task after TASK-002J metrics support is complete.


### MANAGER-DECISION-008 — Mandatory epoch-level DEV and Test monitoring

Status: accepted; supersedes prior wording that deferred Test evaluation.

- Every epoch/validation cycle must report dev, test_none, test_soft, and test_hard metrics.
- All four protocols use the same masked two-task UAR/MF1/Score/Mean_Score definitions.
- dev/mean_score remains the only automatic checkpoint/early-stopping/model-selection signal.
- test_none/test_soft/test_hard are mandatory comparative-monitoring outputs every epoch and must be logged alongside DEV.
- Test metrics must not be consumed by checkpoint_callback, early_stopping_callback, threshold search, or automatic hyperparameter/model-selection logic.
- The video DataModule must expose test_none/test_soft/test_hard as separate named evaluation streams; they must not be merged into val_dataset.
- TASK-002J is updated in place to implement metric/callback support for all four epoch-level streams.


### TASK-002J - Implement masked two-task DEV metrics for the V1 video pipeline

Status: complete; Stage 2 remains partial pending video training/integration.

Branch: codex/task-002j.
Implementation commit: 52bf2f35b01714cf96d13330d53a4094fb40c627.
Push result: successful; origin/codex/task-002j created and pushed.

Changed files:

- src/common/callbacks/wsm_segment_callback.py;
- src/chimera_plugin.py;
- docs/PROGRESS_EN.md.

Implementation facts:

- Preserved both wsm_segment_metrics_callback and wsm_audio_metrics_callback registry keys.
- Added compute_sparse_two_task_metrics(logits, targets, observed_mask, prefix, task_names), using only observed finite binary targets, fixed logit threshold 0.0, binary confusion counts, class-defined UAR and MF1, Score=(UAR+MF1)/2, and prefix/mean_score as the arithmetic mean of the two task Scores.
- The helper supports dev, test_none, test_soft, and test_hard with identical masked-label semantics and raises clearly for zero observed samples.
- Canonical keys include P/depression/{num_samples,uar,mf1,score}, P/parkinson/{num_samples,uar,mf1,score}, P/mean_score, plus TN/FP/FN/TP audit counts. The exact selector key is dev/mean_score.
- Native sparse CachedSplitOutputs are detected from [N,2] predictions/targets. The callback obtains observed_mask from cached mask fields when available, or from native sample metadata/corpus ownership for the current Chimera cache contract. Incompatible native two-task confusion panels are skipped; legacy panels and legacy metric aliases remain supported.
- Native and legacy callback metrics are injected into logs and sent to MLflow when present. Test protocol keys are supported as mandatory future epoch-level comparative-monitoring outputs and are never used for selection.

Exact verification commands/results:

- python3 -m py_compile src/common/callbacks/wsm_segment_callback.py src/chimera_plugin.py — passed.
- Callback registry smoke — passed: wsm_segment_metrics_callback and wsm_audio_metrics_callback registered.
- Exact masked metric smoke — passed: both tasks UAR/MF1/Score and dev/mean_score equal 1.0; changing masked target values did not change metrics; zero-observation task raised ValueError.
- Prefix smoke — passed for test_none, test_soft, and test_hard with the same key schema and values.
- Native sparse callback smoke — passed: cached [N,2] logits/targets plus corpus metadata produced dev/depression/num_samples, dev/parkinson/score, and dev/mean_score=1.0.
- git diff --check — passed; git diff -- src/audio — empty; src/audio remained unchanged.
- No real Test data, Test metrics, or training run was executed in this implementation-only task.

Selector and evaluation policy: dev/mean_score is the sole future checkpoint/early-stopping/model-selection key. TEST_NONE, TEST_SOFT, and TEST_HARD metrics are supported for every later epoch-level monitoring cycle but are comparative outputs only and cannot drive selection, thresholding, tuning, or architecture decisions.

Deviation/blocker: the current Chimera CachedSplitOutputs type does not retain masks as a first-class field, so the callback uses its smallest compatible metadata/ownership fallback; future caches may provide observed_mask directly. Stage 2 remains partial.

Recommended next atomic task: manager review, then integrate the metric callback with the V1 training/evaluation streams without running Test or selecting from Test metrics.


### MANAGER-DECISION-009 — Accept TASK-002J masked four-protocol metric contract

Status: accepted.

- TASK-002J is integrated through PR #15 as 65c6003c917eb0323f965d19adaff1c17f10aacf.
- The callback helper supports dev, test_none, test_soft, and test_hard with identical observed-label-only UAR/MF1/Score/Mean_Score semantics.
- dev/mean_score remains the sole automatic checkpoint/early-stopping/model-selection key.
- The current blocker for real epoch-level Test monitoring is data-path availability: the V1 video cache currently contains TRAIN+DEV only and the accepted video DataModule exposes no Test datasets.
- Legacy WSM Test protocol semantics are fixed by the frozen audio/index path: test_none = all canonical test rows; test_soft = canonical test rows with soft_filter=1; test_hard = canonical test rows with hard_filter=1.
- The canonical manifest schema must not be changed merely to carry soft/hard membership, because changing the canonical serialization/fingerprint would invalidate the already accepted cache contract. Test protocol membership should be joined from the existing raw test metadata/segment index by canonical segment identity.

Recommended next atomic task: build and independently audit the full canonical Test video cache, correct Test extraction/audit telemetry, and extend the video DataModule with separate test_none/test_soft/test_hard datasets using the established raw soft_filter/hard_filter semantics. Do not train yet.


### TASK-002K - Build the full V1 Test video cache and expose separate Test protocols

Status: complete; Stage 2 remains partial pending epoch-level training/evaluation integration.

Branch: codex/task-002k.

Changed files:

- scripts/video/extract_clip_video_features.py;
- scripts/video/audit_video_cache.py;
- src/video/data/wsm_video_cache_datamodule.py;
- docs/PROGRESS_EN.md.

Implementation facts:

- Corrected extraction telemetry for requested Test processing: requested/selected/processed/indexed Test rows are all 1364; labels_passed_to_encoder=false and test_metrics_inspected=false.
- The independent audit now accepts any unique non-empty subset/order of train, dev, and test, ignores non-requested shared-index rows, and reports complete Test coverage.
- The full pinned Test extraction completed with 1294 extracted successes and 70 explicit no_body_detected failures; no other failures occurred.
- Raw protocol membership was joined from build_wsm_multitask_segment_index without changing the canonical manifest or fingerprint: test_none=1364, test_soft=1208, test_hard=1014.
- The DataModule preserves train=6255 and dev=907 and exposes separate test_none/test_soft/test_hard datasets. Valid rows are 1294/1150/1013 with unavailable counts 70/58/1; every Test sample carries split=test and its evaluation_protocol.
- Context exposes the three Test protocol counts, unavailable counts, indexed Test rows=1364, and data.video_epoch_test_monitoring_required=true without Test metric values.

Exact verification commands/results:

- python3 -m py_compile scripts/video/extract_clip_video_features.py scripts/video/audit_video_cache.py src/video/data/wsm_video_cache_datamodule.py - passed.
- Full authorized Test extraction with --splits test --resume --overwrite and the pinned YOLO/CLIP identities - passed: selected=1364, success=1294, failures=70, all failures no_body_detected. Report: /media/maxim/Programs/Features/WSM/video_depart_v1/extraction_report_test_task002k.json.
- Independent Test audit with --splits test - passed: expected=1364, indexed=1364, missing=0, complete=true, artifacts valid, fingerprints unique. Report: /media/maxim/Programs/Features/WSM/video_depart_v1/cache_audit_test_task002k.json.
- DataModule protocol, variable-length collate, raw-membership, and context smoke - passed: train/dev=6255/907; valid Test=1294/1150/1013; unavailable=70/58/1.
- .venv/bin/chimera-ml plugins list - passed: one wsm plugin, no project-module warning. Plugin import and DATAMODULES.get("wsm_video_depart_v1_datamodule") - passed.
- git diff --check - passed; git diff -- src/audio - empty; src/audio remained unchanged.
- No training, Test metrics, Test predictions, or selection/tuning ran.

Deviation/blocker: 70 Test rows remain explicitly unavailable because no body ROI was detected; they are excluded from valid protocol datasets and counted per protocol. Test metrics remain monitoring-only and were not inspected in this task.

Recommended next atomic task: manager review, then wire dev, test_none, test_soft, and test_hard into the epoch-level evaluation/metric callback path while keeping dev/mean_score as the sole selector and leaving Test outputs non-selective.


### MANAGER-DECISION-010 — DEPART-compatible full-frame fallback and full video coverage

Status: accepted; supersedes the prior policy that treated no_body_detected as video-unavailable for the V1 DEPART-comparable path.

- The official released DEPART preprocessing code uses the detected body ROI when a valid YOLO box exists and falls back to the full RGB frame when no box exists.
- Therefore a YOLO miss is not an extraction failure for the DEPART-comparable V1 pipeline.
- The previously accepted cache (TRAIN/DEV 7162 successes + 96 no-body failures; TEST 1294 successes + 70 no-body failures) remains historical evidence but is superseded for future V1 training/evaluation.
- A new cache version must be built under a separate cache root with explicit ROI-or-full-frame fallback in its preprocessing/fingerprint metadata.
- All canonical TRAIN/DEV/TEST rows must be attempted under one identical preprocessing contract. No row may be excluded solely because YOLO failed to detect a body.
- The full evaluation protocols must preserve raw membership: test_none=1364, test_soft=1208, test_hard=1014 before any non-YOLO fatal extraction filtering.
- If any row remains unavailable after the fallback, the failure must be a genuine non-YOLO fatal source/model/runtime error and must be audited individually.
- Training configuration/integration is paused until this full-coverage cache gate is cleared.

Recommended next atomic task: implement the ROI-or-full-frame fallback, version/fingerprint it, rebuild and independently audit the full 8622-row TRAIN+DEV+TEST cache in a new cache root, and repoint the video DataModule to that cache with full protocol coverage. Do not train yet.


### TASK-002K2 - Rebuild the DEPART-compatible full-coverage video cache with full-frame fallback

Status: complete; Stage 2 remains partial pending TASK-002L training/evaluation integration.

Branch: codex/task-002k2.
Implementation commit: 1b5e7c27c86c9dd3aec1b21c0c7b0cca7a659864.
Push result: successful; origin/codex/task-002k2 created and pushed.

Changed files:

- src/video/features/clip_video_features.py;
- scripts/video/extract_clip_video_features.py;
- scripts/video/audit_video_cache.py;
- src/video/data/wsm_video_cache_datamodule.py;
- docs/PROGRESS_EN.md.

Implementation facts:

- Replaced the superseded ROI-only policy with preprocessing_version=depart-v1-fullframe-fallback and roi_policy_version=depart-body-roi-if-detected-otherwise-full-rgb-frame-v2.
- Each sampled frame now uses body_roi when YOLO returns a valid box, otherwise full_frame_fallback. Every readable sampled frame is encoded; YOLO misses are not failures.
- Artifacts record frame_sources, selected_boxes, detected_body_count, full_frame_fallback_count, detection_coverage, and fallback_coverage. valid_mask is all true for every readable temporal position.
- The V1 DataModule default cache root is now /media/maxim/Programs/Features/WSM/video_depart_v1_fullframe_fallback/cache and requires the new policy metadata. The superseded /media/maxim/Programs/Features/WSM/video_depart_v1/cache was preserved and not reused or modified.
- The released DEPART repository has additional implementation/config provenance differences; those are outside this atomic ROI/fallback correction and were not changed here.

Full extraction result:

- New cache root: /media/maxim/Programs/Features/WSM/video_depart_v1_fullframe_fallback/cache.
- Expected/success: train=6325/6325, dev=933/933, test=1364/1364, total=8622/8622.
- Failure total=0; missing total=0; Test failures=0.
- Extraction report: /media/maxim/Programs/Features/WSM/video_depart_v1_fullframe_fallback/extraction_report_all_task002k2.json.
- Preprocessing/model reports: extracted_success_count=8622, reused_success_count=0, no_body_detected_count=0, other_failure_count=0.

Independent audit statistics:

- Audit report: /media/maxim/Programs/Features/WSM/video_depart_v1_fullframe_fallback/cache_audit_all_task002k2.json.
- complete_for_requested_splits=true, artifacts valid, fingerprints unique, test_rows_processed=true, test_rows_indexed=1364.
- train: body detections=332131, full-frame fallbacks=44150, segments with fallback=3041, detection coverage mean/min/max=0.8808096312/0.0/1.0, fallback coverage mean/min/max=0.1191903688/0.0/1.0, temporal length mean/min/max=59.4910672/3/60, shorter-than-60=120.
- dev: body detections=48727, full-frame fallbacks=6596, segments with fallback=405, detection coverage mean/min/max=0.8763821497/0.0/1.0, fallback coverage mean/min/max=0.1236178503/0.0/1.0, temporal length mean/min/max=59.2958199/2/60, shorter-than-60=25.
- test: body detections=68066, full-frame fallbacks=12963, segments with fallback=637, detection coverage mean/min/max=0.8351923528/0.0/1.0, fallback coverage mean/min/max=0.1648076472/0.0/1.0, temporal length mean/min/max=59.4054252/4/60, shorter-than-60=25.

DataModule result:

- train_dataset=6325 and val_dataset=933.
- Raw and valid Test protocol counts: test_none=1364, test_soft=1208, test_hard=1014.
- Unavailable counts: train=0, dev=0, test_none=0, test_soft=0, test_hard=0.
- Separate Test streams and variable-length collate passed.

Exact verification commands/results:

- python3 -m py_compile src/video/features/clip_video_features.py scripts/video/extract_clip_video_features.py scripts/video/audit_video_cache.py src/video/data/wsm_video_cache_datamodule.py - passed.
- Focused injected detector smoke - passed: body ROI plus full-frame fallback produced finite [2,512] features, valid_mask=[true,true], detected=1, fallback=1, and both coverage values=0.5.
- Fingerprint smoke - passed: old ROI-only fingerprint differs from the new fallback-policy fingerprint.
- Full extraction with --splits train,dev,test --resume and pinned YOLO/CLIP identities - passed with 8622/8622 success and zero failures.
- Independent audit and strict coverage assertions - passed.
- DataModule strict coverage and variable-length collate smoke - passed.
- .venv/bin/chimera-ml plugins list and DATAMODULES.get("wsm_video_depart_v1_datamodule") - passed without project-module warning.
- git diff --check - passed; git diff -- src/audio - empty; src/audio remained unchanged.
- No training, Test performance metrics, Test predictions, or model/checkpoint selection ran.

Recommended next atomic task: TASK-002L training config plus every-epoch DEV/TEST_NONE/TEST_SOFT/TEST_HARD wiring, with dev/mean_score as the only selector.


### MANAGER-DECISION-011 — Accept TASK-002K2 full-coverage DEPART-compatible cache

Status: accepted.

- TASK-002K2 is integrated through PR #17 as bf6e6364a6c7673d7dc84a1611a7a1c469571371.
- Full canonical cache coverage is accepted: train=6325/6325, dev=933/933, test=1364/1364, total=8622/8622, failures=0, missing=0.
- ROI-if-detected/full-RGB-frame-fallback semantics are now the authoritative V1 video preprocessing contract.
- The accepted V1 DataModule exposes full protocol coverage: train=6325, dev=933, test_none=1364, test_soft=1208, test_hard=1014, with all unavailable counts zero.
- The superseded cache remains preserved only as historical evidence and must not be used for V1 training.
- The remaining pre-training requirement is integration wiring only: every epoch must evaluate dev/test_none/test_soft/test_hard while checkpointing/early stopping use only dev/mean_score.

Recommended next atomic task: TASK-002L final pre-training config/integration gate. After TASK-002L passes, proceed directly to TASK-002M real V1 video training run.


### TASK-002L - Create the V1 video training config and wire four epoch-level evaluation streams

Status: complete; Stage 2 is ready for the first real V1 training run.

Branch: codex/task-002l.
Implementation commit: pending until the implementation commit is created.
Push result: pending until the branch is pushed.

Changed files:

- src/video/data/wsm_video_cache_datamodule.py;
- configs/wsm_mm_pd_dep_v1/video/00_depart_v1.yaml;
- docs/PROGRESS_EN.md.

Implementation facts:

- Added WSMVideoCacheDataModule.val_dataloader() returning exactly dev, test_none, test_soft, test_hard in that order. DEV remains val_dataset only; Test remains separate test_dataset entries. All four loaders use shuffle=false, drop_last=false, and the variable-length WSM collate. train_dataloader() semantics are unchanged.
- Added the self-contained config at configs/wsm_mm_pd_dep_v1/video/00_depart_v1.yaml with experiment_name=wsm_mm_pd_dep_v1 and run_name=depart_v1_clip_yolo_transformer.
- Fixed V1 model parameters: video_feature_dim=512, hidden_dim=192, num_layers=2, num_heads=4, ff_mult=4, dropout=0.2, sequence_steps=60, num_tasks=2.
- Fixed optimizer parameters: AdamW lr=0.0001 and weight_decay=0.01. Training parameters are epochs=30, device=cuda, mixed_precision=true, grad_clip_norm=0.5, log_every_steps=25, collect_cache=true. No scheduler or sweep is configured.
- The config uses the accepted full-coverage cache root and wsm_video_depart_v1_datamodule, wsm_video_depart_v1_model, wsm_masked_sparse_loss, and adamw_optimizer registry keys.
- Required instrumentation is present: checkpoint_callback, snapshot_callback, early_stopping_callback, wsm_summary_callback, wsm_segment_metrics_callback, console_file_logger, and mlflow_logger. Checkpointing and early stopping both monitor only dev/mean_score in max mode. Test protocol names are not selector inputs.

DataModule counts:

- train_dataset=6325; val_dataset=933;
- test_none=1364; test_soft=1208; test_hard=1014;
- all train, DEV, and Test unavailable counts are zero;
- validation stream keys are exactly dev, test_none, test_soft, test_hard.

Exact verification commands/results:

- PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml validate-config --config-path configs/wsm_mm_pd_dep_v1/video/00_depart_v1.yaml - passed: Config is valid.
- Actual Chimera registry build smoke for DataModule, model, masked loss, AdamW optimizer, five callbacks, and two loggers - passed; no project-module warning. Dataset counts and four stream keys asserted.
- python3 -m py_compile src/video/data/wsm_video_cache_datamodule.py - passed.
- Selector firewall assertions and corrected ripgrep monitor scan - passed: checkpoint and early stopping use dev/mean_score/max; no test_none/test_soft/test_hard monitor reference.
- Bounded one-epoch CPU Trainer.fit smoke used exactly 4 real train samples and 4 real samples for each of dev, test_none, test_soft, and test_hard, with two depression-owned and two Parkinson-owned rows per stream. Forward/loss/backward and the real wsm_segment_metrics_callback completed.
- Structural, non-performance smoke values from that single epoch were dev/mean_score=0.5, test_none/mean_score=0.5, test_soft/mean_score=0.5, and test_hard/mean_score=0.5. All four were finite and appeared in the same epoch logs. Test metrics were monitoring outputs only and were not used for selection.
- No full V1 training experiment, model selection, threshold search, or Test performance evaluation ran.
- git diff --check passed; git diff -- src/audio was empty; src/audio remained unchanged.

Deviation: an initial bounded smoke subset selected only depression-owned rows and correctly raised the sparse-metric zero-observation guard for DEV/Parkinson. The final deterministic smoke used the smallest balanced 2+2 ownership subset for every stream and passed.

Recommended next atomic task: TASK-002M real V1 video training run.
