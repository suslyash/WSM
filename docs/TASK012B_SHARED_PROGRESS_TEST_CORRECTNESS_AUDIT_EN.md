# TASK-012B — Shared + Progress Test correctness audit

## 1. Boundary and verdict

This is a read-only correctness audit of the already-saved TASK-012A post-closure evidence. No production evaluator, DataModule, model, forward pass, inference, training, tuning, Test rerun, or new experiment was executed. The saved closure evaluator and original evidence dossier were not modified.

Overall verdict: **VERIFIED WITH STATED LIMITATIONS**. The inspected source/config/artifact relationships contain no confirmed scientific or metric-correctness defect. The verdict is not a claim that the historical predictions, sample membership, or exact runtime transcript have been independently reproduced: those inputs were intentionally not retained in the supplied artifacts.

## 2. Evidence identity and immutability checks

The audit read the following frozen evidence:

| Evidence | Bytes | SHA-256 |
|---|---:|---|
| `logs/task012a_postclosure_shared_progress_test/dev_preflight.json` | 11,291 | `67a7c70754070dd0594262ab706aac8d3d853d9b2a84f349adf97e08e8ac2e64` |
| `logs/task012a_postclosure_shared_progress_test/test_results.json` | 12,875 | `ddcae2835c04a346c1a27d3e9fa9d7c2b2afc96980b46d206dd2e79120baf285` |
| `logs/task012a_postclosure_shared_progress_test/test_invocation.marker` | 48 | `3a37730e932f29c5928b930263d6c745c6c9f9171b9c42b34990320de329e12d` |

The original evaluator SHA is `7157f32253b6f3e831a00992c180f39fdd0d86d47c4718517b0b7e6584d2a5b7`; the closure evaluator SHA is `108079eb382ac580b3293f8716a58848e4d6e1123f952141f48aa3041a247317`. The closure diff is limited to the permanent `RuntimeError` guard at the start of `test_pass`; no metric, loader, model, or protocol semantics changed.

The five checkpoint payloads were inspected with CPU-only `torch.load(..., map_location="cpu", weights_only=True)`. Each had exactly `epoch`, `global_step`, `model_state_dict`, and `optimizer_state_dict`; each model state had 51 float32 keys and no non-float buffers. The observed epochs/global steps were 7/1386, 9/1782, 5/990, 19/3762, and 3/594 for seeds 42–46. Static inspection supports checkpoint-shape compatibility; it does not replace the prohibited model construction or strict-load execution.

## 3. Protocol and task semantics

| Check | Finding | Status |
|---|---|---|
| Protocol coverage | 15 rows = 5 seeds × `test_none`, `test_soft`, `test_hard`; counts 1364/1208/1014 | Confirmed in saved JSON |
| Task order | Depression then Parkinson in model auxiliary outputs and metric task names | Confirmed by source inspection |
| Unknown labels | `observed_*` masks control inclusion; unobserved targets remain NaN/neutral in the evaluation path | Confirmed by source inspection |
| Decision rule | Raw logits are thresholded at zero; no sigmoid/probability threshold is used | Confirmed by source inspection |
| Per-task score | `Score = (UAR + MF1)/2`; row `Mean = (D Score + P Score)/2` | Independently recomputed from saved JSON |
| Accumulation | Saved result declares model and output accumulation dtype `torch.float32` | Confirmed in all 15 rows |
| Test selection | The saved Test rows are monitoring outputs; no Test row is used for epoch/threshold/model selection | Confirmed from evaluator/source and closure context |

The seed-42 run names the production semantic DataModule; seeds 43–46 name its DEV-only wrapper. The wrapper changes trainer validation exposure, while the evaluator directly requests the base `test_dataset` streams. This is an intentional evaluation-path distinction, not evidence that the Test protocol changed. The evaluator also overrides construction-time training-loader knobs (`num_workers=0`, `shuffle_train=False`, and related flags); its Test DataLoader is explicitly non-shuffled, non-dropping, and single-process. The saved YAML values therefore must not be read as proof of the effective Test loader settings.

## 4. Independent metric and aggregate audit

`scripts/common/audit_task012a_saved_evidence.py` contains a standalone oracle. It imports no production evaluator, model, DataModule, callback, loss, or inference entry point. On fabricated observed labels/scores it independently checks zero-threshold confusion arithmetic, per-class recall/F1, sparse masking convention, task-mean arithmetic, and five-seed sample-standard-deviation confidence intervals. The oracle passed both fabricated cases and all 15 saved rows passed the structural checks.

The saved rows independently summarize as follows (95% intervals use the frozen multiplier `2.7764451051977987` and sample SD):

| Protocol | D mean | P mean | Mean mean |
|---|---:|---:|---:|
| `test_none` | 0.761796 | 0.775681 | 0.768738 |
| `test_soft` | 0.752108 | 0.777454 | 0.764781 |
| `test_hard` | 0.752014 | 0.758632 | 0.755323 |

Compared with the frozen Stage-7 per-seed shared-fusion rows, the saved TASK-012A Test result mean deltas are:

| Protocol | D delta mean | P delta mean | Mean delta mean |
|---|---:|---:|---:|
| `test_none` | -0.037437 | -0.037093 | -0.037265 |
| `test_soft` | -0.060649 | -0.040486 | -0.050568 |
| `test_hard` | -0.063903 | -0.078417 | -0.071160 |

This decomposition is arithmetic over saved rounded historical rows, so the last displayed digits can differ from a decomposition made from unreleased full-precision predictions. It is evidence about where the reported Mean change comes from, not a causal diagnosis.

## 5. Correctness findings

### Confirmed consistent

- The closure guard is administrative only and leaves the original evaluator semantics unchanged.
- Artifact schemas, byte lengths, hashes, protocol counts, row cardinality, task ordering, score arithmetic, and float32 output declarations are internally consistent.
- The source model has two independent disease heads with the documented depression/Parkinson ordering; static checkpoint keys and shapes match the configured 768/512/256/160 architecture family.
- The source DataModule separates `DEV`, `TEST_NONE`, `TEST_SOFT`, and `TEST_HARD`, preserves observed-label masks, and uses the expected protocol counts.
- The saved Test output contains no raw logits, raw predictions, probabilities, labels, or sample metadata, as required by the closure evidence boundary.

### Harmless or intentional differences

- The current closure evaluator cannot be run by design; the source-level permanent guard is the expected TASK-012A-C1 behavior.
- The saved config names and evaluator constructor overrides differ for training-loader settings. The evaluator’s explicit evaluation loader settings, not the training-loader fields, govern the reported Test pass.
- The negative Test result relative to Stage 7 is not itself a correctness defect. It is decomposed above and requires prediction-level evidence for any causal explanation.

### Unresolved provenance limitations

- No raw predictions/logits, labels, sample IDs, row order, or per-sample masks were retained; confusion matrices, class counts, and membership identity cannot be independently recomputed from the saved artifacts.
- Exact UTC command transcript and exact historical autocast dtype are not retained in the TASK-012A artifacts. The source/evidence states CUDA plus enabled autocast and float32 CPU accumulation, but this audit does not infer missing runtime details from current defaults.
- CPU-only static checkpoint inspection cannot independently prove strict `load_state_dict` success or historical forward compatibility; it only verifies payload schema, key count, dtype, and representative shapes.
- The Stage-7 evidence text records an environment/runtime assertion about float16 autocast, while its report-level runtime identity is a separate presentation layer. This is a provenance/reporting limitation, not a demonstrated TASK-012A metric defect.

## 6. Commands and scope

Required repository docs, active-task context, source/config paths, frozen evaluator sources, artifact hashes, and checkpoint payloads were inspected read-only. The reproducible audit command was:

```bash
python3 scripts/common/audit_task012a_saved_evidence.py \
  --dev logs/task012a_postclosure_shared_progress_test/dev_preflight.json \
  --test logs/task012a_postclosure_shared_progress_test/test_results.json \
  --stage7 docs/STAGE7_FINAL_TEST_EVIDENCE_EN.md \
  --output /tmp/task012b_audit_summary.json
```

Result: exit 0; independent oracle passed; 15/15 saved rows passed structural checks; verdict `VERIFIED WITH STATED LIMITATIONS`. Synthetic tests were run separately with `pytest tests/test_task012b_shared_progress_test_correctness_audit.py`.

No production evaluation command, DataModule/model construction, forward pass, inference, training, tuning, Test rerun, or follow-up experiment was run. No Test metric was used to select anything.

## 7. Manager conclusion

TASK-012B is complete as a correctness audit. There is no confirmed correction to authorize from the retained evidence. Any future causal investigation would require a separately authorized task that preserves prediction-level and sample-level provenance; it is outside this audit.
