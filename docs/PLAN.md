# WSM Development Plan

## 1. Execution Contract

Stages are normally sequential, but the owner explicitly overrides the research execution order after Stage 2: execute Stage 4 (Fusion), Stage 5 (RAMPS), and Stage 6 (core ablations/research) before returning to the deferred Stage 3 (Text/Description). Advance only after the active stage gate is met and evidence is recorded in PROGRESS_EN.md.

Authority: latest user instruction → PROJECT_REQUIREMENTS.md → this plan → BASELINES.md/SOTA_REVIEW_EN.md → existing code.

TEST_NONE/SOFT/HARD MUST be reported every epoch/validation cycle for comparative monitoring, using the same per-task UAR/MF1/Score/Mean_Score definitions as DEV. They MUST NOT be consumed by automatic checkpointing, early stopping, threshold search, or hyperparameter/model-selection logic. Sole automatic selection metric:

\[
\mathrm{DEV/Mean\_Score}
=\frac{1}{2}
\left[
\frac{\mathrm{UAR}_D+\mathrm{MF1}_D}{2}
+
\frac{\mathrm{UAR}_P+\mathrm{MF1}_P}{2}
\right].
\]

## 2. Research Hypothesis

Working method name: **RAMPS — Reliability-Aware Multimodal Partial Supervision**.

> Disjoint depression and Parkinson corpora can train one multilabel predictor if the missing disease head receives a soft cross-corpus target only when calibrated disease teachers, independent modality predictions, and semantic evidence agree; target reliability should control both pseudo-loss and inter-task gradient balance.

RAMPS jointly addresses:

1. multimodal fusion with variable stream availability/quality;
2. dataset-level partial labels;
3. negative transfer across tasks and corpora.

Proposed novelty:

- label-query fusion for independent disease labels;
- direct cross-corpus supervision for the missing head;
- reliability combining calibration, uncertainty, multimodal/semantic agreement, and OOD distance;
- reliability-conditioned smooth multi-objective optimization.

This is a working novelty hypothesis, not a guaranteed claim. Re-run a focused literature check before submission.

### 2.1. SOTA synthesis

| Direction | Useful mechanism | Remaining WSM gap |
|---|---|---|
| PASAL-like sparse MTL | Shared encoder learns from all observed tasks | Missing head gets no direct gradient |
| HiTTs-like label discovery | Direct pseudo-target for an unobserved task | Validated mainly for dense prediction |
| PPL/UNITI multi-dataset transfer | Teacher/student stability and distillation | Teacher bias and corpus shortcuts remain |
| SATL/CLS incomplete multilabel | Classwise thresholds and co-training | Usually not four-modality disjoint-disease MTL |
| HST semantic/prototype recovery | Label semantics recover unknowns | Semantic similarity is not clinical truth |
| Dual-branch pseudo-labeling | Independent views reduce confirmation bias | Must map views to A/V/T/D |
| URDF/CTRL uncertainty | Reject noisy supervision | Rarely coupled to task-gradient balance |
| V2L label-wise reliability | Per-label modality weighting | Does not solve cross-corpus missing tasks |

Required ladder: sparse MTL → calibrated pseudo-target → semantic/multimodal agreement → uncertainty/OOD reliability → reliability-aware optimization.

The unresolved intersection is multimodal + multitask + multilabel + dataset-level missing labels. If later work already matches this combination, revise the claim and architecture before expensive final runs.

## 3. Formulation

For sample \(x_i\):

\[
\mathbf y_i=(y_i^D,y_i^P),\qquad
\mathbf m_i=(m_i^D,m_i^P).
\]

\(m_i^t=1\) only for a truly observed label. The model returns:

\[
\mathbf z_i=(z_i^D,z_i^P),\qquad
\hat{\mathbf y}_i=\sigma(\mathbf z_i).
\]

Observed task loss:

\[
\mathcal L_{\mathrm{obs}}^t
=
\frac{\sum_i m_i^t\ell(z_i^t,y_i^t)}
{\sum_i m_i^t+\epsilon}.
\]

For \(m_i^t=0\), a teacher yields detached soft target \(\tilde y_i^t\), reliability \(r_i^t\in[0,1]\), and acceptance \(a_i^t\):

\[
\mathcal L_{\mathrm{pseudo}}^t
=
\frac{\sum_i(1-m_i^t)a_i^t r_i^t\ell(z_i^t,\tilde y_i^t)}
{\sum_i(1-m_i^t)a_i^t r_i^t+\epsilon}.
\]

Use separate positive/negative thresholds and warm-up:

\[
\mathcal L_t
=
\mathcal L_{\mathrm{obs}}^t
+
\mu(e)\mathcal L_{\mathrm{pseudo}}^t
+
\beta\mathcal L_{\mathrm{cons}}^t.
\]

Static STCH baseline:

\[
\mathcal L_{\mathrm{STCH}}
=
\tau\log\sum_t
\exp\left(
\frac{\lambda_t(\mathcal L_t-z_t^\star)}{\tau}
\right).
\]

RAMPS studies a detached, bounded, normalized, EMA-smoothed controller \(\alpha_t\) based on relative DEV progress, gradient norm/conflict, and accepted-pseudo reliability:

\[
\mathcal L_{\mathrm{RA-STCH}}
=
\tau\log\sum_t
\exp\left(
\frac{\alpha_t(\mathcal L_t-z_t^\star)}{\tau}
\right).
\]

Do not claim that dynamic RA-STCH inherits fixed-weight STCH theory without a new proof.

## 4. Architecture

### 4.1. Encoders

- Audio: frozen current WavLM-base-plus layer 9, pool 4, temporal Transformer; use checkpoint/cache through an external adapter.
- Video: frozen CLIP frame encoder, projection, temporal Transformer; one optional prototype-aware variant.
- Text: pretrained encoder with lightweight pooling/temporal head, selected after language/data audit.
- Description: cached observable descriptions/features from Qwen3-Omni-30B-A3B-Instruct or comparable model; no diagnosis or label/task input.

Each stream exposes token sequence, pooled representation, availability mask, quality metadata, and optional unimodal task logits.

### 4.2. Fusion ladder

1. masked concatenate/gated fusion;
2. TACME-like directed experts with task-specific gates;
3. label-query reliability fusion.

Final fusion uses learnable disease queries \(q_D,q_P\). Each query aggregates available modality tokens through cross-attention or a compact expert router. Gates depend on task, availability, and quality—not a corpus shortcut. Outputs remain independent logits.

TACME pairwise experts are a baseline. Prefer shared low-rank experts or query routing in the final model unless \(O(M^2)\) experts show clear benefit.

### 4.3. Teachers and reliability

Each disease teacher trains only on observed labels for that disease. Cross-corpus predictions never become truth automatically.

\[
r_i^t=g(c_i^t,u_i^t,a_{\mathrm{modal},i}^t,
a_{\mathrm{semantic},i}^t,d_{\mathrm{OOD},i}^t),
\]

where \(c\) is calibrated confidence, \(u\) uncertainty, modal/semantic terms measure agreement, and \(d_{\mathrm{OOD}}\) measures domain distance.

Start with a transparent normalized formula and class-specific thresholds. Learnable reliability requires a separate audit/calibration protocol.

### 4.4. Shortcut checks

Train an analysis-only corpus probe. If representations nearly identify corpus, test balanced sampling, train-only corpus normalization, domain-adversarial ablation, metadata restrictions, or stricter OOD filtering. Accept an intervention only if DEV/Mean_Score and negative transfer improve.

## 5. Stages

### Stage 0 — Reproducible base

Tasks:

1. restore the provider required by audio without editing src/audio;
2. create configs/wsm_mm_pd_dep_v1/audio/00_frozen_baseline.yaml from the best run;
3. verify plugin/registry;
4. verify required instrumentation and dev/mean_score monitor;
5. keep new helper scripts under scripts/<modality>;
6. prove src/audio is unchanged.

Gate: clean plugin import, audio model registered, canonical config valid/self-contained, shared experiment_name present, and empty audio diff.

### Stage 1 — Manifest and partial-label contract

Create one manifest with:

    segment_id, video_id, speaker_id, corpus, split,
    y_depression, y_parkinson,
    observed_depression, observed_parkinson,
    audio_available, video_available,
    text_available, description_available

Add audit reports, cache fingerprints, unified split identity, and separate DEV/Test dataloaders.

Gate: no video split leakage; current train/dev/test speaker independence is accepted from the dataset-owner split contract despite unavailable speaker_id and must be recorded as an assumption rather than a measured identity audit; unknown never maps to zero; counts/missing streams saved machine-readably; Test absent from fit validation.

### Stage 2 — At most two video families

V1: DEPART-like uniform temporal sequence, YOLOv8 single-class human-body ROI detection/cropping per sampled frame, frozen CLIP frame encoding, projection, temporal Transformer, masked pooling, two sigmoid heads.

V2: V1 plus task-specific class prototypes, classwise prototype/MLP gating, and controlled contrastive ablation.

Gate: reproducible preprocessing/cache, extraction failure report, one DEV winner selected without Test.

### Stage 3 — At most two text/description families [DEFERRED UNTIL AFTER STAGES 4-6]

T1: audited-language transcript encoder with mask-aware pooling and two heads.

T2: T1 plus cached observable description/semantic feature and simple T/D gating. Include prompt/model revision in the fingerprint and manually audit a stratified sample.

Gate: at most two configs, diagnosis-free prompt, one DEV-selected representation.

### Stage 4 — Honest multimodal baselines [COMPLETE — NO SAFE STRONG-AUDIO FUSION WINNER]

- F0: simple masked late/gated fusion.
- F1: sparse masked two-head MTL with no pseudo-labeling.
- F2: TACME-like task-aware relation bank with observed loss; add PAGB comparator only if budget permits.
- Owner-authorized pre-RAMPS strong-audio ablation: replace the pooled `audio_cls` fusion branch with the frozen DEV-selected historical temporal-audio checkpoint through a fusion-side adapter; do not edit `src/audio`.
- The first strong-audio residual run produced a valid negative result, but its randomly initialized residual path did not start from the exact frozen-audio function. The owner therefore authorizes one bounded audio-first temporal-fusion research sprint before RAMPS rather than continuing one hand-designed model at a time.
- The sprint may implement/train up to three seed-42 candidate fusion models in one manager task. All candidates MUST keep the exact historical temporal audio checkpoint frozen, MUST start with final logits exactly equal to the frozen audio base logits, and MUST use video only through a zero-initialized additive correction path.
- Candidate architectures are chosen by Codex from a manager-bounded design space emphasizing the stronger audio modality: zero-init residual fusion, task-specific audio-query temporal-video attention, audio-confidence/gated video correction, and/or one F2-like directed temporal relation variant. Exact internal details may vary within the fixed parameter/compute caps, but the candidate manifest MUST be frozen before the first training run.
- All candidates use the same canonical data, seed=42 screening protocol, trainable-only AdamW, DEV-only selector, and four evaluation streams every epoch. TEST_NONE/SOFT/HARD remain monitoring-only and MUST NOT influence candidate design/ranking.
- The screening goal is to exceed frozen audio DEV Mean_Score=0.7878268765 while avoiding a >1.0 absolute percentage-point DEV Score drop on either task. A single-seed screen may nominate a candidate only; promotion still requires later multi-seed confirmation.
- RAMPS Stage 5 is deferred until this bounded audio-first temporal-fusion sprint is completed or explicitly stopped by the manager.

Gate: same canonical splits/metric and DEV-only selector; exact frozen-audio reproduction at initialization for every candidate; no Test-driven search; at most three screened candidate models; preserve all negative results; nominate at most one DEV winner for later multi-seed confirmation.

### Stage 5 — RAMPS [ACTIVE]

- R1: disease teachers calibrated only on the corresponding observed-task DEV data; soft cross-corpus targets, warm-up, separate thresholds, stop-gradient teacher.
- R2: uncertainty, multimodal agreement, OOD distance, coverage curves; semantic agreement only after the base reliability mechanism.
- R3: disease-query fusion, task-conditioned modality gates, availability masks, auxiliary unimodal agreement heads.
- R4: equal weights, static STCH, PAGB/progress comparator, and RA-STCH.

R-full includes only components that improve DEV and do not worsen calibration/negative transfer.

Gate: missing head receives a non-zero direct gradient on the other corpus; observed truth wins; accepted class balance/coverage logged; gradients finite; gains not explained only by parameter count; final composition frozen before Test.

### Stage 6 — Ablations and claims audit

Required:

1. F1 sparse MTL;
2. + task-aware fusion;
3. + direct pseudo-supervision;
4. + uncertainty/reliability;
5. + semantic evidence;
6. + RA-STCH;
7. remove each modality;
8. shuffled/mismatched pseudo-target negative control;
9. corpus probe;
10. equal-parameter control if size grows materially.

Log DEV scores, UAR/MF1, ECE/Brier, pseudo coverage by class, task gradient norms/cosines, negative-transfer deltas, and gate distributions. Use 3 seeds for ablations.

If a dual-annotated audit subset exists, measure missing-label precision/recall/calibration but do not use it for ordinary training.

Gate: each claim maps to an ablation; negative results are retained; final config/thresholds are immutable.

### Stage 7 — Final evaluation

After freeze:

1. train final compared methods on at least 5 seeds;
2. choose every checkpoint only by DEV/Mean_Score;
3. run separate TEST_NONE/SOFT/HARD evaluation once;
4. report per-task UAR/MF1/Score and Mean_Score;
5. report mean, standard deviation, confidence interval, and paired comparison where applicable;
6. trace every table entry to config, commit, checkpoint, and MLflow run.

Gate: no Test-driven revision; complete traceability; claims match annotation evidence.

## 6. Experiment Matrix

| ID | Method | Question | Pre-final seeds |
|---|---|---|---:|
| A0 | Frozen audio | Is the original baseline reproducible? | 1 smoke + saved evidence |
| V1 | CLIP + Transformer | Is the base DEPART-like encoder sufficient? | 1–3 |
| V2 | V1 + prototypes | Do prototypes help? | 1–3 |
| T1 | Transcript encoder | Does text add independent signal? | 1–3 |
| T2 | Transcript + description | Does the semantic stream help? | 1–3 |
| F0 | Simple masked fusion | Basic multimodal gain | 3 |
| F1 | Sparse masked MTL | Shared two-head benefit | 3 |
| F2 | TACME-like fusion | Directed task-relation benefit | 3 |
| R1 | F2 + pseudo | Direct missing-head benefit | 3 |
| R2 | R1 + reliability | Negative-transfer reduction | 3 |
| R3 | Label-query RAMPS | Task/label-conditioned fusion benefit | 3 |
| R4 | R3 + RA-STCH | Reliability-aware balancing benefit | 3 |
| Final | Best baselines and RAMPS | Final comparison | 5+ |

This caps experiment families; it does not authorize a full grid search.

## 7. Promotion Rule

Compare DEV/Mean_Score first, then task-specific regressions, calibration/negative transfer, compute/stability, and diagnostic support.

Promote a component only when:

- the effect repeats across at least 3 seeds;
- mean DEV/Mean_Score beats the comparator;
- neither task shows a persistent unacceptable drop;
- diagnostics support the proposed mechanism.

Fix the unacceptable-drop threshold before final runs; initial recommendation is 1.0 absolute percentage point unless the confidence interval indicates noise.

## 8. Risks and Fallbacks

| Risk | Detection | Fallback |
|---|---|---|
| Corpus identity replaces disease signal | corpus probe/calibration | stricter OOD filter, balanced sampling, domain ablation |
| Pseudo-label majority collapse | classwise coverage/precision | separate thresholds, class cap, longer warm-up |
| Description leaks diagnosis/bias | prompt/manual audit | remove diagnostic tokens, fixed semantic features, drop stream |
| Video extraction fails | failure report/mask | mask stream; never synthesize fake features |
| Pairwise experts are too costly | memory/time profile | shared low-rank query router |
| Dynamic weighting is unstable | norms/cosines/NaN | static STCH or equal weights |
| Fusion underperforms audio | negative-transfer table | preserve strong audio path; improve gating/dropout |
| No dual-annotated subset | missing truth unevaluable | restrict claims to observed-task transfer |

## 9. Planned Layout

    configs/wsm_mm_pd_dep_v1/{audio,video,text,description,fusion,ablations}/
    scripts/{audio,video,text,description,fusion,common}/
    src/{audio,video,text,description,fusion,common}/

Universal manifest/masks, metrics, diagnostics, losses, and generic blocks belong in common. Teachers, routers, the multimodal datamodule, and the final model belong in fusion. All selectable components must be registered.

## 10. Manager Orchestration

Each cycle:

1. manager reads PROGRESS_EN and verifies the previous gate;
2. manager writes exactly one atomic task to NEXT_TASK_EN;
3. Codex reads AGENTS, REQUIREMENTS, PLAN, PROGRESS_EN, and NEXT_TASK_EN;
4. Codex executes only that scope, verifies it, updates PROGRESS_EN, and stops;
5. manager reviews diff/evidence and either corrects the task or advances the plan.
