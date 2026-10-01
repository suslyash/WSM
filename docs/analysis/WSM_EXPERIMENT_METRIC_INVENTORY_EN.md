# WSM experiment metric inventory for DEPART comparison

## Scope and metric definitions

This documentation-only audit extracts metrics already recorded in accepted evidence and preserved summaries. It introduces no training, inference, checkpoint loading, dataset iteration, thresholding, metric recomputation, or role change. Empty CSV cells mean unavailable, never zero.

- **D UAR**, **D MF1**, **D Score**: depression unweighted average recall, macro F1, and composite Score.
- **P UAR**, **P MF1**, **P Score**: Parkinson counterparts.
- **Mean**: two-task composite mean.

The repository defines `Score = (UAR + MF1) / 2` and `Mean = (D Score + P Score) / 2`. UAR/MF1 are never reverse-engineered from Score.

## Final frozen methods

Every final seed 42--46 is represented in the CSV for DEV, TEST_NONE, TEST_SOFT, and TEST_HARD. DEV derives from the frozen final ledger; Test derives from the one standardized Final Test report. Full-R4 DEV seeds42--44 task UAR/MF1 are transcribed from preserved selected summaries and cross-checked against frozen Score/Mean.

| Method | Seed | Protocol | D UAR / MF1 / Score | P UAR / MF1 / Score | Mean |
|---|---:|---|---|---|---:|
| temporal audio | 42 | DEV | 0.7480392157/0.7477975633/0.7479183895 | 0.8209799862/0.8344907407/0.8277353635 | 0.7878268765 |
| temporal audio | 42 | TEST_NONE | 0.768681/0.763094/0.765887 | 0.865406/0.840764/0.853085 | 0.809486 |
| temporal audio | 42 | TEST_SOFT | 0.781461/0.775970/0.778716 | 0.863621/0.839571/0.851596 | 0.815156 |
| temporal audio | 42 | TEST_HARD | 0.794493/0.786820/0.790656 | 0.883176/0.848053/0.865615 | 0.828135 |
| temporal audio | 43 | DEV | 0.709057/0.708341/0.708699 | 0.811042/0.812500/0.811771 | 0.760235 |
| temporal audio | 43 | TEST_NONE | 0.730448/0.733508/0.731978 | 0.870589/0.832216/0.851403 | 0.791690 |
| temporal audio | 43 | TEST_SOFT | 0.752115/0.754909/0.753512 | 0.872253/0.831131/0.851692 | 0.802602 |
| temporal audio | 43 | TEST_HARD | 0.772319/0.774505/0.773412 | 0.888623/0.837311/0.862967 | 0.818189 |
| temporal audio | 44 | DEV | 0.735201/0.733592/0.734396 | 0.797447/0.821317/0.809382 | 0.771889 |
| temporal audio | 44 | TEST_NONE | 0.777767/0.782953/0.780360 | 0.840002/0.846956/0.843479 | 0.811919 |
| temporal audio | 44 | TEST_SOFT | 0.794414/0.798231/0.796322 | 0.841036/0.847734/0.844385 | 0.820354 |
| temporal audio | 44 | TEST_HARD | 0.806129/0.809945/0.808037 | 0.855499/0.855499/0.855499 | 0.831768 |
| temporal audio | 45 | DEV | 0.697572/0.697224/0.697398 | 0.799034/0.799034/0.799034 | 0.748216 |
| temporal audio | 45 | TEST_NONE | 0.724572/0.724890/0.724731 | 0.849410/0.809085/0.829247 | 0.776989 |
| temporal audio | 45 | TEST_SOFT | 0.749585/0.750457/0.750021 | 0.857408/0.815141/0.836274 | 0.793148 |
| temporal audio | 45 | TEST_HARD | 0.768700/0.767829/0.768265 | 0.869070/0.816850/0.842960 | 0.805612 |
| temporal audio | 46 | DEV | 0.720495/0.719880/0.720187 | 0.801035/0.788813/0.794924 | 0.757556 |
| temporal audio | 46 | TEST_NONE | 0.767695/0.755935/0.761815 | 0.889479/0.838395/0.863937 | 0.812876 |
| temporal audio | 46 | TEST_SOFT | 0.803643/0.798463/0.801053 | 0.892854/0.841371/0.867112 | 0.834083 |
| temporal audio | 46 | TEST_HARD | 0.804714/0.797439/0.801077 | 0.899365/0.849013/0.874189 | 0.837633 |
| full R4 trial012 | 42 | DEV | 0.759197/0.758854/0.759025 | 0.873499/0.885020/0.879259 | 0.819142 |
| full R4 trial012 | 42 | TEST_NONE | 0.749184/0.749769/0.749477 | 0.848712/0.858579/0.853646 | 0.801561 |
| full R4 trial012 | 42 | TEST_SOFT | 0.759360/0.757326/0.758343 | 0.858541/0.870757/0.864649 | 0.811496 |
| full R4 trial012 | 42 | TEST_HARD | 0.762184/0.758727/0.760456 | 0.855251/0.862857/0.859054 | 0.809755 |
| full R4 trial012 | 43 | DEV | 0.718861/0.718193/0.718527 | 0.794893/0.813833/0.804363 | 0.761445 |
| full R4 trial012 | 43 | TEST_NONE | 0.813524/0.815573/0.814549 | 0.819828/0.839975/0.829902 | 0.822225 |
| full R4 trial012 | 43 | TEST_SOFT | 0.825096/0.824832/0.824964 | 0.818778/0.843862/0.831320 | 0.828142 |
| full R4 trial012 | 43 | TEST_HARD | 0.828496/0.827441/0.827968 | 0.848121/0.863222/0.855671 | 0.841820 |
| full R4 trial012 | 44 | DEV | 0.728525/0.727910/0.728217 | 0.818979/0.845142/0.832060 | 0.780139 |
| full R4 trial012 | 44 | TEST_NONE | 0.837944/0.838105/0.838025 | 0.799886/0.815453/0.807670 | 0.822847 |
| full R4 trial012 | 44 | TEST_SOFT | 0.857478/0.854491/0.855984 | 0.802712/0.819144/0.810928 | 0.833456 |
| full R4 trial012 | 44 | TEST_HARD | 0.869716/0.864687/0.867202 | 0.833767/0.836804/0.835285 | 0.851243 |
| full R4 trial012 | 45 | DEV | 0.725350/0.724803/0.725077 | 0.849758/0.867178/0.858468 | 0.791772 |
| full R4 trial012 | 45 | TEST_NONE | 0.767634/0.767120/0.767377 | 0.793884/0.793144/0.793514 | 0.780446 |
| full R4 trial012 | 45 | TEST_SOFT | 0.774540/0.771895/0.773217 | 0.789632/0.790435/0.790033 | 0.781625 |
| full R4 trial012 | 45 | TEST_HARD | 0.777391/0.774104/0.775747 | 0.809760/0.797743/0.803752 | 0.789749 |
| full R4 trial012 | 46 | DEV | 0.725537/0.725273/0.725405 | 0.859420/0.880909/0.870164 | 0.797785 |
| full R4 trial012 | 46 | TEST_NONE | 0.783609/0.787306/0.785458 | 0.786087/0.804730/0.795408 | 0.790433 |
| full R4 trial012 | 46 | TEST_SOFT | 0.798442/0.799500/0.798971 | 0.794406/0.815624/0.805015 | 0.801993 |
| full R4 trial012 | 46 | TEST_HARD | 0.798304/0.799390/0.798847 | 0.829657/0.837717/0.833687 | 0.816267 |
| equal-parameter shared fusion | 42 | DEV | 0.765546/0.765102/0.765324 | 0.875983/0.891228/0.883606 | 0.824465 |
| equal-parameter shared fusion | 42 | TEST_NONE | 0.757123/0.756877/0.757000 | 0.838672/0.851825/0.845248 | 0.801124 |
| equal-parameter shared fusion | 42 | TEST_SOFT | 0.767962/0.764983/0.766473 | 0.847576/0.863487/0.855531 | 0.811002 |
| equal-parameter shared fusion | 42 | TEST_HARD | 0.771905/0.767158/0.769531 | 0.857181/0.865940/0.861561 | 0.815546 |
| equal-parameter shared fusion | 43 | DEV | 0.725070/0.723922/0.724496 | 0.809179/0.827126/0.818152 | 0.771324 |
| equal-parameter shared fusion | 43 | TEST_NONE | 0.809871/0.810312/0.810092 | 0.817306/0.839029/0.828168 | 0.819130 |
| equal-parameter shared fusion | 43 | TEST_SOFT | 0.820138/0.818400/0.819269 | 0.820108/0.846296/0.833202 | 0.826236 |
| equal-parameter shared fusion | 43 | TEST_HARD | 0.824848/0.822011/0.823430 | 0.848121/0.863222/0.855671 | 0.839551 |
| equal-parameter shared fusion | 44 | DEV | 0.731746/0.731149/0.731448 | 0.826087/0.850658/0.838372 | 0.784910 |
| equal-parameter shared fusion | 44 | TEST_NONE | 0.833403/0.833244/0.833324 | 0.781322/0.784889/0.783106 | 0.808215 |
| equal-parameter shared fusion | 44 | TEST_SOFT | 0.853370/0.850259/0.851814 | 0.782765/0.785913/0.784339 | 0.818077 |
| equal-parameter shared fusion | 44 | TEST_HARD | 0.866569/0.861576/0.864073 | 0.803720/0.795529/0.799624 | 0.831848 |
| equal-parameter shared fusion | 45 | DEV | 0.739916/0.739549/0.739733 | 0.825880/0.843464/0.834672 | 0.787202 |
| equal-parameter shared fusion | 45 | TEST_NONE | 0.786689/0.785999/0.786344 | 0.800073/0.803115/0.801594 | 0.793969 |
| equal-parameter shared fusion | 45 | TEST_SOFT | 0.795062/0.791983/0.793522 | 0.798941/0.805642/0.802291 | 0.797907 |
| equal-parameter shared fusion | 45 | TEST_HARD | 0.798641/0.794614/0.796627 | 0.825204/0.820324/0.822764 | 0.809696 |
| equal-parameter shared fusion | 46 | DEV | 0.727451/0.727451/0.727451 | 0.833264/0.858518/0.845891 | 0.786671 |
| equal-parameter shared fusion | 46 | TEST_NONE | 0.805555/0.813255/0.809405 | 0.796081/0.815430/0.805755 | 0.807580 |
| equal-parameter shared fusion | 46 | TEST_SOFT | 0.830540/0.834875/0.832708 | 0.802494/0.826186/0.814340 | 0.823524 |
| equal-parameter shared fusion | 46 | TEST_HARD | 0.823424/0.828416/0.825920 | 0.840399/0.850852/0.845626 | 0.835773 |

Five-seed UAR/MF1 means are simple arithmetic values derived from frozen recorded rows, explicitly labeled and cross-checked in [DEPART_COMPARISON_READY_TABLE_EN.md](DEPART_COMPARISON_READY_TABLE_EN.md). The already-frozen five-seed Score/Mean summaries are: temporal audio D/P/Mean 0.721720/0.808569/0.765145; full R4 D/P/Mean 0.731250/0.848863/0.790057; shared fusion D/P/Mean 0.737690/0.844139/0.790914.

Final-Test frozen five-seed Score/Mean summaries (D/P/Mean) are: TEST_NONE audio 0.752954/0.848230/0.800592, R4 0.790977/0.816028/0.803502, shared 0.799233/0.812774/0.806003; TEST_SOFT audio 0.775925/0.850212/0.813068, R4 0.802296/0.820389/0.811343, shared 0.812757/0.817941/0.815349; TEST_HARD audio 0.788289/0.860246/0.824268, R4 0.806044/0.837490/0.821767, shared 0.815916/0.837049/0.826483.

## Internal DEPART-like baselines

V1 DEV: D `0.662558/0.661287/0.661923`; P `0.737474/0.751778/0.744626`; Mean `0.703274`. V2 DEV: D `0.620401/0.619801/0.620101`; P `0.785231/0.800854/0.793043`; Mean `0.706572`. Historical V1 Test Mean NONE/SOFT/HARD is `0.732454/0.742796/0.751864`; V2 is `0.722250/0.721935/0.730925`. These Test values are monitoring-only. Task UAR/MF1/Score is **NOT FOUND IN TRACKED EVIDENCE** and was not reconstructed.

## Stage-6 controls / ablations

All accepted Stage-6 selected DEV rows are in the CSV with task UAR/MF1/Score and summary-file/epoch locator. No historical monitoring Test is promoted to final evidence. The full R4 reference is additionally included above.

| Control/ablation | Selected DEV seed Means |
|---|---|
| no direct pseudo-supervision | s42=0.799900, s43=0.727236, s44=0.782498 |
| uniform accepted reliability | s42=0.813839, s43=0.759915, s44=0.781012 |
| no video online branch | s42=0.766129, s43=0.765037, s44=0.768127 |
| no audio online branch | s42=0.722839, s43=0.708919, s44=0.736350 |
| shared fusion control | s42=0.824465, s43=0.771324, s44=0.784910 |
| no semantic depression acceptance | s42=0.813509, s43=0.748237, s44=0.812478 |
| shuffled/mismatched pseudo control | s42=0.737158, s43=0.733074, s44=0.741122 |
| equal balancing | s42=0.777920, s43=0.779671, s44=0.779927 |
| static-STCH balancing | s42=0.787281, s43=0.776064, s44=0.777847 |
| progress weighting | s42=0.787299, s43=0.792364, s44=0.774590 |
| RA-STCH fixed composition | s42=0.782645, s43=0.781834, s44=0.753193 |
| sparse-MTL task-isolated | D-only and P-only task-specific selected rows; no two-task Mean is asserted (see CSV). |

The frozen Stage-6 claim ledger is unchanged; this inventory makes no new mechanism claim.

## Stage-3 text evidence

T1 transcript text-only DEV rows are in the CSV: seed42 D/P/Mean `0.721153/0.899381/0.810267`; seed43 `0.562911/0.825768/0.694339`; seed44 `0.596355/0.809025/0.702690`. T1 is standalone text evidence only, not additive A+V evidence. T2 has no performance rows because it was blocked before performance evaluation.

## Published DEPART reference

`docs/BDCC-10-00089.pdf` Table 7 (p.16) and Table 8 (p.19) record the binary D/P UAR and MF1 shown in the companion table. `docs/Ryumin_EMNLP.pdf`, named in the task Tier-3 list, is a different TACME paper and does not provide these DEPART D/P values. This is a source-name mismatch, not a numeric conflict: the published UAR agrees with `docs/BASELINES.md`.

## Comparison boundary and audit checks

DEPART UAR is not WSM Mean_Score. DEPART uses a different formulation/evaluation, and manually cleaned DEPART Test is not assumed identical to WSM TEST_HARD. A valid presentation compares UAR-to-UAR and MF1-to-MF1 where available, retains units/protocols, and makes no cross-study winning claim.

Frozen evidence supports: R4/shared each have `295239` trainable parameters; shared has `task_aware_fusion: false`; full R4 is task-aware; both are distinct from temporal audio; Final Test was invoked exactly once; and no post-Test role revision occurred. No numeric source conflicts were found.
