# POST-CLOSURE EXPERIMENTAL EXTENSION — DOES NOT REVISE FROZEN WSM EVIDENCE

TASK-010A is a manager-authorized, DEV-only comparison of four
actual-parameter gradient-aware multi-task training mechanisms on the frozen
R4 composition. It is not a reopening of Stage 5, Stage 6, Stage 7, Final
Test, the paper, or any frozen method role.

The authoritative evidence is:

- [TASK010A_MTL_COMPARISON_EN.md](TASK010A_MTL_COMPARISON_EN.md);
- [task010a_mtl_runs.csv](task010a_mtl_runs.csv);
- [TASK010A_METHOD_SPEC.md](TASK010A_METHOD_SPEC.md);
- [task010a_config_equivalence.json](task010a_config_equivalence.json).

Exactly twelve completed config/seed results are reported: PCGrad,
CAGrad, GradNorm, and DB-MTL, each on seeds 42/43/44. A CAGrad seed43
process was interrupted after TRAIN optimizer steps during the user's stop;
the explicitly authorized replacement completed normally and is the only
accepted seed43 CAGrad result. This procedural exception is recorded without
changing the frozen scientific scope.

All checkpoint selection was DEV-only. No Test loader was exposed or
iterated, no Test metrics were inspected, and no method was promoted,
demoted, selected, or recommended. No significance claim is made from three
seeds. The extension remains descriptive evidence only.
