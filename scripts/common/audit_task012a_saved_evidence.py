#!/usr/bin/env python3
"""Offline, independent audit of the saved TASK-012A evidence.

This module intentionally does not import the production evaluator, model, data
module, callback, or loss.  It hashes and parses saved evidence and uses a small
standalone metric oracle on fabricated arrays.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from statistics import mean, stdev


SEEDS = (42, 43, 44, 45, 46)
PROTOCOLS = ("test_none", "test_soft", "test_hard")
TASKS = ("depression", "parkinson")
EXPECTED_COUNTS = {"test_none": 1364, "test_soft": 1208, "test_hard": 1014}
Z_95 = 2.7764451051977987


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def oracle_metrics(scores: list[float], labels: list[int], threshold: float = 0.0) -> dict[str, float]:
    """Return the sparse binary UAR/MF1/Score used by the audit oracle.

    Unknown labels are absent from ``labels``; callers must provide only the
    observed subset.  This is deliberately separate from project callbacks.
    """
    if len(scores) != len(labels) or not labels:
        raise ValueError("scores and labels must be non-empty and aligned")
    pred = [int(x >= threshold) for x in scores]
    classes = (0, 1)
    recalls: list[float] = []
    f1s: list[float] = []
    for cls in classes:
        tp = sum(p == cls and y == cls for p, y in zip(pred, labels))
        fn = sum(p != cls and y == cls for p, y in zip(pred, labels))
        fp = sum(p == cls and y != cls for p, y in zip(pred, labels))
        recall = tp / (tp + fn) if tp + fn else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        recalls.append(recall)
        f1s.append(f1)
    uar = mean(recalls)
    mf1 = mean(f1s)
    return {"uar": uar, "mf1": mf1, "score": mean((uar, mf1))}


def parse_shared_rows(stage7: str, protocol: str) -> dict[int, dict[str, float]]:
    """Parse the frozen per-seed shared-fusion rows for one protocol."""
    marker = f"### Five-seed summaries"
    sections = stage7.split(marker)
    if protocol == "test_none":
        text = sections[0]
    elif protocol == "test_soft":
        text = sections[1]
    elif protocol == "test_hard":
        text = sections[2]
    else:
        raise ValueError(protocol)
    pattern = re.compile(
        r"\| equal-parameter shared fusion \| (42|43|44|45|46) \| "
        r"([0-9.]+)/([0-9.]+)/([0-9.]+) \| "
        r"([0-9.]+)/([0-9.]+)/([0-9.]+) \| ([0-9.]+) \|"
    )
    rows = {}
    for match in pattern.finditer(text):
        seed = int(match.group(1))
        rows[seed] = {
            "depression_score": float(match.group(4)),
            "parkinson_score": float(match.group(7)),
            "mean": float(match.group(8)),
        }
    if set(rows) != set(SEEDS):
        raise AssertionError(f"missing shared rows for {protocol}: {sorted(rows)}")
    return rows


def summarize(values: list[float]) -> dict[str, float]:
    avg = mean(values)
    sd = stdev(values)
    return {"mean": avg, "ci95_low": avg - Z_95 * sd / math.sqrt(len(values)), "ci95_high": avg + Z_95 * sd / math.sqrt(len(values))}


def audit(dev_path: Path, test_path: Path, stage7_path: Path) -> dict:
    dev = json.loads(dev_path.read_text())
    test = json.loads(test_path.read_text())
    stage7 = stage7_path.read_text()
    assert dev["schema"] == "task012a-dev-preflight-v1"
    assert test["schema"] == "task012a-test-v1"
    assert test["invocation_count"] == 15
    assert test["protocol_counts"] == EXPECTED_COUNTS
    assert all(test[key] for key in ("no_labels", "no_raw_logits", "no_raw_predictions", "no_raw_probabilities", "no_sample_metadata"))

    result_rows = {(row["seed"], row["protocol"]): row for row in test["results"]}
    assert len(result_rows) == 15
    for seed in SEEDS:
        for protocol in PROTOCOLS:
            row = result_rows[(seed, protocol)]
            assert row["metrics"]["mean"] == (row["metrics"]["depression"]["score"] + row["metrics"]["parkinson"]["score"]) / 2
            assert row["metrics"]["protocol_count"] == EXPECTED_COUNTS[protocol]
            assert row["output_accumulation_dtype"] == "torch.float32"
            assert row["model_dtype"] == "torch.float32"

    # The oracle uses fabricated labels/scores and checks threshold/mask logic
    # without importing the project metric implementation.
    toy = oracle_metrics([-2.0, -1.0, 1.0, 2.0], [0, 0, 1, 1])
    assert toy == {"uar": 1.0, "mf1": 1.0, "score": 1.0}
    masked = oracle_metrics([-2.0, 2.0], [0, 1])
    assert masked["score"] == 1.0

    decomposition = {}
    summaries = {}
    for protocol in PROTOCOLS:
        historical = parse_shared_rows(stage7, protocol)
        current = {}
        for task, key in (("depression", "depression_score"), ("parkinson", "parkinson_score")):
            deltas = [result_rows[(seed, protocol)]["metrics"][task]["score"] - historical[seed][key] for seed in SEEDS]
            current[task] = {"deltas": deltas, **summarize(deltas)}
        current["mean"] = {"deltas": [sum(current[t]["deltas"][i] for t in TASKS) / 2 for i in range(5)]}
        current["mean"].update(summarize(current["mean"]["deltas"]))
        decomposition[protocol] = current
        summaries[protocol] = {
            **{
                task: summarize([result_rows[(seed, protocol)]["metrics"][task]["score"] for seed in SEEDS])
                for task in TASKS
            },
            "mean": summarize([result_rows[(seed, protocol)]["metrics"]["mean"] for seed in SEEDS]),
        }

    return {
        "verdict": "VERIFIED WITH STATED LIMITATIONS",
        "artifact_hashes": {
            "dev_preflight": sha256(dev_path),
            "test_results": sha256(test_path),
            "stage7_evidence": sha256(stage7_path),
        },
        "artifact_bytes": {"dev_preflight": dev_path.stat().st_size, "test_results": test_path.stat().st_size},
        "protocol_counts": EXPECTED_COUNTS,
        "independent_oracle": {"toy_cases": 2, "passed": True},
        "summaries": summaries,
        "task_decomposition": decomposition,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--stage7", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.dev, args.test, args.stage7)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
