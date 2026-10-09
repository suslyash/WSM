from scripts.common.audit_task012a_saved_evidence import oracle_metrics, summarize


def test_independent_sparse_oracle_uses_zero_threshold_and_two_tasks():
    assert oracle_metrics([-2.0, -1.0, 1.0, 2.0], [0, 0, 1, 1]) == {
        "uar": 1.0,
        "mf1": 1.0,
        "score": 1.0,
    }
    result = oracle_metrics([-1.0, 1.0, 1.0, -1.0], [0, 0, 1, 1])
    assert result == {"uar": 0.5, "mf1": 0.5, "score": 0.5}


def test_frozen_five_seed_ci_uses_sample_standard_deviation():
    result = summarize([0.8, 0.8, 0.8, 0.8, 0.8])
    assert result["mean"] == 0.8
    assert result["ci95_low"] == result["ci95_high"] == 0.8
