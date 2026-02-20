from gabion.mesh.aggregator import fedavg


def test_fedavg_weighted_average() -> None:
    results = [
        {
            "worker_id": "a",
            "round_id": 1,
            "sample_count": 10,
            "weights": [1.0, 3.0],
            "metrics": {"loss": 1.0},
        },
        {
            "worker_id": "b",
            "round_id": 1,
            "sample_count": 30,
            "weights": [3.0, 7.0],
            "metrics": {"loss": 0.5},
        },
    ]
    assert fedavg(results, [0.0, 0.0]) == [2.5, 6.0]


def test_fedavg_fallback_when_no_samples() -> None:
    results = [
        {
            "worker_id": "a",
            "round_id": 1,
            "sample_count": 0,
            "weights": [99.0],
            "metrics": {},
        }
    ]
    assert fedavg(results, [1.0]) == [1.0]
