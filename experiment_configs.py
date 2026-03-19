CONFIGS = {
    "world": {
        "type": "coordinated",
        "horizon": 10,
        "endowment": 20.0,
    },
    "adversary": {
        "fair": {
            "type": "rl",
            "objective": "fair",
            "alpha": 0.1,
            "gamma": 0.95,
            "epsilon": 0.10,
        },
        "mid": {
            "type": "rl",
            "objective": "mixed",
            "alpha": 0.1,
            "gamma": 0.95,
            "epsilon": 0.10,
        },
        "max": {
            "type": "rl",
            "objective": "max",
            "alpha": 0.1,
            "gamma": 0.95,
            "epsilon": 0.10,
        },
        "worlds": {
            "independent": {
                "aggregation": "sample"
            },
            "coordinated": {
                "aggregation": "min",
                "shared_regime_prob": 0.90,
                "seed": 123,
            },
        },
    },
    "investor": {
        "type": "reactive",
    },
}