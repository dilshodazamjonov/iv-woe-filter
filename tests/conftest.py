from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_data() -> tuple[pd.DataFrame, np.ndarray]:
    """Create a mixed-type dataset with signal, noise, constants, and special codes."""
    rng = np.random.default_rng(42)
    n = 120

    df = pd.DataFrame(
        {
            "num_feat": rng.uniform(0, 100, n),
            "cat_feat": rng.choice(["A", "B", "C"], n, p=[0.45, 0.35, 0.20]),
            "constant_feat": np.ones(n, dtype=int),
            "special_feat": np.resize(np.array([10, 20, 30, -99]), n),
        }
    )

    y = (df["num_feat"] > 50).astype(int).to_numpy()
    noise = rng.choice([0, 1], n, p=[0.9, 0.1])
    y = np.bitwise_xor(y, noise)

    return df, y


@pytest.fixture
def output_dir(tmp_path) -> str:
    """Provide an isolated directory for audit artifact tests."""
    return str(tmp_path / "artifacts")
