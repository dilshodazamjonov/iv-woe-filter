from __future__ import annotations

import numpy as np
import pytest

from iv_woe_filter.metrics import calculate_gini


def test_calculate_gini_raises_on_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        calculate_gini(np.array([0, 1, 0]), np.array([0.1, 0.9]))


def test_calculate_gini_returns_zero_for_all_nan_scores():
    value = calculate_gini(np.array([0, 1, 0, 1]), np.array([np.nan, np.nan, np.nan, np.nan]))

    assert value == 0.0
