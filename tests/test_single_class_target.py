from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from iv_woe_filter import IVWOEFilter


def test_single_class_target_raises_value_error():
    X = pd.DataFrame({"feature": [1, 2, 3, 4]})
    y = np.array([1, 1, 1, 1])
    transformer = IVWOEFilter(min_iv=0.0, min_gini=0.0, n_jobs=1, verbose=False)

    with pytest.raises(ValueError, match="both classes 0 and 1"):
        transformer.fit(X, y)
