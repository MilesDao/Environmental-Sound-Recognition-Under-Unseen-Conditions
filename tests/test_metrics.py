import math

import numpy as np
import pytest

from esr.metrics import mean_average_precision


def test_perfect_scores_give_one():
    y = np.array([[1, 0], [0, 1], [1, 1]], np.float32)
    assert mean_average_precision(y, y.copy()) == pytest.approx(1.0)


def test_map_skips_classes_without_positives():
    y = np.array([[1, 0, 0], [0, 1, 0]], np.float32)
    s = np.array([[0.9, 0.1, 0.5], [0.2, 0.8, 0.5]], np.float32)
    m = mean_average_precision(y, s)
    assert not math.isnan(m) and m == pytest.approx(1.0)


def test_map_no_positives_at_all_raises():
    with pytest.raises(ValueError):
        mean_average_precision(np.zeros((3, 2)), np.random.rand(3, 2))


def test_random_scores_are_worse_than_perfect():
    rng = np.random.default_rng(0)
    y = (rng.random((200, 10)) < 0.2).astype(np.float32)
    assert mean_average_precision(y, rng.random((200, 10))) < 0.5
