import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paired_inference_v2_3 import holm_adjust


def test_holm_known_example():
    assert np.allclose(holm_adjust([.03, .01, .04]), [.06, .03, .06])


def test_holm_monotonicity():
    p = [.01, .02, .5, .9]
    a = holm_adjust(p)
    assert np.all(np.diff(a) >= 0)
    assert np.all(a >= p)


def test_holm_zero_and_one():
    assert np.allclose(holm_adjust([0, 1]), [0, 1])


def test_holm_reject_invalid():
    with pytest.raises(ValueError):
        holm_adjust([-.1, .2])
    with pytest.raises(ValueError):
        holm_adjust([np.nan])
