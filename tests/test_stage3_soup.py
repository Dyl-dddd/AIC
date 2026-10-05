from collections import OrderedDict

import pytest
import torch

from scripts.build_semifinal_soups import interpolate_state_dicts
from scripts.run_semifinal_stage3 import ALPHAS


def test_interpolation_values_and_integer_buffer():
    old = OrderedDict(weight=torch.tensor([0.0, 2.0]), count=torch.tensor(1))
    new = OrderedDict(weight=torch.tensor([4.0, 6.0]), count=torch.tensor(2))
    result = interpolate_state_dicts(old, new, 0.25)
    assert torch.equal(result["weight"], torch.tensor([1.0, 3.0]))
    assert result["count"].item() == 1


def test_interpolation_rejects_incompatible_or_endpoint():
    with pytest.raises(ValueError):
        interpolate_state_dicts({"a": torch.zeros(1)}, {"b": torch.zeros(1)}, 0.5)
    with pytest.raises(ValueError):
        interpolate_state_dicts({"a": torch.zeros(1)}, {"a": torch.zeros(1)}, 0.0)


def test_stage3_is_bounded():
    assert ALPHAS == (0.25, 0.50, 0.75)

