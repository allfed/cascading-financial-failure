import pytest
from typing import Callable
import numpy as np

from src.loss_transfer import beta, linear, quadratic


@pytest.fixture
def transfer_functions():
    return [beta(), linear(), quadratic()]


@pytest.mark.parametrize("impact", [-5, 0.5, 5, np.random.random()])
def test_transfer_functions(transfer_functions, impact):
    for t_func, bounds in transfer_functions:
        assert isinstance(t_func, Callable)
        assert isinstance(bounds, list)
        for val in bounds:
            assert isinstance(val, float)
            ret = t_func(val, 0.5)
            assert isinstance(ret, float)
            assert 0.0 <= ret <= 1.0
        if 0.0 <= impact <= 1.0:
            ret = t_func(2.0, impact)
            assert isinstance(ret, float)
            assert 0.0 <= ret <= 1.0
        else:
            with pytest.raises(AssertionError):
                t_func(2.0, impact)
    beta_func, lin_func, quad_func = [func for func, _ in transfer_functions]
    for alpha in np.linspace(-15, 15, 25):
        if alpha <= 1.0 or impact > 1.0 or impact < 0.0:
            with pytest.raises(AssertionError):
                beta_func(alpha, impact)
        else:
            ret = beta_func(alpha, impact)
            assert isinstance(ret, float)
            assert 0.0 <= ret <= 1.0
        if alpha < 0.0 or impact > 1.0 or impact < 0.0:
            with pytest.raises(AssertionError):
                lin_func(alpha, impact)
        else:
            ret = lin_func(alpha, impact)
            assert isinstance(ret, float)
            assert 0.0 <= ret <= 1.0
        if impact > 1.0 or impact < 0.0:
            with pytest.raises(AssertionError):
                quad_func(alpha, impact)
        else:
            ret = quad_func(alpha, impact)
            assert isinstance(ret, float)
            assert 0.0 <= ret <= 1.0
