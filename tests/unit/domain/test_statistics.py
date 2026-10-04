import numpy as np
import pytest

from curvefit_api.domain.statistics import reduced_chi_squared


def test_known_value() -> None:
    residuals = np.array([1.0, 2.0, 2.0, 1])
    assert reduced_chi_squared(residuals, num_params=2) == pytest.approx(5.0)


def test_perfect_fit() -> None:
    assert reduced_chi_squared(np.zeros(10), num_params=5) == 0.0


def test_accepts_plain_list() -> None:
    assert reduced_chi_squared([1.0, 2.0, 3.0, 4.0], num_params=2) == pytest.approx(15.0)


def test_raises_value_error_for_non_positive_dof() -> None:
    with pytest.raises(ValueError, match="Number of degrees of freedom must be positive"):
        reduced_chi_squared([1.0, 2.0], num_params=2)


@pytest.mark.parametrize("num_params", [0, -2])
def test_raises_for_non_positive_num_params(num_params: int) -> None:
    with pytest.raises(ValueError, match="Number of parameters must be positive"):
        reduced_chi_squared([1.0, 2.0], num_params=num_params)


@pytest.mark.parametrize("bad_values", [np.inf, np.nan, -np.inf])
def test_raises_for_non_finite_residuals(bad_values: float) -> None:
    with pytest.raises(ValueError, match="Residuals must be finite"):
        reduced_chi_squared([bad_values, 2.0, 3.0], num_params=1)


def test_raises_empty_residuals() -> None:
    with pytest.raises(ValueError, match="Number of degrees of freedom must be positive"):
        reduced_chi_squared([], num_params=1)
