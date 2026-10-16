from typing import Any

import numpy as np
import pytest

from curvefit_api.domain.fitting import fit
from curvefit_api.domain.models import get_model

LINEAR = get_model("linear")
EXPONENTIAL_DECAY = get_model("exponential_decay")
GAUSSIAN = get_model("gaussian")


def test_raises_when_x_and_y_have_different_length() -> None:
    x = np.linspace(0.0, 5.0, 10)
    y = np.linspace(1.0, 6.0, 9)
    with pytest.raises(ValueError, match="x, y and sigma must have the same length"):
        fit(LINEAR, x, y, initial_parameters=[1.0, 0.0])


def test_raises_when_sigma_has_wrong_length() -> None:
    x = np.linspace(0.0, 5.0, 10)
    with pytest.raises(ValueError, match="sigma must have the same length"):
        fit(LINEAR, x, 2.0 * x + 1.0, initial_parameters=[1.0, 0.0], sigma=np.ones(5))


def test_raises_when_x_is_not_1d() -> None:
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match="x must be one-dimensional"):
        fit(LINEAR, x, np.linspace(1.0, 6.0, 4), initial_parameters=[1.0, 0.0])


def test_raises_when_y_is_not_1d() -> None:
    y = np.array([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match="y must be one-dimensional"):
        fit(LINEAR, np.linspace(0.0, 2.0, 4), y, initial_parameters=[1.0, 0.0])


def test_raises_when_sigma_is_not_1d() -> None:
    x = np.linspace(0.0, 5.0, 4)
    y = 2.0 * x + 1.0
    sigma = np.array([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match="sigma must be one-dimensional"):
        fit(LINEAR, x, y, initial_parameters=[1.0, 0.0], sigma=sigma)


def test_raises_when_initial_parameter_is_not_1d() -> None:
    x = np.linspace(0.0, 5.0, 4)
    y = 2.0 * x + 1.0
    initial_parameters = np.array([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match="initial_parameters must be one-dimensional"):
        fit(LINEAR, x, y, initial_parameters=initial_parameters)


def valid_inputs() -> dict[str, Any]:
    x = np.linspace(0.0, 5.0, 10)
    return {
        "x": x,
        "y": 2.0 * x + 1.0,
        "initial_parameters": [1.0, 0.0],
        "sigma": np.ones_like(x),
    }


@pytest.mark.parametrize("name", ["x", "y", "sigma", "initial_parameters"])
@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_raises_for_non_finite_values(name: str, bad_value: float) -> None:
    inputs = valid_inputs()
    inputs[name][1] = bad_value
    with pytest.raises(ValueError, match="must contain only finite values"):
        fit(LINEAR, **inputs)


@pytest.mark.parametrize("sigma_value", [0.0, -1.0])
def test_raises_when_sigma_contains_zero_or_negative_values(sigma_value: float) -> None:
    x = np.linspace(0.0, 2.0, 3)
    y = 2.0 * x + 1.0
    sigma = np.array([1.0, sigma_value, 1.0])
    with pytest.raises(ValueError, match="sigma must be strictly positive"):
        fit(LINEAR, x, y, initial_parameters=[1.0, 0.0], sigma=sigma)


def test_raises_when_initial_parameters_have_wrong_length() -> None:
    x = np.linspace(0.0, 2.0, 3)
    y = 2.0 * x + 1.0
    with pytest.raises(ValueError, match="initial parameters, got"):
        fit(LINEAR, x, y, initial_parameters=[1.0, 0.0, 1.0])


@pytest.mark.parametrize("tau_value", [0.0, -5.0])
def test_raises_when_initial_parameters_not_strictly_inside_bounds(tau_value: float) -> None:
    x = np.linspace(0.0, 2.0, 10)
    y = np.exp(-1.0 * x / 1.0) + 1.0
    with pytest.raises(ValueError, match="must lie strictly"):
        fit(EXPONENTIAL_DECAY, x, y, initial_parameters=[1.0, tau_value, 0.0])


def test_raises_when_number_of_points_do_not_exceed_number_of_parameters() -> None:
    x = np.linspace(0.0, 2.0, 4)
    y = 0.5 * np.exp(-((x - 2.0) ** 2)) + 2.0
    with pytest.raises(ValueError, match="Need more data points than parameters"):
        fit(GAUSSIAN, x, y, initial_parameters=[1.0, 0.0, 1.0, 1.0])
