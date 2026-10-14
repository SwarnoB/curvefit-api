"""Registry of fitting models: functions, their Jacobians and parameter bounds"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
ModelFunction = Callable[[FloatArray, FloatArray], FloatArray]


class UnknownModelError(ValueError):
    """Raised when an unknown model is requested."""


@dataclass(frozen=True)
class Model:
    """A fittable model y = f(x, parameters) with its analytical Jacobian

    `function(x, params)` returns an array shaped like x
    `jacobian(x, params)` returns an array of shape (len(x), len(params))
    """

    name: str
    description: str
    param_names: tuple[str, ...]
    function: ModelFunction
    jacobian: ModelFunction
    lower_bounds: tuple[float, ...]
    upper_bounds: tuple[float, ...]


def _linear(x: FloatArray, parameters: FloatArray) -> FloatArray:
    slope, intercept = parameters
    result: FloatArray = slope * x + intercept
    return result


def _linear_jacobian(x: FloatArray, parameters: FloatArray) -> FloatArray:
    _, _ = parameters
    result = np.column_stack([x, np.ones_like(x)])
    return result


def _exponential_decay(x: FloatArray, parameters: FloatArray) -> FloatArray:
    amplitude, tau, baseline = parameters
    result: FloatArray = amplitude * np.exp(-x / tau) + baseline
    return result


def _exponential_decay_jacobian(x: FloatArray, parameters: FloatArray) -> FloatArray:
    amplitude, tau, _ = parameters
    decay = np.exp(-x / tau)
    return np.column_stack([decay, amplitude * x * decay / tau**2, np.ones_like(x)])


def _gaussian(x: FloatArray, parameters: FloatArray) -> FloatArray:
    height, center, sigma_width, baseline = parameters
    result: FloatArray = height * np.exp(-((x - center) ** 2) / (2 * sigma_width**2)) + baseline
    return result


def _gaussian_jacobian(x: FloatArray, parameters: FloatArray) -> FloatArray:
    height, center, sigma_width, baseline = parameters
    diff = x - center
    exp_val = np.exp(-(diff**2) / (2 * sigma_width**2))
    result = np.column_stack(
        [
            exp_val,
            height * diff * exp_val / sigma_width**2,
            height * diff**2 * exp_val / sigma_width**3,
            np.ones_like(x),
        ]
    )
    return result


def _lorentzian(x: FloatArray, params: FloatArray) -> FloatArray:
    height, center, hwhm, baseline = params
    diff = x - center
    result: FloatArray = height * hwhm**2 / (diff**2 + hwhm**2) + baseline
    return result


def _lorentzian_jacobian(x: FloatArray, params: FloatArray) -> FloatArray:
    height, center, hwhm, _ = params
    diff = x - center
    denom = diff**2 + hwhm**2
    shape = hwhm**2 / denom
    result = np.column_stack(
        [
            shape,
            2.0 * height * shape * diff / denom,
            2 * height * hwhm * diff**2 / denom**2,
            np.ones_like(x),
        ]
    )
    return result


LINEAR = Model(
    name="linear",
    description="slope * x + intercept",
    param_names=("slope", "intercept"),
    function=_linear,
    jacobian=_linear_jacobian,
    lower_bounds=(-np.inf, -np.inf),
    upper_bounds=(np.inf, np.inf),
)


EXPONENTIAL_DECAY = Model(
    name="exponential_decay",
    description="amplitude * exp(-x / tau) + offset",
    param_names=("amplitude", "tau", "offset"),
    function=_exponential_decay,
    jacobian=_exponential_decay_jacobian,
    lower_bounds=(-np.inf, 0, -np.inf),
    upper_bounds=(np.inf, np.inf, np.inf),
)


GAUSSIAN = Model(
    name="gaussian",
    description="amplitude * exp(-(x - mean)**2 / (2 * std**2)) + baseline",
    param_names=("amplitude", "mean", "std", "baseline"),
    function=_gaussian,
    jacobian=_gaussian_jacobian,
    lower_bounds=(-np.inf, -np.inf, 0, -np.inf),
    upper_bounds=(np.inf, np.inf, np.inf, np.inf),
)


LORENTZIAN = Model(
    name="lorentzian",
    description="height * hwhm**2 / ((x - center)**2 + hwhm**2) + offset",
    param_names=("height", "center", "hwhm", "offset"),
    function=_lorentzian,
    jacobian=_lorentzian_jacobian,
    lower_bounds=(-np.inf, -np.inf, 0.0, -np.inf),
    upper_bounds=(np.inf, np.inf, np.inf, np.inf),
)


MODELS: dict[str, Model] = {
    model.name: model for model in (LINEAR, EXPONENTIAL_DECAY, GAUSSIAN, LORENTZIAN)
}


def get_model(name: str) -> Model:
    """Returns the registered model of the given name

    Args:
        name: Name of the model

    Returns:
        Model object

    Raises:
        UnknownModelError: If the model is not found
    """
    try:
        return MODELS[name]
    except KeyError:
        raise UnknownModelError(
            f"Unknown model {name!r}. Available models: {list(MODELS.keys())}"
        ) from None
