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


def _exponential_decay(x: FloatArray, parameters: FloatArray) -> FloatArray:
    amplitude, tau, offset = parameters
    result: FloatArray = amplitude * np.exp(-x / tau) + offset
    return result


def _exponential_decay_jacobian(x: FloatArray, parameters: FloatArray) -> FloatArray:
    amplitude, tau, _ = parameters
    decay = np.exp(-x / tau)
    return np.column_stack([decay, amplitude * x * decay / tau**2, np.ones_like(x)])


EXPONENTIAL_DECAY = Model(
    name="exponential_decay",
    description="amplitude * exp(-x / tau) + offset",
    param_names=("amplitude", "tau", "offset"),
    function=_exponential_decay,
    jacobian=_exponential_decay_jacobian,
    lower_bounds=(-np.inf, 0, -np.inf),
    upper_bounds=(np.inf, np.inf, np.inf),
)

MODELS: dict[str, Model] = {model.name: model for model in (EXPONENTIAL_DECAY,)}


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
