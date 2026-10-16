from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import least_squares

from curvefit_api.domain.models import FloatArray, Model
from curvefit_api.domain.statistics import reduced_chi_squared


@dataclass(frozen=True)
class FitResult:
    """Represents the result of a curve fitting operation.

    success = False must not be trusted
    """

    model_name: str
    parameter_names: tuple[str, ...]
    parameters: tuple[float, ...]
    standard_errors: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]
    covariance_valid: bool
    covariance_scaled: bool
    reduced_chi_squared: float
    degrees_of_freedom: int
    success: bool
    message: str
    n_function_calls: int


def _validate_inputs(
    model: Model,
    x: ArrayLike,
    y: ArrayLike,
    initial_parameters: ArrayLike,
    sigma: ArrayLike | None,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Convert inputs to float arrays and check them; return x, y, p0, sigma."""

    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    p0_arr = np.asarray(initial_parameters, dtype=np.float64)
    if sigma is not None:
        sigma_arr = np.asarray(sigma, dtype=np.float64)
    else:
        sigma_arr = np.ones_like(x_arr, dtype=np.float64)

    for name, arr in (
        ("x", x_arr),
        ("y", y_arr),
        ("sigma", sigma_arr),
        ("initial_parameters", p0_arr),
    ):
        if arr.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional, got {arr.ndim} dimensions.")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} must contain only finite values.")

    if not (len(y_arr) == len(x_arr) == len(sigma_arr)):
        raise ValueError(
            f"x, y and sigma must have the same length "
            f"(got {len(x_arr)}, {len(y_arr)} and {len(sigma_arr)})."
        )

    if np.any(sigma_arr <= 0.0):
        raise ValueError("sigma must be strictly positive.")

    if len(p0_arr) != len(model.parameter_names):
        raise ValueError(
            f"Model {model.name!r} expects {len(model.parameter_names)} "
            f"initial parameters, got {len(p0_arr)}."
        )

    for name, value, lower, upper in zip(
        model.parameter_names, p0_arr, model.lower_bounds, model.upper_bounds, strict=True
    ):
        if not lower < value < upper:
            raise ValueError(
                f"Initial value of {name!r} must lie strictly between "
                f"{lower} and {upper}, got {value}."
            )

    if x_arr.size <= len(model.parameter_names):
        raise ValueError(
            f"Need more data points than parameters: got {x_arr.size} points "
            f"and {len(model.parameter_names)} parameters."
        )

    return x_arr, y_arr, p0_arr, sigma_arr


def fit(
    model: Model,
    x: ArrayLike,
    y: ArrayLike,
    initial_parameters: ArrayLike,
    sigma: ArrayLike | None = None,
    max_function_evals: int = 1000,
) -> FitResult:

    x_arr, y_arr, p0_arr, sigma_arr = _validate_inputs(model, x, y, initial_parameters, sigma)

    def residuals(params: FloatArray) -> FloatArray:
        return (y_arr - model.function(x_arr, params)) / sigma_arr

    def jacobian(params: FloatArray) -> FloatArray:
        return -1.0 * model.jacobian(x_arr, params) / sigma_arr[:, np.newaxis]

    solution = least_squares(
        residuals,
        p0_arr,
        jac=jacobian,
        bounds=(model.lower_bounds, model.upper_bounds),
        method="trf",
        max_nfev=max_function_evals,
    )

    n_params = len(model.parameter_names)
    reduced_chi_sq = reduced_chi_squared(solution.fun, n_params)
    covariance_scaled = sigma is None

    jac = solution.jac
    if not isinstance(jac, np.ndarray):
        raise TypeError(f"Expected a dense Jacobian, got {type(jac).__name__}.")
    _, s, vt = np.linalg.svd(jac, full_matrices=False)
    threshold = np.finfo(float).eps * max(jac.shape) * s[0]
    covariance_valid = bool(np.all(s > threshold))

    if covariance_valid:
        covariance = vt.T @ np.diag(1.0 / s**2) @ vt
        if covariance_scaled:
            covariance *= reduced_chi_sq
        std_errors = np.sqrt(np.diag(covariance))
    else:
        covariance = np.full((n_params, n_params), np.inf)
        std_errors = np.full(n_params, np.inf)

    success = solution.status > 0
    message = solution.message
    n_func_calls = solution.nfev

    result = FitResult(
        model_name=model.name,
        parameter_names=model.parameter_names,
        parameters=tuple(solution.x.tolist()),
        standard_errors=tuple(std_errors.tolist()),
        covariance=tuple(tuple(row) for row in covariance.tolist()),
        covariance_valid=covariance_valid,
        covariance_scaled=covariance_scaled,
        reduced_chi_squared=reduced_chi_sq,
        degrees_of_freedom=solution.fun.size - n_params,
        success=success,
        message=message,
        n_function_calls=n_func_calls,
    )

    return result
