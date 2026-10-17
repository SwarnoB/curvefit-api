from typing import Any

import numpy as np
import pytest
from scipy.optimize import curve_fit

from curvefit_api.domain.fitting import fit
from curvefit_api.domain.models import FloatArray, get_model

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


FIT_CASES: dict[str, tuple[list[float], list[float]]] = {
    "linear": ([2.0, 1.0], [1.0, 0.0]),
    "exponential_decay": ([2.0, 1.5, 0.3], [1.5, 1.0, 0.0]),
    "gaussian": ([2.0, 2.5, 0.7, 0.3], [1.5, 2.2, 1.0, 0.0]),
    "lorentzian": ([2.0, 2.5, 0.7, 0.3], [1.5, 2.2, 1.0, 0.0]),
}

X_GRID = np.linspace(0.0, 5.0, 50)
X_GRID.setflags(write=False)  # shared by all tests: make accidental modification an error


def gaussian_for_curve_fit(x: FloatArray, *params: float) -> FloatArray:
    """Adapt GAUSSIAN.function to the f(x, *params) signature curve_fit expects."""
    return GAUSSIAN.function(x, np.array(params))


@pytest.mark.parametrize("model_name", sorted(FIT_CASES))
def test_fit_recovers_exact_parameters_without_noise(model_name: str) -> None:
    """With noise-free data, fit() recovers the true parameters."""
    model = get_model(model_name)
    true_parameters, initial_parameters = FIT_CASES[model_name]
    y = model.function(X_GRID, np.array(true_parameters))

    result = fit(model, X_GRID, y, initial_parameters=initial_parameters)

    assert result.success
    # Not machine precision: least_squares stops at its convergence tolerances (~1e-8).
    np.testing.assert_allclose(result.parameters, true_parameters, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("model_name", sorted(FIT_CASES))
def test_fit_estimates_agree_with_truth_within_standard_errors(model_name: str) -> None:
    """With noisy data, estimates lie within a few standard errors of the truth."""
    model = get_model(model_name)
    true_parameters, initial_parameters = FIT_CASES[model_name]
    noise = 0.05
    rng = np.random.default_rng(42)
    y = model.function(X_GRID, np.array(true_parameters)) + rng.normal(0.0, noise, X_GRID.size)

    result = fit(
        model,
        X_GRID,
        y,
        initial_parameters=initial_parameters,
        sigma=np.full(X_GRID.size, noise),
    )

    assert result.success
    deviation = np.abs(np.array(result.parameters) - np.array(true_parameters))
    # An estimate lies within 4 standard errors more than 99.99% of the time.
    # The seed is fixed, so the outcome is deterministic: it always passes or always fails.
    assert np.all(deviation < 4.0 * np.array(result.standard_errors))


def test_linear_fit_matches_weighted_least_squares_solution() -> None:
    """For the linear model, fit() reproduces the closed-form weighted solution."""
    true_parameters, initial_parameters = FIT_CASES["linear"]
    rng = np.random.default_rng(42)
    # Non-uniform sigma: with equal weights, a weighting bug would go unnoticed.
    sigma = rng.uniform(0.05, 0.3, X_GRID.size)
    y = LINEAR.function(X_GRID, np.array(true_parameters)) + rng.normal(0.0, sigma)

    result = fit(LINEAR, X_GRID, y, initial_parameters=initial_parameters, sigma=sigma)

    # Weighted linear least squares: beta = (X^T W X)^-1 X^T W y, cov = (X^T W X)^-1.
    design = np.column_stack([X_GRID, np.ones_like(X_GRID)])  # columns in parameter order
    weights = 1.0 / sigma**2
    normal_matrix = design.T @ (weights[:, np.newaxis] * design)
    expected_parameters = np.linalg.solve(normal_matrix, design.T @ (weights * y))
    expected_covariance = np.linalg.inv(normal_matrix)

    # Parameters are limited by the optimizer's stopping tolerance.
    np.testing.assert_allclose(result.parameters, expected_parameters, rtol=1e-7)
    # The linear Jacobian does not depend on the parameters, so the covariance is
    # exact up to rounding, wherever the optimizer stopped.
    np.testing.assert_allclose(result.covariance, expected_covariance, rtol=1e-10)


@pytest.mark.parametrize("sigma_given", [True, False])
def test_fit_agrees_with_curve_fit(sigma_given: bool) -> None:
    """fit() and scipy's curve_fit agree on parameters and covariance.

    Noise is added in both cases; only whether sigma is passed differs. With
    sigma omitted, this checks the reduced chi-squared scaling of the covariance.
    """
    true_parameters, initial_parameters = FIT_CASES["gaussian"]
    noise = 0.05
    rng = np.random.default_rng(42)
    y = GAUSSIAN.function(X_GRID, np.array(true_parameters)) + rng.normal(0.0, noise, X_GRID.size)
    sigma = np.full(X_GRID.size, noise) if sigma_given else None

    result = fit(GAUSSIAN, X_GRID, y, initial_parameters=initial_parameters, sigma=sigma)
    popt, pcov = curve_fit(
        gaussian_for_curve_fit,
        X_GRID,
        y,
        p0=initial_parameters,
        sigma=sigma,
        absolute_sigma=sigma_given,
        bounds=(GAUSSIAN.lower_bounds, GAUSSIAN.upper_bounds),
    )

    np.testing.assert_allclose(result.parameters, popt, rtol=1e-5)
    # atol only matters for off-diagonal entries close to zero.
    np.testing.assert_allclose(result.covariance, pcov, rtol=1e-4, atol=1e-10)
    assert result.covariance_scaled == (not sigma_given)


def test_reduced_chi_squared_is_consistent_with_one_for_correct_sigma() -> None:
    """With correctly specified sigma, reduced chi-squared is close to 1."""
    true_parameters, initial_parameters = FIT_CASES["exponential_decay"]
    x = np.linspace(0.0, 5.0, 2000)
    noise = 0.05
    rng = np.random.default_rng(42)
    y = EXPONENTIAL_DECAY.function(x, np.array(true_parameters)) + rng.normal(0.0, noise, x.size)

    result = fit(
        EXPONENTIAL_DECAY,
        x,
        y,
        initial_parameters=initial_parameters,
        sigma=np.full(x.size, noise),
    )

    assert result.degrees_of_freedom == x.size - len(true_parameters)
    # Reduced chi-squared has a standard deviation of about sqrt(2 / dof);
    # five of those is a very safe margin.
    tolerance = 5.0 * np.sqrt(2.0 / result.degrees_of_freedom)
    assert abs(result.reduced_chi_squared - 1.0) < tolerance


def test_fit_result_contains_plain_python_types() -> None:
    """FitResult holds built-in Python types, not NumPy scalars."""
    true_parameters, initial_parameters = FIT_CASES["gaussian"]
    y = GAUSSIAN.function(X_GRID, np.array(true_parameters))

    result = fit(GAUSSIAN, X_GRID, y, initial_parameters=initial_parameters)

    floats = [
        *result.parameters,
        *result.standard_errors,
        *(value for row in result.covariance for value in row),
        result.reduced_chi_squared,
    ]
    # `type(v) is float`, not isinstance: np.float64 is a subclass of float,
    # so isinstance would pass for NumPy values too and the test could never fail.
    assert all(type(value) is float for value in floats)
    assert type(result.success) is bool
    assert type(result.covariance_valid) is bool
    assert type(result.covariance_scaled) is bool
    assert type(result.degrees_of_freedom) is int
    assert type(result.n_function_calls) is int
