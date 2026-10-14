import numpy as np
import pytest

from curvefit_api.domain.models import MODELS, UnknownModelError, get_model

EXAMPLE_PARAMS: dict[str, list[float]] = {
    "linear": [1.0, 2.0],
    "exponential_decay": [2.0, 1.5, 0.3],
    "gaussian": [1.0, 0.0, 1.0, 0.0],
    "lorentzian": [1.0, 0.0, 1.0, 0.0],
}


def test_every_model_has_example_params() -> None:
    assert set(EXAMPLE_PARAMS) == set(MODELS)


@pytest.mark.parametrize("name", sorted(MODELS))
def test_every_model_has_consistent_definition(name: str) -> None:
    model = MODELS[name]
    n = len(model.param_names)
    assert len(model.lower_bounds) == n
    assert len(model.upper_bounds) == n
    assert len(EXAMPLE_PARAMS[name]) == n


@pytest.mark.parametrize("name", sorted(MODELS))
def test_analytic_jacobian_matches_finite_differences(name: str) -> None:
    model = MODELS[name]
    x = np.array([0.0, 10.0, 100.0])
    params = np.array(EXAMPLE_PARAMS[name])

    analytical = model.jacobian(x, params)

    finite_differences = np.empty_like(analytical)
    for i in range(params.size):
        step = 1e-6 * max(1.0, abs(params[i]))
        forward, backward = params.copy(), params.copy()
        forward[i] += step
        backward[i] -= step

        y_plus = model.function(x, forward)
        y_minus = model.function(x, backward)

        finite_differences[:, i] = (y_plus - y_minus) / (2 * step)

    np.testing.assert_allclose(analytical, finite_differences, rtol=1.0e-6, atol=1.0e-8)


def test_linear_returns_expected_values() -> None:
    model_name = "linear"
    model = get_model(model_name)
    x = np.array([0.0, 1.0])
    params = np.array(EXAMPLE_PARAMS[model_name])

    y = model.function(x, params)
    np.testing.assert_allclose(y, np.array([params[1], params[0] + params[1]]))


def test_exponential_decay_returns_expected_values() -> None:
    model_name = "exponential_decay"
    model = get_model(model_name)
    x = np.array([0.0, 1.5])
    params = np.array(EXAMPLE_PARAMS[model_name])

    y = model.function(x, params)
    np.testing.assert_allclose(y, np.array([params[0] + params[2], params[0] / np.e + params[2]]))


def test_gaussian_returns_expected_values() -> None:
    model_name = "gaussian"
    model = get_model(model_name)
    x = np.array([0.0, 1.0])
    params = np.array(EXAMPLE_PARAMS[model_name])

    y = model.function(x, params)
    np.testing.assert_allclose(
        y, np.array([params[0] + params[3], params[0] * np.exp(-1 / 2) + params[3]])
    )


def test_loretizan_returns_expected_values() -> None:
    model_name = "lorentzian"
    model = get_model(model_name)
    x = np.array([0.0, 1.0])
    params = np.array(EXAMPLE_PARAMS[model_name])

    y = model.function(x, params)
    np.testing.assert_allclose(
        y, np.array([params[0] + params[3], params[0] * params[2] ** 2 / (1 + params[2] ** 2)])
    )


def test_get_model_raises_for_unknown_name() -> None:
    with pytest.raises(UnknownModelError, match="Unknown model"):
        get_model("model_does_not_exist")
