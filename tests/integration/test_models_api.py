import pytest
from fastapi.testclient import TestClient

from curvefit_api.domain.models import MODELS
from curvefit_api.main import create_app


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


def test_list_models_returns_every_registered_model(client: TestClient) -> None:
    response = client.get("/v1/models")

    assert response.status_code == 200
    assert [model["name"] for model in response.json()] == sorted(MODELS)


def test_model_parameters_match_the_domain_model(client: TestClient) -> None:
    response = client.get("/v1/models/gaussian")

    assert response.status_code == 200
    names = [parameter["name"] for parameter in response.json()["parameters"]]
    assert names == list(MODELS["gaussian"].parameter_names)


def test_infinite_bounds_are_returned_as_null(client: TestClient) -> None:
    response = client.get("/v1/models/exponential_decay")

    tau = next(p for p in response.json()["parameters"] if p["name"] == "tau")
    assert tau["lower_bound"] == 0.0
    assert tau["upper_bound"] is None


def test_unknown_model_returns_404(client: TestClient) -> None:
    response = client.get("/v1/models/no_such_model")

    assert response.status_code == 404
    assert "no_such_model" in response.json()["detail"]
