"""HTTP endpoints of the API, version 1."""

import math

from fastapi import APIRouter, HTTPException

from curvefit_api.api.schemas import ModelInfo, ParameterInfo
from curvefit_api.domain.models import MODELS, Model, UnknownModelError, get_model

router = APIRouter(prefix="/v1")


def _finite_or_none(value: float) -> float | None:
    """Represent an infinite bound as None, since JSON has no infinity."""
    return value if math.isfinite(value) else None


def _to_model_info(model: Model) -> ModelInfo:
    """Translate a domain model into its public API representation."""
    return ModelInfo(
        name=model.name,
        description=model.description,
        parameters=[
            ParameterInfo(
                name=name,
                lower_bound=_finite_or_none(lower),
                upper_bound=_finite_or_none(upper),
            )
            for name, lower, upper in zip(
                model.parameter_names, model.lower_bounds, model.upper_bounds, strict=True
            )
        ],
    )


@router.get("/models")
def list_models() -> list[ModelInfo]:
    """List all models available for fitting, sorted by name."""
    return [_to_model_info(model) for model in sorted(MODELS.values(), key=lambda m: m.name)]


@router.get("/models/{name}")
def get_model_info(name: str) -> ModelInfo:
    """Return one model by name."""
    try:
        model = get_model(name)
    except UnknownModelError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return _to_model_info(model)
