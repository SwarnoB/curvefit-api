"""Request and response schemas: the public contract of the API."""

from pydantic import BaseModel, ConfigDict


class Schema(BaseModel):
    """Base for all API schemas."""

    # JSON has no infinity or NaN: serialize both as null.
    model_config = ConfigDict(ser_json_inf_nan="null")


class ParameterInfo(Schema):
    """A model parameter. An infinite bound is serialized to null (unbounded)."""

    name: str
    lower_bound: float | None
    upper_bound: float | None


class ModelInfo(Schema):
    """A model available for fitting."""

    name: str
    description: str
    parameters: list[ParameterInfo]
