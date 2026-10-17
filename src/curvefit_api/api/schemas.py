"""Request and response schemas: the public contract of the API."""

from pydantic import BaseModel


class ParameterInfo(BaseModel):
    """A model parameter. A bound of None means unbounded."""

    name: str
    lower_bound: float | None
    upper_bound: float | None


class ModelInfo(BaseModel):
    """A model available for fitting."""

    name: str
    description: str
    parameters: list[ParameterInfo]
