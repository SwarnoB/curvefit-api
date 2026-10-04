import numpy as np
from numpy.typing import ArrayLike


def reduced_chi_squared(residuals: ArrayLike, num_params: int) -> float:
    """Reduced chi squared ( Chi-square per degree of freedom )

    Args:
        residuals: (Observed - Predicted) data assumed to be weighted by 1/sigma for each data point
                                 i.e. residuals are (y_i - f(x_i,params))/sigma_i
        num_params: number of parameters of the model

    Returns:
        reduced chi squared value

    Raises:
        ValueError: if number of degrees of freedom is not positive
        ValueError: if residuals are not finite
        ValueError: if number of parameters is not positive
    """
    if num_params <= 0:
        raise ValueError("Number of parameters must be positive.")

    r = np.asarray(residuals, dtype=float)

    if not np.all(np.isfinite(r)):
        raise ValueError("Residuals must be finite.")

    num_dof = r.size - num_params

    if num_dof <= 0:
        raise ValueError(
            f"Number of degrees of freedom must be positive: "
            f"got {r.size} points and {num_params} parameters."
        )

    return float(np.sum(r**2) / num_dof)
