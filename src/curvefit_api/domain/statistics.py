import numpy as np
from numpy.typing import ArrayLike

def reduced_chi_squared(residuals : ArrayLike , num_params : int) -> float:
    """ Reduced chi squared ( Chi-square per degree of freedom )

    Args: 
        residuals (ArrayLike) : (Observed - Predicted) data assumed to be weighted by 1/sigma^2 for each data point 
                                 i.e. residuals are (y_i - f(x_i,params))/sigma_i
        num_params (int) : number of parameters of the model

    Returns: 
        float : reduced chi squared value
    
    Raises: 
        ValueError : if number of degrees of freedom is not positive
    """
    r = np.asarray(residuals, dtype=float)
    num_dof = r.size - num_params
    
    if (num_dof <= 0):
        raise ValueError(f"Number of degrees of freedom must be positive.\n"
                         f"Currently we have {r.size} points and {num_params} parameters. \n"
                         f"Number of degrees of freedom is {num_dof}.\n"
                         "Please increase number of data points or decrease number of parameters.")
    
    return float(np.sum(r ** 2) / num_dof)
    
                         