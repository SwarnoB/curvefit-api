import numpy as np 
import pytest

from curvefit_api.domain.statistics import reduced_chi_squared

def test_known_value() -> None: 
    residuals = np.array([1.0, 2.0, 2.0, 1])
    assert reduced_chi_squared(residuals , num_params=2) == pytest.approx(5.0)   

def test_perfect_fit() -> None: 
    assert reduced_chi_squared(np.zeros(10) , num_params=5) == 0.0    

def test_accepts_plain_list() -> None: 
    assert reduced_chi_squared([1.0, 2.0, 3.0, 4.0] , num_params=2) == pytest.approx(15.0)

def test_raise_value_error_for_non_positive_dof() -> None: 
    with pytest.raises(ValueError, match="Number of degrees of freedom must be positive"):
        reduced_chi_squared([1.0, 2.0] , num_params=2) 


        


    
