
from dataclasses import dataclass


@dataclass(eq=False)
class Material:
    """ TODO """

    young_mod: float
    poisson: float
    shear_mod: float
    therm_exp_coeff: float

    def __post_init__(self):
        if self.young_mod <= 0:
            raise ValueError('young_mod has to be greater than zero.')
        if self.poisson <= 0:
            raise ValueError('poisson has to be greater than zero.')
        if self.shear_mod <= 0:
            raise ValueError('shear_mod has to be greater than zero.')
        if self.therm_exp_coeff <= 0:
            raise ValueError('therm_exp_coeff has to be greater than zero.')
