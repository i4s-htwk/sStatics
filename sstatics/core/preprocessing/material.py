
from dataclasses import dataclass


@dataclass(eq=False)
class Material:
    r"""Create a material for a statical system.

    Parameters
    ----------
    young_mod : :any:`float`
        Young's modulus (:math:`E`), a measure of the material's stiffness.
    poisson : :any:`float`
        Poisson's ratio (:math:`\nu`), the negative ratio of transverse to
        axial strain.
    shear_mod : :any:`float`
        Shear modulus (:math:`G`), a measure of the material's response
        to shear stress.
    therm_exp_coeff : :any:`float`
        Thermal expansion coefficient (:math:`\alpha_T`), describing how the
        material's dimensions change with temperature (in 1/K).

    Raises
    ------
    ValueError
        :py:attr:`young_mod`, :py:attr:`poisson`, :py:attr:`shear_mod`,
        and :py:attr:`therm_exp_coeff` have to be greater than
        zero.
    """

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
