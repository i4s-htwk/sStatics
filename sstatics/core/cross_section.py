
from dataclasses import dataclass


@dataclass(eq=False)
class CrossSection:
    r"""Create a cross-section for a statical system.

    Parameters
    ----------
    mom_of_int : :any:`float`
        Moment of inertia (:math:`I`), a measure of an object's resistance to
        rotational acceleration.
    area : :any:`float`
        Area (:math:`A`), the cross-sectional area of the system.
    height : :any:`float`
        Height (:math:`h`) of the cross-section.
    width : :any:`float`
        Width of the cross-section.
    shear_cor : :any:`float`
        Shear correction factor (:math:`\kappa`), a dimensionless parameter.

    Raises
    ------
    ValueError
        :py:attr:`mom_of_int`, :py:attr:`area`, :py:attr:`height`,
        :py:attr:`width` and :py:attr:`shear_cor` have to be greater than
        zero.
    ValueError
        :py:attr:`area` has to be less than :py:attr:`width` times
        :py:attr:`height` are set to zero.
    """

    mom_of_int: float
    area: float
    height: float
    width: float
    shear_cor: float

    def __post_init__(self):
        if self.mom_of_int <= 0:
            raise ValueError('mom_of_int has to be greater than zero.')
        if self.height <= 0:
            raise ValueError('height has to be greater than zero.')
        if self.width <= 0:
            raise ValueError('width has to be greater than zero.')
        if not 0 <= self.area <= self.height * self.width:
            raise ValueError(
                f'area has to be greater than or equal to zero or less than '
                f'or equal to {self.width * self.height}.')
        if self.shear_cor <= 0:
            raise ValueError('shear_cor has to be greater than zero.')
