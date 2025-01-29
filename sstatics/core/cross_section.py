
from dataclasses import dataclass


@dataclass(eq=False)
class CrossSection:
    """ TODO """

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
