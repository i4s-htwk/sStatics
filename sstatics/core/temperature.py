
from dataclasses import dataclass
from functools import cached_property


@dataclass(eq=False)
class BarTemp:
    """ TODO """

    temp_o: float
    temp_u: float

    def __post_init__(self):
        if self.temp_o < 0:
            raise ValueError(
                'temp_o has to be greater than or equal to zero since its '
                'unit is Kelvin.'
            )
        if self.temp_u < 0:
            raise ValueError(
                'temp_u has to be greater than or equal to zero since its '
                'unit is Kelvin.'
            )

    @cached_property
    def temp_s(self):
        """ TODO """
        return (self.temp_o + self.temp_u) / 2

    @cached_property
    def temp_delta(self):
        """ TODO """
        return self.temp_u - self.temp_o
