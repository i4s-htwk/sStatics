
from dataclasses import dataclass
from functools import cached_property


@dataclass(eq=False)
class BarTemp:
    """Create a temperature load case for a statical system.

    Parameters
    ----------
    temp_o : :any:`float`
        Temperature change above the neutral axis [K].
    temp_u : :any:`float`
        Temperature change below the neutral axis [K].

    Raises
    ------
    ValueError
        :py:attr:`temp_o`, :py:attr:`temp_u` have to be greater than or
        equal to zero since its unit is Kelvin.
    """

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
        """Calculates the uniform temperature change in the unit Kelvin.

        Returns
        -------
        :any:`float`
            Averaged value of temperature changes above and below the neutral
            axis in Kelvin.

        Examples
        --------
        >>> from sstatics import BarTemp
        >>> temp = BarTemp(15, 30).temp_s
        22.5
        """
        return (self.temp_o + self.temp_u) / 2

    @cached_property
    def temp_delta(self):
        """Calculates the non-uniform temperature change in the unit Kelvin.

        Returns
        -------
        :any:`float`
            The temperature difference between the upper and lower side of the
            neutral axis, indicating the non-uniform temperature change in
            Kelvin.

        Examples
        --------
        >>> from sstatics import BarTemp
        >>> temp_diff = BarTemp(10, 20).temp_delta
        10.0
        """
        return self.temp_u - self.temp_o
