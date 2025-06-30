
from dataclasses import dataclass
from functools import cached_property

import numpy as np


@dataclass(eq=False)
class DegreesOfFreedom:
    """Represents the degrees of freedom in a 2D space with translation along
    :py:attr:`x`, :py:attr:`z`, and rotation :py:attr:`phi`.

    Parameters
    ----------
    x : :any:`float`
        The displacement or force along the x-axis.
    z : :any:`float`
        The displacement or force along the z-axis.
    phi : :any:`float`
        The rotational displacement (angle) or moment around the origin.
    """

    x: float
    z: float
    phi: float

    @cached_property
    def vector(self):
        """Returns the degrees of freedom as a 3x1 vector.

        Returns
        -------
         :any:`numpy.array`
            A 3x1 vector representing
            [:py:attr:`x`, :py:attr:`z`, :py:attr:`phi`].
        """
        return np.array([[self.x], [self.z], [self.phi]])


NodeDisplacement = DegreesOfFreedom
""" Alias of :py:class:`DegreesOfFreedom` to make clear that this class has
the purpose to shift nodes. """
