
from dataclasses import dataclass
from functools import cached_property

import numpy as np


@dataclass(eq=False)
class DegreesOfFreedom:
    """ TODO """

    x: float
    z: float
    phi: float

    @cached_property
    def vector(self):
        """ TODO """
        return np.array([[self.x], [self.z], [self.phi]])


NodeDisplacement = DegreesOfFreedom
""" Alias of :py:class:`DegreesOfFreedom` to make clear that this class has
the purpose to shift nodes. """
