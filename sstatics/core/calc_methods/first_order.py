
from dataclasses import dataclass

from sstatics.core.preprocessing.system import Bar
from sstatics.core.solution.solver import Solver


@dataclass(eq=False)
class FirstOrder(Solver):
    """
    First-order static solver derived from Solver.

    Inherits all functionality from Solver. Additionally, it validates that
    all bars in the system are instances of the base `Bar` class. If any
    `BarSecond` instances are found, a ValueError is raised.
    """

    def __post_init__(self):
        if hasattr(super(), '__post_init__'):
            super().__post_init__()

        for bar in self.system.bars:
            if not isinstance(bar, Bar):
                raise ValueError(
                    f"All bars must be instances of `Bar`. "
                    f"Found {type(bar).__name__}."
                )
