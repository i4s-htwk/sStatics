
from dataclasses import dataclass

from sstatics.core.postprocessing.results import SystemResult


@dataclass
class EquationOfWork:

    result_system_1: SystemResult
    result_system_2: SystemResult

    def delta_s1_s2(self):
        # gibt Verformung zurück
        return
