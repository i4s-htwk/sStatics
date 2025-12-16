
from dataclasses import dataclass

import numpy as np

from sstatics.core.logger_mixin import table_matrix
from sstatics.core.calc_methods import ForceMethod


@dataclass
class DMG(ForceMethod):

    support_moment = ForceMethod.redundants

    def log_linear_system(self, decimals: int = 6):
        """
        Log the linear system of equations (A x = b) for the force method.

        Builds the system matrix from the influence numbers (A, obtained
        via :meth:`influence_coef`) and the preliminary coefficients
        (b, obtained via :meth:`load_coef`). The system matrix with the
        right-hand side is tabulated with labeled columns and logged.

        Parameters
        ----------
        decimals : int, optional
            Number of decimal places used for floating point formatting in
            the tabulated output. Default is ``6``.

        Returns
        -------
        None
            The method logs the linear system but does not return data.
        """
        A = self.influence_coef * 6
        b = self.load_coef.reshape(-1, 1) * 6
        system_matrix = np.hstack([A, b])

        n = A.shape[0]
        headers = [f"M{i + 1}" for i in range(n)] + ["b"]

        self.logger.info("Linear system of equations (A x = b):\n%s",
                         table_matrix(matrix=system_matrix,
                                      column_names=headers,
                                      decimals=decimals)
                         )
