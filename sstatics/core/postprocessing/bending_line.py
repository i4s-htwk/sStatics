
from dataclasses import dataclass
from functools import cached_property
from typing import List

import numpy as np

from sstatics.core.postprocessing import DifferentialEquation


@dataclass
class BendingLine:
    """
    Combines local beam deformations and transforms them into the global
    coordinate system.

    This class collects the local deformations u (axial) and w (transverse)
    derived from DifferentialEquation objects and transforms them into the
    global coordinate system of the overall structure. For each beam,
    the global deformations are returned as lists of x- and z-coordinates
    representing the deformed shape.

    Parameters
    ----------
    dgl_list : list of DifferentialEquation
        List of DifferentialEquation objects, one per beam.
    """

    dgl_list: List[DifferentialEquation]

    def __post_init__(self):
        """
        Initializes global coordinate arrays and computes transformed values.

        Attributes created:
        ------------------
        x_global : list of np.ndarray
            Undeformed global x-coordinates per bar.
        z_global : list of np.ndarray
            Undeformed global z-coordinates per bar.
        u_global : list of np.ndarray
            Global axial displacements per bar.
        w_global : list of np.ndarray
            Global transverse displacements per bar.
        start_coords : list of tuple
            Starting coordinates (x, z) of each bar in global system.
        angles : list of float
            Inclination angles of each bar in radians.
        """
        self.x_global = []
        self.z_global = []
        self.u_global = []
        self.w_global = []
        self.start_coords = []
        self.angles = []

        self._compute()

    @cached_property
    def delta_max(self):
        """
        Computes the global maximum deformation scaling factor.

        The scaling factor Δ_max is defined as the maximum absolute value of
        any displacement (axial, transverse) or rotation over all bars. Used
        for normalized plotting of deformed beam lines.

        Returns
        -------
        float
            Maximum absolute deformation across all bars. Returns 1.0 if
            no deformation data is available to avoid division by zero.
        """
        values = []

        for dgl in self.dgl_list:
            deform = dgl.deform.flatten()
            values.extend([abs(deform[0]), abs(deform[1]), abs(deform[2])])
        return max(values) if values else 1.0

    def _compute(self):
        """
        Computes global coordinates and transforms local displacements.

        For each bar in `dgl_list`, the method performs:
        1. Local discretization (x coordinates along the bar).
        2. Transformation of local coordinates to global undeformed
           coordinates.
        3. Extraction of local displacements along the bar.
        4. Transformation of local displacements into global coordinates.
        5. Storage of the global start coordinates and bar inclination.
        """
        for dgl in self.dgl_list:
            bar = dgl.bar

            # Local discretization
            x_loc = dgl.x

            # Global undeformed coordinates
            x_glob = x_loc * np.cos(bar.inclination)
            z_glob = -x_loc * np.sin(bar.inclination)

            # Local displacements
            u_loc = dgl.deform_disc[:, 0]
            w_loc = dgl.deform_disc[:, 1]

            # Transform to global displacements
            u_glob = u_loc * np.cos(bar.inclination) + w_loc * np.sin(
                bar.inclination)
            w_glob = -u_loc * np.sin(bar.inclination) + w_loc * np.cos(
                bar.inclination)

            # Start coordinates in global system
            xi, zi = bar.node_i.x, bar.node_i.z

            # Store results
            self.x_global.append(x_glob + xi)
            self.z_global.append(z_glob + zi)
            self.u_global.append(u_glob)
            self.w_global.append(w_glob)
            self.start_coords.append((xi, zi))
            self.angles.append(bar.inclination)

    def deformed_lines(self):
        """
        Returns deformed global beam lines normalized by Δ_max.

        For each bar, computes the global deformed coordinates by adding
        the scaled displacements to the undeformed global coordinates.

        Returns
        -------
        list of tuple of np.ndarray
            Each tuple contains:
            - x_def : np.ndarray
                Deformed global x-coordinates for a bar.
            - z_def : np.ndarray
                Deformed global z-coordinates for a bar.
        """
        lines = []
        for xg, zg, ug, wg in zip(self.x_global, self.z_global,
                                  self.u_global, self.w_global):
            x_def = xg + ug / self.delta_max
            z_def = zg + wg / self.delta_max
            lines.append((x_def, z_def))
        return lines
