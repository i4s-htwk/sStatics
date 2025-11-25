
import numpy as np

from sstatics.core.preprocessing import CrossSection
from sstatics.core.preprocessing.geometry import Polygon

from sstatics.core.postprocessing import BarResult

from typing import Literal

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon


def plot_cross_section_with_shear_stress(cs, z_values, tau_values, title,
                                         v_z=0, invert_y=True):
    if not hasattr(cs, "geometry"):
        raise AttributeError(
            "CrossSection object must have a 'geometry' attribute.")

    fig, (ax_geo, ax_tau) = plt.subplots(1, 2, figsize=(10, 6), sharey=True,
                                         gridspec_kw={'width_ratios': [2, 1]})

    # --- Querschnitt zeichnen ---
    for poly in cs.geometry:
        points = getattr(poly, "points", None)
        if points is None:
            raise AttributeError(
                "Each Polygon must have a 'points' attribute.")
        patch = MplPolygon(points, closed=True, facecolor='lightgray',
                           edgecolor='black', alpha=0.8)
        ax_geo.add_patch(patch)

        # --- Pfeil und V_z Text ---

    # yb, zb = cs.boundary()
    # w = (yb[0] - yb[1]) / 2
    #
    # ax_geo.arrow(-w, zb[0] - 10, 0, 9, head_width=0.7,
    #              head_length=0.5,
    #              fc='blue', ec='blue')
    #
    # ax_geo.text(
    #     -w + 0.5,
    #     zb[0] - 6,
    #     f"V_z = {v_z:.2f}",  # Text mit 2 Nachkommastellen
    #     color='blue',
    #     fontsize=12,
    #     weight='bold'
    # )

    ax_geo.set_aspect('equal', 'box')
    ax_geo.set_xlabel("y")
    ax_geo.set_ylabel("z")
    ax_geo.set_title("Cross-section")
    ax_geo.autoscale_view()
    if invert_y:
        ax_geo.invert_yaxis()
    ax_geo.grid(True, linestyle='--', alpha=0.5)

    # --- Schubspannungsverlauf τ(z) ---
    ax_tau.set_xlim(0, max(tau_values) * 1.25)  # 10% Platz rechts lassen
    ax_tau.plot(tau_values, z_values, color='red', linewidth=2)
    ax_tau.fill_betweenx(z_values, 0, tau_values, color='red', alpha=0.3)

    ax_tau.set_title(title)
    ax_tau.grid(True, linestyle='--', alpha=0.5)

    # --- Maximalwert markieren ---
    max_idx = np.argmax(tau_values)
    z_max = z_values[max_idx]
    tau_max = tau_values[max_idx]

    ax_tau.plot(tau_max, z_max, 'ro')  # Punkt an Maximum

    # Grenzen abrufen
    x_min, x_max = ax_tau.get_xlim()
    text_offset = 0.05 * (x_max - x_min)  # 5% des Achsenbereichs
    text_x = tau_max + text_offset

    # Text innerhalb des Plots halten
    if text_x > x_max:
        text_x = tau_max - text_offset  # falls rechts nicht mehr passt

    ax_tau.text(text_x, z_max, f"{tau_max:.4f}", color='red', fontsize=10,
                va='bottom')

    ax_tau.text(tau_values[0] * 1.1, z_values[0], f"{tau_values[0]:.4f}",
                color='red', fontsize=10, va='bottom')
    ax_tau.text(tau_values[-1] * 1.1, z_values[-1], f"{tau_values[-1]:.4f}",
                color='red', fontsize=10, va='bottom')

    plt.tight_layout()
    plt.show()


def plot_line(x_values, z_values, title="Stress Distribution",
              xlabel="Bar Length", ylabel="Stress", color="blue"):
    """
    Plot a line given x and z values using matplotlib.

    Parameters:
        x_values (list or array): x-coordinates (e.g., position along the bar)
        z_values (list or array): z-coordinates (e.g., stress values)
        title (str): Plot title
        xlabel (str): Label for x-axis
        ylabel (str): Label for y-axis
        color (str): Line color
    """
    plt.figure(figsize=(8, 5))
    plt.plot(x_values, z_values, marker='o', linestyle='-', color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


class CrossSectionStress:
    """
    Calculate stresses in a cross-section under various loading conditions.

    This class provides methods to compute normal, bending, and shear stresses
    for a given cross-section geometry, and combines these effects as needed.

    Parameters
    ----------
    cross_section : CrossSection
        The cross-section object containing all geometric properties.

    Attributes
    ----------
    cross_section : CrossSection
        The cross-section object containing all geometric properties.
    _normal_stress_disc : float or None
        Cache for normal stress values.
    _bending_stress_disc : float or None
        Cache for bending stress values.
    _shear_stress_disc : float or None
        Cache for shear stress values.

    Examples
    --------
    Define a T-shaped cross-section geometry using polygons:

    >>> from sstatics.core.preprocessing import CrossSection
    >>> from sstatics.core.preprocessing.geometry.objects import Polygon
    >>> geometry = [
    ...     Polygon([(0, 0), (30, 0), (30, 3), (0, 3), (0, 0)]),
    ...     Polygon([(14, 3), (16, 3), (16, 43), (14, 43), (14, 3)])
    ... ]
    >>> cs = CrossSection(geometry=geometry)
    >>> cs_stress = CrossSectionStress(cross_section=cs)

    Calculate normal, bending, and shear stresses:

    >>> n = 2  # Normal force
    >>> m_yy = 10  # Bending moment
    >>> v_z = 10  # Shear force

    >>> # Normal stress calculation
    >>> cs_stress.normal_stress(n)
    0.011764705882352941

    >>> # Bending stress calculation
    >>> cs_stress.bending_stress(m_yy)
    0.010353175572198118

    >>> # Shear stress calculation at specific height
    >>> cs_stress.shear_stress(v_z)
    0.162453504934344

    >>> # Get shear stress distribution for plotting
    >>> z_values, tau_values = cs_stress.shear_stress_disc(
    ...     v_z=v_z, z_i=3.01, z_j=43, n_disc=10
    ... )

    >>> # Combined axial and bending stress
    >>> cs_stress.combine_axial_bending_stress(n=n, m_yy=m_yy)
    0.007931993275962822
    """

    def __init__(self, cross_section: CrossSection):
        """Initialize the CrossSectionStress calculator."""
        self.cross_section = cross_section

        # Initialize stress caches
        self._normal_stress_disc = None
        self._bending_stress_disc = None
        self._shear_stress_disc = None

    def normal_stress(self, n: float = 0, z=None):
        """
        Calculate normal stress due to axial force.

        Parameters
        ----------
        n : float, optional
            Normal force applied to the cross-section (default: 0).
        z : float, optional
            Vertical coordinate. Not used for normal stress but kept for
            interface consistency with other stress methods (default: None).

        Returns
        -------
        float
            Normal stress distribution (uniform across the cross-section).

        Raises
        ------
        ValueError
            If z is outside the cross-sectional bounds.

        Notes
        -----
        Normal stress is calculated as the ratio of normal force to
        cross-sectional area according to the formula: σ = N/A
        """
        # Get cross-section boundaries
        _, zb = self.cross_section.boundary()
        if z is not None:
            if zb[0] <= z <= zb[1]:
                raise ValueError(
                    f'z must be in the range of the cross-section '
                    f'z_min = {zb[0]} and z_max = {zb[1]}. Given z = {z}'
                )
        n_stress = n / self.cross_section.area
        self._normal_stress = n_stress  # Cache the result
        return n_stress

    def bending_stress(self, m_yy: float = 0, z=None):
        """
        Calculate bending stress due to bending moment.

        Parameters
        ----------
        m_yy : float, optional
            Bending moment about the y-axis (default: 0).
        z : float, optional
            Vertical coordinate from the neutral axis. If None, automatically
            selects the farthest point from the neutral axis (default: None).

        Returns
        -------
        float
            Bending stress at the specified location.

        Raises
        ------
        ValueError
            If z is outside the cross-sectional bounds.

        Notes
        -----
        Bending stress is calculated using the flexure formula: σ = My*c/Iy
        where c is the distance from the neutral axis.
        """
        # Get cross-section boundaries
        _, zb = self.cross_section.boundary()

        # Determine z if not specified
        if z is None:
            # Find the point farthest from the center of mass
            zs = self.cross_section.center_of_mass_z
            if abs(zb[0] - zs) > abs(zb[1] - zs):
                z = zb[0]
            else:
                z = zb[1]
            # Convert to distance from center of mass
            z = z - self.cross_section.center_of_mass_z
        else:
            # Validate z is within bounds
            if zb[0] <= z <= zb[1]:
                z = z - self.cross_section.center_of_mass_z
            else:
                raise ValueError(
                    f'z must be in the range of the cross-section '
                    f'z_min = {zb[0]} and z_max = {zb[1]}. Given z = {z}'
                )

        # Calculate bending stress
        i_yy = self.cross_section.mom_of_int
        return m_yy / i_yy * z

    def combine_axial_bending_stress(self, n: float = 0, m_yy: float = 0, z=0):
        """
        Calculate combined normal and bending stress.

        Parameters
        ----------
        n : float, optional
            Normal force applied to the cross-section (default: 0).
        m_yy : float, optional
            Bending moment about the y-axis (default: 0).
        z : float, optional
            Vertical coordinate for bending stress calculation (default: 0).

        Returns
        -------
        float
            Combined stress at the specified location.

        Notes
        -----
        When both normal force and bending moment are present, the total stress
        is the algebraic sum of axial stress and bending stress.
        """
        return self.normal_stress(n=n) + self.bending_stress(m_yy=m_yy, z=z)

    def zero_line(self, n: float = 0, m_yy: float = 0):
        """
        Calculate the location of the zero-stress line (neutral axis) under
        combined loading.

        Parameters
        ----------
        n : float, optional
            Normal force applied to the cross-section (default: 0).
        m_yy : float, optional
            Bending moment about the y-axis (default: 0).

        Returns
        -------
        float
            Vertical coordinate of the zero-stress line.

        Notes
        -----
        The zero-stress line is where the combined normal and bending stress
        equals zero. This is found by setting σ = N/A + My*c/I = 0 and solving
        for c, which gives c = -N*I/(M*A).
        """
        return (n * self.cross_section.mom_of_int /
                (m_yy * self.cross_section.area))

    def shear_stress(self, v_z: float = 0, z=None):
        """
        Calculate shear stress distribution due to transverse force.

        Parameters
        ----------
        v_z : float, optional
            Transverse force in the z-direction (default: 0).
        z : float, optional
            Vertical coordinate where shear stress is calculated. If None,
            defaults to the center of mass (default: None).

        Returns
        -------
        float
            Shear stress at the specified location.

        Notes
        -----
        This method calculates shear stress using Jourawski's formula:
        τ = V*S/(I*t), where S is the first moment of area, I is the second
        moment of area, and t is the width at the location of interest.

        **Calculation of the First Moment of Area (S):**
        The first moment of area S is calculated for the portion of the
        cross-section above the specified height z. The process involves:

        1. Creating a partial cross-section above the specified height z
        2. Calculating the area of this partial section
        3. Determining the distance between the centroid of the full
           cross-section and the centroid of the partial section
        4. Computing S as the product of the partial area and this distance

        Mathematically: S = A_partial * c, where c is the distance between the
        centroids of the full and partial cross-sections.

        **Important Restriction:**
        This implementation is most accurate for thin-walled cross-sections
        where the thickness/height ratio is small. For thick-walled sections
        or complex geometries, more advanced methods (such as elasticity
        solutions) or finite element methods should be used to capture the
        complete stress state accurately.
        """
        h = self.cross_section.height

        # Set default z position if not specified
        if z is None:
            z = self.cross_section.center_of_mass_z

        # Return zero at extreme fibers
        if z == 0 or z == h:
            return 0

        # Calculate shear stress based on cross-section geometry
        if self.cross_section.geometry:
            # Create a partial cross-section above height z to calculate
            # the first moment of area (Sy) for shear stress calculation

            # Get original boundaries
            yb, zb = self.cross_section.boundary()
            y_min, y_max = yb[0], yb[1]
            z_min, z_max = zb[0], zb[1]

            # Adjust z bounds based on height
            if z <= h / 2:
                z_min = z_min + z
            else:
                z_max = h / 2 + (z - h / 2)

            # Create polygon for partial cross-section
            points = [
                (y_min, z_min), (y_max, z_min),
                (y_max, z_max), (y_min, z_max), (y_min, z_min)]
            poly_boundary = Polygon(points=points, positive=False)

            # Combine geometries for partial cross-section
            geometry = (
                    self.cross_section.circular_sector +
                    [self.cross_section.polygon, poly_boundary])
            new_cs = CrossSection(geometry=geometry)

            # Calculate first moment of area
            sy = abs(new_cs.area *
                     (self.cross_section.center_of_mass_z -
                      new_cs.center_of_mass_z))

            # Get thickness at the specified height
            w = self.thickness_at_height(new_cs.polygon.points, z)
        else:
            # For simple rectangular sections
            w = self.cross_section.width
            # Calculate first moment of area for rectangular section
            poly = Polygon(points=[(0, 0), (w, 0), (w, z), (0, z), (0, 0)])
            sy = poly.static_moment[0]

        # Calculate final shear stress
        iy = self.cross_section.mom_of_int
        return v_z * sy / (iy * w)

    @staticmethod
    def thickness_at_height(points, z):
        """
        Computes the cross-section thickness (horizontal distance)
        at a given height z based on polygon points.

        Parameters
        ----------
        points : list[tuple[float, float]]
            List of (y, z) or (x, z) coordinates of the polygon contour.
            The polygon must be closed (first == last point).
        z : float
            The height at which the thickness should be computed.

        Returns
        -------
        float
            The horizontal thickness (distance in y-direction) at height z.
            Returns 0.0 if there are no points exactly at that height.
        """
        # Collect all y-values that lie exactly on height z
        y_values = [y for (y, z_point) in points if z_point == z]

        if not y_values:
            # No intersection found at this height
            return 0.0

        # Sort and compute horizontal thickness
        y_min = min(y_values)
        y_max = max(y_values)
        thickness = y_max - y_min

        return thickness

    def normal_stress_disc(self, n: float = 0):
        """
        Calculate normal stress distribution across the cross-section.

        Since normal stress is uniform across the cross-section, this method
        returns a constant stress value at the top and bottom boundaries.

        Parameters
        ----------
        n : float, optional
            Normal force applied to the cross-section (default: 0).

        Returns
        -------
        list of [tuple, list]
            A list containing:
            - Boundaries of the cross-section (zb)
            - List of normal stress values at top and bottom boundaries
              [normal_stress_top, normal_stress_bottom]

        Notes
        -----
        The normal stress is constant throughout the cross-section and is
        calculated using the formula: σ = N/A, where N is the normal force
        and A is the cross-sectional area.
        """
        # Get cross-section boundaries
        _, zb = self.cross_section.boundary()

        # Calculate normal stress (constant across entire cross-section)
        n_stress = self.normal_stress(n=n)

        # Store results with boundaries
        self._normal_stress_disc = [zb, [n_stress, n_stress]]

        return self._normal_stress_disc

    def bending_stress_disc(self, m_yy: float = 0):
        """
        Calculate bending stress distribution at the extreme fibers of the
        cross-section.

        This method computes the bending stress at the top and bottom
        boundaries of the cross-section, which will be the maximum and
        minimum values.

        Parameters
        ----------
        m_yy : float, optional
            Bending moment about the y-axis (default: 0).

        Returns
        -------
        list of [tuple, list]
            A list containing:
            - Boundaries of the cross-section (zb)
            - List of bending stress values at top and bottom boundaries
              [bending_stress_bottom, bending_stress_top]

        Notes
        -----
        Bending stress varies linearly across the cross-section, with maximum
        values at the extreme fibers (top and bottom). The stress is calculated
        using: σ = My*c/I, where c is the distance from the neutral axis.
        """
        # Get cross-section boundaries
        _, zb = self.cross_section.boundary()

        # Calculate bending stress at top and bottom boundaries
        self._bending_stress_disc = [
            zb,
            [
                self.bending_stress(m_yy=m_yy, z=zb[0]),  # Bottom boundary
                self.bending_stress(m_yy=m_yy, z=zb[1])  # Top boundary
            ]
        ]

        return self._bending_stress_disc

    def shear_stress_disc(self, v_z: float, z_i: float, z_j: float, n_disc=20):
        """
        Calculate shear stress distribution between two specified heights.

        This method computes shear stress values at multiple points along
        the height of the cross-section between given z-coordinates.

        Parameters
        ----------
        v_z : float
            Transverse force in the z-direction.
        z_i : float
            Starting z-coordinate for the calculation.
        z_j : float
            Ending z-coordinate for the calculation.
        n_disc : int, optional
            Number of discretization points between z_i and z_j (default: 20).

        Returns
        -------
        list of [array, list]
            A list containing:
            - Array of z-values from z_i to z_j with n_disc+1 points
            - List of shear stress values corresponding to each z-position

        Notes
        -----
        This method uses linear spacing to create discrete points between
        z_i and z_j, then calculates the shear stress at each point using
        the shear_stress method. The results can be used for plotting or
        further analysis of shear stress distribution.
        """
        # Create array of z-values between z_i and z_j
        z_values = np.linspace(z_i, z_j, n_disc + 1)

        # Calculate shear stress at each z position
        tau_values = [self.shear_stress(v_z=v_z, z=z) for z in z_values]

        # Store results for later use
        self._shear_stress_disc = [z_values, tau_values]

        return self._shear_stress_disc

    def plot(self, kind: Literal['normal', 'bending', 'shear']):
        if kind == 'normal':
            if self._normal_stress_disc is None:
                raise AttributeError(
                    "`normal_stress_disc` must be called before accessing "
                    "`plot`."
                )
            # geo_obj = GeoNormalStress(self, value=self.normal_stress)
            stress_plot = self._normal_stress_disc
        elif kind == 'bending':
            if self._bending_stress_disc is None:
                raise AttributeError(
                    "`bending_stress_disc` must be called before accessing "
                    "`plot`."
                )
            # geo_obj = GeoBendingStress(self, value=self.bending_stress)
            stress_plot = self._bending_stress_disc
        elif kind == 'shear':
            if self._shear_stress_disc is None:
                raise AttributeError(
                    "`bending_stress_disc` must be called before accessing "
                    "`plot`."
                )
            # geo_obj = GeoBendingStress(self, value=self.bending_stress)
            stress_plot = self._shear_stress_disc
        else:
            raise AttributeError('gibts nicht!')
        # geo_cs = GeoCrossSection(self.cross_section)
        # Renderer([geo_cs, geo_obj], mpl).show()
        plot_cross_section_with_shear_stress(
            self.cross_section,
            z_values=stress_plot[0],
            tau_values=stress_plot[1],
            title=kind
        )


class BarStressDistribution:
    """
    Compute stress distribution along a beam using solutions to its
    differential equations.

    This class calculates normal, shear, and bending stress distributions
    along a beam by combining the results from the differential equation
    solution for beam bending with local cross-section stress calculations.

    Parameters
    ----------
    bar : Bar
        Bar object with cross-section properties and geometry
    deform : Deformation
        Deformation parameters for the beam analysis
    force : Force
        Applied force parameters for the beam analysis
    disc : array-like
        Discretization parameters (array indices) defining the analysis points

    Attributes
    ----------
    bar : Bar
        Bar object containing cross-section properties
    deform : Deformation
        Deformation parameters used in the analysis
    force : Force
        Force parameters applied to the beam
    disc : array-like
        Discretization points along the beam
    internal_forces : ndarray
        Array of internal forces at discretization points computed from
        solving the differential equations of beam bending
    cross_section_stress : CrossSectionStress
        Calculator for cross-section stresses based on internal forces
    _stress_map : dict
        Mapping of stress types to their corresponding calculation parameters

    Examples
    --------
    >>> from sstatics.core.preprocessing import (Bar, System, Material, Node,
    ... CrossSection, BarLineLoad)
    >>> from sstatics.core.solution import FirstOrder
    >>> from sstatics.core.postprocessing import BarStressDistribution
    >>> # First, set up and solve the structural system
    >>> n1 = Node(0, 0, u='fixed', w='fixed')
    >>> n2 = Node(4, 0, w='fixed')
    >>> mat = Material(210_000_000, 0.1, 81_000_000, 0.1)
    >>> geometry = [
    ...     Polygon([(0, 0), (30, 0), (30, 3), (0, 3), (0, 0)]),
    ...     Polygon([(14, 3), (16, 3), (16, 43), (14, 43), (14, 3)])
    ... ]
    >>> cs = CrossSection(geometry=geometry)
    >>> line_load = BarLineLoad(1, 1)
    >>> b1 = Bar(n1, n2, cs, mat, line_loads=line_load)
    >>> system = System([b1])
    >>> solution = FirstOrder(system)
    >>>
    >>> # Then create stress distribution calculator
    >>> stressbar = BarStressDistribution(
    ...     bar=b1,
    ...     deform=solution.bar_deform_list[0],
    ...     force=solution.internal_forces[0],
    ...     disc=10
    ... )
    >>>
    >>> # Calculate stress at discretization points
    >>> normal_stresses = stressbar.compute('normal')
    >>> shear_stresses = stressbar.compute('shear', z=0.5)
    >>> bending_stresses = stressbar.compute('bending')

    """

    def __init__(self, bar, deform, force, disc):
        """
        Initialize the stress distribution calculator for a beam.

        Parameters
        ----------
        bar : Bar
            Bar object with cross-section properties
        deform : Deformation
            Deformation parameters for the beam analysis
        force : Force
            Applied force parameters for the beam analysis
        disc : array-like
            Discretization parameters (array indices) defining the analysis
            points

        Notes
        -----
        The 'internal_forces' attribute contains the results from solving the
        differential equations of beam bending, which provide the internal
        forces (normal force, shear force, bending moment) at each
        discretization point along the beam. This attribute replaces the
        previous objects that contained only this essential data.
        """
        self.bar = bar
        self.deform = deform
        self.force = force
        self.disc = disc

        # Compute internal forces at discretization points using beam theory
        # This solves the differential equation for beam bending
        self._beam_solver = BarResult(bar, deform, force, disc)
        self.internal_forces = self._beam_solver.forces_disc

        # Initialize cross-section stress calculator
        self.cross_section_stress = CrossSectionStress(bar.cross_section)

        # Mapping of stress types to their calculation parameters
        # Each entry contains:
        # - column index in the internal_forces array
        # - corresponding stress calculation function
        # - argument name for the force value in the stress function
        self._stress_map = {
            "normal":  (0, self.cross_section_stress.normal_stress, "n"),
            "shear":   (1, self.cross_section_stress.shear_stress, "v_z"),
            "bending": (2, self.cross_section_stress.bending_stress, "m_yy"),
        }

    def compute(self,
                stress_type: Literal['normal', 'shear', 'bending'],
                **kwargs
                ):
        """
        Compute stress of the specified type along the beam.

        This method calculates the stress distribution along the beam by
        applying the appropriate cross-section stress calculation at each
        discretization point based on the internal forces from beam theory.

        Parameters
        ----------
        stress_type : {'normal', 'shear', 'bending'}
            Type of stress to compute:
            - 'normal': Axial normal stress from normal force
            - 'shear': Transverse shear stress from shear force
            - 'bending': Bending stress from bending moment
        **kwargs : float
            Additional parameters to pass to the stress calculation function.
            Common parameters include:
            - z : float, optional
                Vertical coordinate for shear stress calculation
                (defaults to center of mass if not specified)

        Returns
        -------
        list of float
            Stress values at each discretization point along the beam.
            The list has the same length as the number of discretization
            points.

        Raises
        ------
        ValueError
            If an invalid stress type is provided.
        """
        # Validate the requested stress type
        if stress_type not in self._stress_map:
            raise ValueError(
                f"Unknown stress type: {stress_type}. "
                f"Available: {list(self._stress_map.keys())}")

        # Get the mapping parameters for the specified stress type
        component_index, stress_func, arg_name = self._stress_map[stress_type]

        # Extract the internal force component at each discretization point
        forces = self.internal_forces[:, component_index]

        # Compute stresses at all discretization points
        results = []
        for i in range(len(forces)):
            # Prepare parameters for the stress calculation function
            params = {arg_name: forces[i], **kwargs}
            # Calculate stress at this discretization point
            results.append(stress_func(**params))

        return results

    def plot(self, kind, **kwargs):
        plot_line(self._beam_solver.x,
                  self.compute(stress_type=kind, **kwargs))
