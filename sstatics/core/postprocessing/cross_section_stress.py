
from typing import Literal

import numpy as np

from sstatics.core.preprocessing import CrossSection, Polygon
from sstatics.core.postprocessing.graphic_objects import ObjectRenderer
from sstatics.core.postprocessing.graphic_objects.geo.cross_section import \
    CrossSectionGeo
from sstatics.core.postprocessing.graphic_objects.geo.state_line import \
    StateLineGeo


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
        geo_cs = CrossSectionGeo(self.cross_section)
        yb, zb = self.cross_section.boundary()
        translation = (-yb[1] * 3, zb[0])
        if kind == 'normal':
            if self._normal_stress_disc is None:
                raise AttributeError(
                    "`normal_stress_disc` must be called before accessing "
                    "`plot`."
                )
            x = self._normal_stress_disc[0]
            z = self._normal_stress_disc[1]

        elif kind == 'bending':
            if self._bending_stress_disc is None:
                raise AttributeError(
                    "`bending_stress_disc` must be called before accessing "
                    "`plot`."
                )
            x = self._bending_stress_disc[0]
            z = self._bending_stress_disc[1]
        elif kind == 'shear':
            if self._shear_stress_disc is None:
                raise AttributeError(
                    "`bending_stress_disc` must be called before accessing "
                    "`plot`."
                )
            x = self._shear_stress_disc[0]
            z = self._shear_stress_disc[1]
            translation = (-yb[1] * 2, x[0])
        else:
            raise AttributeError('gibts nicht!')

        geo_obj = StateLineGeo(
            [dict(x=x,
                  z=z,
                  translation=translation,
                  rotation=-np.pi/2
                  )],
            global_scale=geo_cs.global_scale,
            show_connecting_line=True,
            show_maximum=True,
        )

        ObjectRenderer([geo_cs, geo_obj], 'mpl').show()
