
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry.polygon import orient


@dataclass(eq=False)
class Polygon:
    """
    Represents a two-dimensional polygon with optional interior holes,
    defined by a sequence of (y, z) coordinate tuples.

    Parameters
    ----------
    points : list of tuple of float
        A list of 2D coordinates defining the exterior boundary of the polygon.
        The polygon must be closed; i.e., the first and last point must be
        identical. A minimum of three distinct points (excluding the closing
        point) is required.
    holes : list of tuple of float, optional
        A list of holes, where each hole is defined by a list of coordinate
        tuples similar to the outer boundary. Each hole must also be closed.
        Default is an empty list (no holes).
    positive : bool, optional
        Specifies whether the polygon should be oriented positively
        (counterclockwise). This affects the internal construction of the
        Shapely polygon.
        Default is True.

    Notes
    -----
    The constructor validates the input geometry to ensure that the polygon
    is well-defined:
    - At least three distinct points are required to define a polygon.
    - The first and last point of the outer boundary (and any hole) must be
    the same to ensure closure.
    - All coordinate entries must be tuples of two numeric values.
    - No two consecutive points may be identical.

    Internally, the polygon is represented using a Shapely `Polygon` object
    and re-oriented if necessary.
    The coordinate arrays `self.y` and `self.z` are extracted for further
    geometric computations, such as area, centroid, and static moments.

    Examples
    --------
    Create a rectangular polygon with no holes:

    >>> from sstatics.core.preprocessing.geometry.objects import Polygon
    >>> outer = [(0, 0), (4, 0), (4, 2), (0, 2), (0, 0)]
    >>> p = Polygon(points=outer)

    Create a polygon with one rectangular hole:

    >>> hole = [[(1, 0.5), (3, 0.5), (3, 1.5), (1, 1.5), (1, 0.5)]]
    >>> p = Polygon(points=outer, holes=hole)

    Access area and centroid:

    >>> round(p.area, 2)
    6.0
    >>> round(p.center_of_mass_y, 2), round(p.center_of_mass_z, 2)
    (2.0, 1.0)
    """
    points: List[Tuple[float, float]]
    holes: List[List[Tuple[float, float]]] = field(default_factory=list)
    positive: bool = True
    polygon: ShapelyPolygon = field(init=False)

    def __post_init__(self):
        """Validates the input geometry and constructs an oriented Shapely
        polygon."""
        if len(self.points) < 3:
            raise ValueError("A polygon must have at least 3 points.")
        if self.points[0] != self.points[-1]:
            raise ValueError(
                "The first and last point in a polygon must be the same to "
                "close the shape."
            )
        for i, pt in enumerate(self.points):
            if not (isinstance(pt, tuple) and len(pt) == 2):
                raise ValueError(
                    f"Point #{i} is not a valid (x, y) tuple: {pt}"
                )
            if not all(isinstance(coord, (int, float)) for coord in pt):
                raise ValueError(
                    f"Coordinates of point #{i} must be numeric: {pt}"
                )
        for i in range(1, len(self.points) - 1):
            if self.points[i] == self.points[i - 1]:
                raise ValueError(
                    f"Consecutive duplicate points found at indices {i - 1} "
                    f"and {i}: {self.points[i]}"
                )
        distinct_points = set(self.points[:-1])
        if len(distinct_points) < 3:
            raise ValueError(
                "A polygon must contain at least 3 distinct points "
                "(excluding the closing point)."
            )
        if not isinstance(self.positive, bool):
            raise ValueError("The 'positive' attribute must be a boolean.")

        self.polygon = orient(
            ShapelyPolygon(shell=self.points, holes=self.holes),
            sign=1.0
        )
        self.y, self.z = self._extract_coord()

    def _extract_coord(self) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Extracts all vertex coordinates from the polygon's exterior and holes.

        Returns
        -------
        tuple of numpy.ndarray
            A tuple (y, z), where:

            - `y` contains all horizontal coordinates (e.g., local y-axis)
            - `z` contains all vertical coordinates (e.g., local z-axis)

            The coordinates include all vertices of the polygon’s exterior
            ring, followed by all vertices from interior rings (holes),
            appended in order.

        Notes
        -----
        These coordinates are used for geometric computations such as area and
        moment of inertia. The extraction is performed by traversing the
        exterior and each interior ring of the polygon sequentially.

        Examples
        --------
        >>> from sstatics.core.preprocessing.geometry.objects import Polygon

        Without holes:

        >>> outer = [(0, 0), (4, 0), (4, 2), (0, 2), (0, 0)]
        >>> p = Polygon(points=outer)
        >>> coord_y, coord_z = p._extract_coord()
        >>> coord_y
        array([0., 4., 4., 0., 0.])
        >>> coord_z
        array([0., 0., 2., 2., 0.])
        --------------------------------
        With holes:

        >>> hole = [[(1, 0.5), (3, 0.5), (3, 1.5), (1, 1.5), (1, 0.5)]]
        >>> p = Polygon(points=outer, holes=hole)
        >>> y
        array([0., 4., 4., 0., 0., 1., 3., 3., 1., 1.])
        >>> z
        array([0., 0., 2., 2., 0., 0.5, 0.5, 1.5, 1.5, 0.5])
        """
        ex, ey = self.polygon.exterior.xy
        y = list(ex)
        z = list(ey)
        for interior in self.polygon.interiors:
            ix, iy = interior.xy
            y.extend(ix)
            z.extend(iy)
        return np.array(y), np.array(z)

    @property
    def area(self) -> float:
        r"""Computes the signed area of the polygon, including all interior
        holes.

        Returns
        -------
        float
            The signed area of the polygon. A positive value indicates
            counterclockwise orientation (positive geometry),
            while a negative value indicates clockwise orientation.

        Notes
        -----
            The area is computed using the shoelace formula:

            .. math::
                A = \frac{1}{2} \left| \sum_{i=1}^{n} y_i \cdot z_{(i+1)} - y_{
                (i+1)} \cdot z_i \right|

                where
            .. math::
                (y_i, z_i)

            are the coordinates of the vertices.

        Examples
        --------
        >>> from sstatics.core.preprocessing.geometry.objects import Polygon
        >>> outer = [(0, 0), (4, 0), (4, 2), (0, 2), (0, 0)]
        >>> hole = [[(1, 0.5), (3, 0.5), (3, 1.5), (1, 1.5), (1, 0.5)]]
        >>> p = Polygon(points=outer, holes=hole)
        >>> round(p.area, 3)
        6.0
        """
        return 0.5 * abs(np.dot(self.y, np.roll(self.z, 1)) -
                         np.dot(self.z, np.roll(self.y, 1)))

    @property
    def static_moment(self) -> Tuple[np.float64, np.float64]:
        r"""
        Computes the static moments \( S_z \) and \( S_y \) with respect to the
        coordinate axes.

        Returns
        -------
        tuple of float or np.float64
            The static moments (S_z, S_y) of the polygon.

        Notes
        -----
            The static moments (also called first moments of area) are used
            to compute the centroid of the polygon.

        .. math::

            S_z = -\frac{1}{6} \sum (y_i z_{i-1} - y_{i-1} z_i)(z_i + z_{i-1})

        .. math::

            S_y = -\frac{1}{6} \sum (y_i z_{i-1} - y_{i-1} z_i)(y_i + y_{i-1})

        These are discrete approximations of the first moments:

        .. math::

            S_z = \int_A z \, \mathrm{d}A, \quad
            S_y = \int_A y \, \mathrm{d}A

        Examples
        --------
        Without holes:

        >>> from sstatics.core.preprocessing.geometry.objects import Polygon
        >>> outer = [(0, 0), (4, 0), (4, 2), (0, 2), (0, 0)]
        >>> p = Polygon(points=outer)
        >>> sz, sy = p.static_moment
        >>> round(sz, 2)
        10.67
        >>> round(sy, 2)
        21.33
        """
        y_, z_ = np.roll(self.y, 1), np.roll(self.z, 1)
        cross = self.y * z_ - y_ * self.z
        return (
            -np.dot(cross, self.z + z_) / 6,
            -np.dot(cross, self.y + y_) / 6,
        )

    @property
    def center_of_mass_y(self) -> np.float64:
        r"""Computes the y-coordinate of the centroid (center of mass).

        Returns
        -------
        np.float64
            The centroid's y-coordinate.

        Notes
        -----
        The centroid represents the geometric center of the polygon. It is
        computed as the ratio of the first moment of area about the z-axis to
        the total area.

        .. math::

            \bar{y} = \frac{S_y}{A}

        where

        - :math:`S_y` is the static moment about the z-axis
        - :math:`A` is the total area of the polygon

        This assumes uniform density and thickness.
        """
        return self.static_moment[1] / self.area

    @property
    def center_of_mass_z(self) -> np.float64:
        r"""Computes the z-coordinate of the centroid (center of mass).

        Returns
        -------
        np.float64
            The centroid's z-coordinate.

        Notes
        -----
        The centroid represents the geometric center of the polygon. It is
        computed as the ratio of the first moment of area about the y-axis to
        the total area.

        .. math::

            \bar{z} = \frac{S_z}{A}

        where

        - :math:`S_z` is the static moment about the y-axis
        - :math:`A` is the total area of the polygon

        This assumes uniform density and thickness.
        """
        return self.static_moment[0] / self.area

    @property
    def width(self) -> np.float64:
        r"""Computes the width of the polygon in the local y-direction.

        Returns
        -------
        np.float64
            The width of the polygon (i.e., the range of y-coordinates).

        Notes
        -----
        The width is defined as:

        .. math::

            w = y_\text{max} - y_\text{min}

        where :math:`y_\text{max}` and :math:`y_\text{min}` are the maximum
        and minimum horizontal coordinates of the polygon's boundary,
        including holes.
        """
        y = sorted(self.y)
        return y[-1] - y[0]

    @property
    def height(self) -> np.float64:
        r"""Computes the height of the polygon in the local z-direction.

        Returns
        -------
        np.float64
            The height of the polygon (i.e., the range of z-coordinates).

        Notes
        -----
        The height is defined as:

        .. math::

            h = z_\text{max} - z_\text{min}

        where :math:`z_\text{max}` and :math:`z_\text{min}` are the maximum
        and minimum vertical coordinates of the polygon's boundary, including
        holes.
        """
        z = sorted(self.z)
        return z[-1] - z[0]

    def mom_of_int_steiner(self, center) -> float:
        r"""
        Applies the parallel axis theorem (Steiner's theorem) to compute the
        moment of inertia about a shifted axis.

        Parameters
        ----------
        center : float
            The perpendicular distance from the centroid to the desired axis
            (in the same units as the polygon coordinates).

        Returns
        -------
        float
            The additional moment of inertia due to the axis shift, i.e.
            :math:`I_{\text{shift}} = A \cdot d^2`,
            where :math:`A` is the area and :math:`d` is the distance
            to the new axis.

        Notes
        -----
        This is used to shift the moment of inertia from the centroidal axis
        to another parallel axis according to the parallel axis theorem:

        .. math::
            I = I_c + A \cdot d^2
        """
        return self.area * center ** 2

    def _iyy_origin(self) -> np.float64:
        r"""
        Computes the second moment of area :math:`I_{yy}` about the y-axis
        (horizontal axis), referenced to the global origin (0, 0).

        Returns
        -------
        np.float64
            The second moment of area :math:`I_{yy}`, in units of length⁴.

        Notes
        -----
        This calculation is based on Green's Theorem for polygonal shapes:

        .. math::
            I_{yy} = \frac{1}{12} \sum (z_i^2 + z_i z_{i+1} + z_{i+1}^2)
            (x_i z_{i+1} - x_{i+1} z_i)

        where :math:`(y_i, z_i)` are the polygon vertex coordinates.
        """
        y1, z1 = np.roll(self.y, -1), np.roll(self.z, -1)
        a = self.y * z1 - y1 * self.z
        return (1 / 12) * np.sum((self.z ** 2 + self.z * z1 + z1 ** 2) * a)

    def _izz_origin(self) -> np.float64:
        r"""
        Computes the second moment of area :math:`I_{zz}` about the z-axis
        (vertical axis), referenced to the global origin (0, 0).

        Returns
        -------
        np.float64
            The second moment of area :math:`I_{zz}`, in units of length⁴.

        Notes
        -----
        Uses the polygon area integration formula (Green's Theorem):

        .. math::
            I_{zz} = \frac{1}{12} \sum (y_i^2 + y_i y_{i+1} + y_{i+1}^2)
            (y_i z_{i+1} - y_{i+1} z_i)
        """
        y1, z1 = np.roll(self.y, -1), np.roll(self.z, -1)
        a = self.y * z1 - y1 * self.z
        return (1 / 12) * np.sum((self.y ** 2 + self.y * y1 + y1 ** 2) * a)

    def _iyz_origin(self) -> np.float64:
        r"""
        Computes the product moment of inertia :math:`I_{yz}` with respect to
        the global origin (0, 0).

        Returns
        -------
        np.float64
            Product moment of inertia :math:`I_{yz}`, in units of length⁴.

        Notes
        -----
        This mixed moment measures the coupling between the y and z axes.

        Computed as:

        .. math::
            I_{yz} = -\frac{1}{24} \sum (y_i z_{i+1} + 2 y_i z_i +
            2 y_{i+1} z_{i+1} + y_{i+1} z_i) (y_i z_{i+1} - y_{i+1} z_i)
        """
        y1, z1 = np.roll(self.y, -1), np.roll(self.z, -1)
        a = self.y * z1 - y1 * self.z
        return (-1 / 24) * np.sum(
            (self.y * z1 + 2 * self.y * self.z + 2 * y1 * z1 + y1 * self.z) * a
        )

    @property
    def moments_of_inertia_tensor(self) -> np.ndarray:
        r"""
        Computes the second moment of area tensor (also known as the area
        moment of inertia tensor) relative to the centroid of the polygon.

        Returns
        -------
        ndarray
            A 2×2 symmetric numpy array representing the inertia tensor:

            .. math::
                \begin{bmatrix}
                I_{yy} & I_{yz} \\
                I_{zy} & I_{zz}
                \end{bmatrix}

            All entries are in units of length⁴.

        Notes
        -----
        The components are initially computed relative to the origin and
        then shifted to the centroid using the parallel axis theorem:

        .. math::
            I_{yy}^{\text{centroid}} = I_{yy}^0 - A \cdot z_c^2 \\
            I_{zz}^{\text{centroid}} = I_{zz}^0 - A \cdot y_c^2 \\
            I_{yz}^{\text{centroid}} = I_{yz}^0 - A \cdot y_c z_c

        where:
          - :math:`A` is the area,
          - :math:`(y_c, z_c)` is the centroid.

        This tensor is useful for calculating principal axes and moments,
        and for evaluating bending behavior in 2D beam cross-sections.
        """
        iyy_0 = self._iyy_origin()
        izz_0 = self._izz_origin()
        iyz_0 = self._iyz_origin()

        iyy = abs(iyy_0) - self.mom_of_int_steiner(self.center_of_mass_z)
        izz = abs(izz_0) - self.mom_of_int_steiner(self.center_of_mass_y)
        iyz = iyz_0 - self.area * self.center_of_mass_y * self.center_of_mass_z

        return np.array([[iyy, iyz], [iyz, izz]])

    @property
    def iyy(self):
        r"""Returns the second moment of area about the y-axis.

        Returns
        -------
        float
            Moment of inertia :math:`I_{yy}` about the y-axis,
            relative to the centroidal coordinate system,
            in units of length⁴.

        Notes
        -----
        This value is extracted from the full second moment of inertia
        tensor and reflects the resistance of the shape to bending
        about the y-axis.
        """
        return self.moments_of_inertia_tensor[0, 0]

    @property
    def izz(self):
        r"""Returns the second moment of area about the z-axis.

        Returns
        -------
        float
            Moment of inertia :math:`I_{zz}` about the z-axis,
            relative to the centroidal coordinate system,
            in units of length⁴.

        Notes
        -----
        This quantity indicates the resistance of the shape to bending
        about the z-axis and is derived from the inertia tensor.
        """
        return self.moments_of_inertia_tensor[1, 1]

    @property
    def iyz(self):
        r"""Returns the product moment of inertia.

        Returns
        -------
        float
            Product moment of inertia :math:`I_{yz}` relative to the
            centroidal coordinate system, in units of length⁴.

        Notes
        -----
        A nonzero product moment of inertia indicates that the principal
        axes of the shape are rotated relative to the coordinate system.
        This value is symmetric, i.e., :math:`I_{yz} = I_{zy}`.
        """
        return self.moments_of_inertia_tensor[0, 1]


@dataclass(eq=False)
class CircularSector:
    """
    Represents a two-dimensional circular sector defined by a center point,
    radius, angular span and a starting angle.

    Parameters
    ----------
    center : tuple of float
        The (y_0, z_0) coordinates of the center of the circular sector.
    radius : float
        The radius of the sector. Must be a positive value.
    angle : float
        The angular span of the sector in radians. Must be a
        non-zero number and lie within the range ± 2π.
    start_angle : float
        The angle (in radians) where the sector arc begins, measured
        clockwise from the positive y'-axis.
    positive : bool, optional
        Specifies whether the sector is a partial or cutout area of the total
        cross-section. If “True,” the sector contributes positively to
        calculations (e.g., area and moment). If “False,” it is treated as a
        ‘cutout’ and its contributions are subtracted.
        The default value is “True.”

    Notes
    -----
    The constructor validates input to ensure that the circular sector is
    well-defined:
    - The center must be a 2D coordinate tuple of numeric values.
    - The radius must be greater than zero.
    - The angle must not be zero and must be in the range (-2π <= angle <= 2π).
    - Start angle must be a number (in radians)."
    - The 'positive' attribute must be a boolean value.

    The sector is internally represented with geometric formulas for
    properties such as area, centroid, and moments of inertia.
    Optionally, the sector can be discretized into a polygon via
    the `convert_to_polygon()` method.

    Examples
    --------
    Create a circular sector in the first quadrant:

    >>> from sstatics.core.preprocessing.geometry.objects import CircularSector
    >>> import numpy as np
    >>> c = CircularSector(center = (0,0), radius = 1, angle = np.pi/4,
    >>>                    start_angle = np.pi/8, positive = True
    >>>                    )

    Creates the same circular sector with a negative angle range:

    >>> c = CircularSector(center = (0,0), radius = 1, angle = -np.pi/4,
    >>>                    start_angle = 3*np.pi/8, positive = True
    >>>                    )

    Access area and centroid:

    >>> round(c.area, 2)
    0.39
    >>> round(c.center_of_mass_y, 2), round(c.center_of_mass_z, 2)
    (0.46, 0.46)
    """
    center: Tuple[float, float]
    radius: float
    angle: float = 0
    start_angle: float = 0
    positive: bool = True

    def __post_init__(self):
        """
        Validates the input geometry (see the description of the
        class CircularSector)
        """
        if not isinstance(self.center, tuple) or len(self.center) != 2:
            raise ValueError(
                "Center must be a tuple of two numeric values (y, z)."
            )
        if not all(isinstance(coord, (int, float)) for coord in self.center):
            raise ValueError(
                "Center coordinates must be numeric (float or int)."
            )
        if not isinstance(self.radius, (int, float)) or self.radius <= 0:
            raise ValueError("Radius must be a positive number.")
        if not isinstance(self.angle, (int, float)) or self.angle == 0:
            raise ValueError("Angle must be a non-zero number (in radians).")
        if not isinstance(self.angle, (int, float)) or abs(
                self.angle) > 2 * np.pi:
            raise ValueError("Angle must be a number in the range ± 2π.")
        if not isinstance(self.positive, bool):
            raise ValueError(
                "The 'positive' attribute must be a boolean value."
            )
        if not isinstance(self.start_angle, (int, float)):
            raise ValueError("Start angle must be a number (in radians).")

    def convert_to_polygon(self, num_points=500):
        r"""
        Converts the circular sector into an approximate polygon
        representation.

        Parameters
        ----------
        num_points : int, optional
            Number of points used to discretize the arc of the sector.
            Higher values yield a finer approximation. Default is 500.

        Returns
        -------
        Polygon
            A polygonal approximation of the circular sector, represented
            by a `Polygon` object.

        Notes
        -----
        The polygon is constructed by sampling `num_points` points
        along the arc defined by the sector's start angle :math:`\varphi_1`,
        opening angle :math:`\varphi_M`, and radius :math:`r`. These points are
        connected together with the center point of the sector to form a closed
        polygonal shape.

        - The starting point of the arc is determined by `start_angle`.
        - Points are sampled at equidistant angles along the arc.
        - The last arc point is explicitly added (if not already present),
          to avoid floating point errors.
        - The center point is both the first and last point in the polygon
          to ensure proper closure.

        The resulting polygon can be used for geometric computations such
        as area, static moments, and moment of inertia using existing
        polygon-based methods.
        """
        y_0, z_0 = self.center
        points = []

        is_full_circle = abs(abs(self.angle) - 2 * np.pi) < 1e-12

        if not is_full_circle:
            points.append((y_0, z_0))

        for i in range(num_points + 1):
            theta = self.start_angle + i * self.angle / num_points
            y = y_0 + self.radius * np.cos(theta)
            z = z_0 + self.radius * np.sin(theta)
            points.append((y, z))

        end_point = (
            round(y_0 + self.radius * np.cos(self.start_angle + self.angle),
                  12),
            round(z_0 + self.radius * np.sin(self.start_angle + self.angle),
                  12)
        )

        if end_point != points[-1]:
            points.append(end_point)

        if not is_full_circle:
            points.append((y_0, z_0))
        return Polygon(points, positive=self.positive)

    @property
    def area(self):
        r"""
        Computes the area of the circular sector.

        Returns
        -------
        float
            The area of the circular sector.

        Notes
        -----
            The area of a circular sector is computed using the formula:

            .. math::
                A = \frac{1}{2} \cdot r^2 \cdot \mid \varphi_M \mid

            where

            - :math:`r` is the radius of circular sector
            - :math:`\varphi_M` is the angular span of the sector in radians

        Examples
        --------
        >>> from sstatics.core.preprocessing.geometry.objects import
        >>>     CircularSector
        >>> import numpy as np
        >>> c = CircularSector(center = (0,0), radius = 1, angle = np.pi/4,
        >>>                    start_angle = np.pi/8, positive = True
        >>>                    )
        >>> round(c.area, 3)
        0.393
        """
        return 0.5 * self.radius ** 2 * abs(self.angle)

    @property
    def center_of_mass_y(self):
        r"""
        Computes the y-coordinate of the centroid (center of mass).

        Returns
        -------
        np.float64
            The centroid's y-coordinate.

        Notes
        -----
        The centroid represents the geometric center of the circular sector.
        It is used to compute the statics moments.

        .. math::

            y_c = \frac{2 \cdot r}{3 \cdot \varphi_M}
                    \cdot (sin(\varphi_2) - sin(\varphi_1)) + y_0

        where

        - :math:`r` is the radius of circular sector
        - :math:`\varphi_M` is the angular span of the sector in radians
        - :math:`\varphi_1` is the starting angle
        - :math:`\varphi_2` is the ending angle
        - :math:`y_0` is the y-coordinate of the center of the circular sector

        This assumes uniform density and thickness.
        """
        return (
                2 * self.radius *
                (np.sin(self.angle + self.start_angle) - np.sin(
                    self.start_angle)) / (3 * self.angle) + self.center[0]
                )

    @property
    def center_of_mass_z(self):
        r"""
        Computes the z-coordinate of the centroid (center of mass).

        Returns
        -------
        np.float64
            The centroid's z-coordinate.

        Notes
        -----
        The centroid represents the geometric center of the circular sector.
        It is used to compute the statics moments.

        .. math::

            z_c = \frac{2 \cdot r}{3 \cdot \varphi_M}
                    \cdot (cos(\varphi_1) - cos(\varphi_2)) + z_0

        where

        - :math:`r` is the radius of circular sector
        - :math:`\varphi_M` is the angular span of the sector in radians
        - :math:`\varphi_1` is the starting angle
        - :math:`\varphi_2` is the ending angle
        - :math:`z_0` is the z-coordinate of the center of the circular sector

        This assumes uniform density and thickness.
        """
        return (
                2 * self.radius *
                (np.cos(self.start_angle) - np.cos(
                    self.angle + self.start_angle)) / (3 * self.angle) +
                self.center[1]
                )

    @property
    def static_moment(self):
        r"""
        Computes the static moments :math:`S_y` and :math:`S_z` of the circular
        sector with respect to the coordinate axes.

        Returns
        -------
        tuple of np.float64
            The static moments (S_y, S_z) of the circular sector.

        Notes
        -----
            The static moments (also called first moments of area) are used to
            compute the centroid of the entire composite cross-section
            (see class CrossSection).

            .. math::
                S_y = A \cdot z_c

            .. math::
                S_z = A \cdot y_c

        Examples
        --------
        >>> from sstatics.core.preprocessing.geometry.objects import
        >>> CircularSector
        >>> import numpy as np
        >>> c = CircularSector(center = (0,0), radius = 1, angle = np.pi/2,
        >>>                    start_angle = 0, positive = True
        >>>                    )
        >>> round(c.static_moment[0], 3)
        0.333
        >>> round(c.static_moment[1], 3)
        0.333
        """
        s_y = self.area * self.center_of_mass_z
        s_z = self.area * self.center_of_mass_y

        return [s_y, s_z]

    def mom_of_int_steiner(self, distance):
        r"""
        Applies the parallel axis theorem (Steiner's theorem) to compute the
        moment of inertia about a shifted axis.

        Parameters
        ----------
        distance : float
            The perpendicular distance from the centroid to the desired axis
            (in the same units as the circular sector coordinates).

        Returns
        -------
        float
            The additional moment of inertia due to the axis shift, i.e.
            :math:`I_{\text{shift}} = A \cdot d^2`,
            where :math:`A` is the area and :math:`d` is the distance
            to the new axis.

        Notes
        -----
        This is used to shift the moment of inertia from one axis
        to another parallel axis according to the parallel axis theorem:

        .. math::
            I = I_c + A \cdot d^2
        """
        return self.area * distance ** 2

    @property
    def mom_of_int_y(self) -> float:
        r"""
        Computes the second moment of area :math:`I_{\bar{y}}` about the
        \bar{y}-axis, referenced to the centroid coordinate system.

        Returns
        -------
        np.float64
            The second moment of area :math:`I_{\bar{y}}`, in units of length⁴.

        Notes
        -----
        The components are initially computed relative to the center of the
        circular sector :math:`(y_0, z_0)` and then shifted to the centroid
        using the parallel axis theorem:

        .. math::
            I_{y'} = \frac{r^4}{8} \cdot [|\varphi_M| - \frac{1}{2} \cdot
                    (sin(2\varphi_2) - sin(2\varphi_1))]

            I_{\bar{y}} = I_{y'} - A \cdot (z_c - z_0)^2


        where

        - :math:`r` is the radius of circular sector
        - :math:`\varphi_M` is the angular span of the sector in radians
        - :math:`\varphi_1` is the starting angle
        - :math:`\varphi_2` is the ending angle
        - :math:`z_c` is the z-coordinate of the centroid of
          the circular sector
        - :math:`z_0` is the z-coordinate of the center of the circular sector
        """
        iy_center = (
            self.radius ** 4 * (
                abs(self.angle)
                - 0.5 * (
                    np.sin(2 * (self.angle + self.start_angle))
                    - np.sin(2 * self.start_angle)
                )
            ) / 8
            )
        iy = (
             iy_center -
             self.mom_of_int_steiner(self.center_of_mass_z - self.center[1])
             )

        return iy

    @property
    def mom_of_int_z(self) -> float:
        r"""
        Computes the second moment of area :math:`I_{\bar{z}}` about the
        \bar{z}-axis, referenced to the centroid coordinate system.

        Returns
        -------
        np.float64
            The second moment of area :math:`I_{\bar{z}}`, in units of length⁴.

        Notes
        -----
        The components are initially computed relative to the center of the
        circular sector :math:`(y_0, z_0)` and then shifted to the centroid
        using the parallel axis theorem:

        .. math::
            I_{z'} = \frac{r^4}{8} \cdot [|\varphi_M| + \frac{1}{2} \cdot
                    (sin(2\varphi_2) - sin(2\varphi_1))]

            I_{\bar{z}} = I_{z'} - A \cdot (y_c - y_0)^2


        where

        - :math:`r` is the radius of circular sector
        - :math:`\varphi_M` is the angular span of the sector in radians
        - :math:`\varphi_1` is the starting angle
        - :math:`\varphi_2` is the ending angle
        - :math:`y_c` is the z-coordinate of the centroid of the
          circular sector
        - :math:`y_0` is the z-coordinate of the center of the circular sector
        """
        iz_center = (
            self.radius ** 4 * (
                abs(self.angle)
                + 0.5 * (
                    np.sin(2 * (self.angle + self.start_angle))
                    - np.sin(2 * self.start_angle)
                )
            ) / 8
        )
        iz = (
             iz_center -
             self.mom_of_int_steiner(self.center_of_mass_y - self.center[0])
             )

        return iz

    @property
    def width(self) -> float:
        r"""
        Computes the width of the circular sector in the local y-direction.

        Returns
        -------
        np.float64
            Width of the circular sector (i.e., the range of y-coordinates).

        Notes
        -----
        The width is defined as:

        .. math::

            w = y_{\text{max}} - y_{\text{min}}

        where :math:`y_{\text{max}}` and :math:`y_{\text{min}}` are the maximum
        and minimum horizontal coordinates of the sectors's boundary
        """
        y_boundary = self.boundary()[0]

        return y_boundary[1] - y_boundary[0]

    @property
    def height(self) -> float:
        r"""
        Computes the height of the circular sector in the local z-direction.

        Returns
        -------
        np.float64
            Height of the circular sector (i.e., the range of z-coordinates).

        Notes
        -----
        The height is defined as:

        .. math::

            h = z_{\text{max}} - z_{\text{min}}

        where :math:`z_{\text{max}}` and :math:`z_{\text{min}}` are the maximum
        and minimum vertical coordinates of the sectors's boundary
        """
        z_boundary = self.boundary()[1]

        return z_boundary[1] - z_boundary[0]

    def boundary(self):
        r"""
        Computes the minimum and maximum boundaries of the circular
        sector in both the y- and z-directions.

        Returns
        -------
        tuple of list of float or np.float64
            A tuple containing two lists:
            - The first list is [y_min, y_max], representing the
              horizontal extent.
            - The second list is [z_min, z_max], representing the
              vertical extent.

        Notes
        -----
        This method determines the spatial boundaries of the circular sector by
        evaluating the coordinates of:

        - The two arc endpoints (start and end angles),
        - The circle center,
        - And optionally the extreme values (i.e., if the local
          :math:`y'`- or :math:`z'`-axis lies between the start and
          end points of the circle sector).

        The method checks whether those extreme values within the angular range
        of the circular sector using `_angle_in_sector`, and includes them if
        applicable. This ensures the correct bounding box even for partial
        sectors that span multiple quadrants.

        These boundary values are used to compute the overall width and
        height of the sector geometry.

        Examples
        --------
        >>> from sstatics.core.preprocessing.geometry.objects import
        >>>     (CircularSector)
        >>> import numpy as np
        >>> c = CircularSector(center = (0,0), radius = 1, angle = np.pi/2,
        >>>                    start_angle = np.pi/4, positive = True
        >>>                    )
        >>> c.boundary()
        ([-0.707, 0.707], [0.0, 1.0])
        """
        # Start and end point y-coordinates
        y = [
            self.center[0] + self.radius * np.cos(self.start_angle),
            self.center[0] + self.radius * np.cos(self.start_angle +
                                                  self.angle),
            self.center[0]
        ]
        # Check if y-extrema (cos=±1) are within the angular range
        for extreme_angle in [0, np.pi]:
            if self._angle_in_sector(extreme_angle):
                y.append(self.center[0] + self.radius * np.cos(extreme_angle))

        # Start and end point z-coordinates
        z = [
            self.center[1] + self.radius * np.sin(self.start_angle),
            self.center[1] + self.radius * np.sin(self.start_angle +
                                                  self.angle),
            self.center[1]
        ]
        # Check if z-extrema (sin = ±1) are within the angular range
        for extreme_angle in [0.5 * np.pi, 1.5 * np.pi]:
            if self._angle_in_sector(extreme_angle):
                z.append(self.center[1] + self.radius * np.sin(extreme_angle))

        return ([min(y), max(y)],
                [min(z), max(z)])

    def _angle_in_sector(self, angle: float) -> bool:
        r"""
        Checks whether a given angle lies within the angular span of the
        circular sector.

        Parameters
        ----------
        angle : float
            The angle (in radians) to test. It will be normalized to the
            interval :math:`[0, 2\pi)`.

        Returns
        -------
        bool
            True if the angle lies within the circular sector's angular range,
            False otherwise.

        Notes
        -----
        This method accounts for circular wrapping around :math:`2\pi`
        and supports both positive (clockwise) and negative
        (counterclockwise) angular spans.

        - The start and end angles of the sector are normalized to the range
          :math:`[0, 2\pi)`.
        - The method distinguishes between two cases:
            1. The angular range does **not** wrap around :math:`2\pi`.
            2. The angular range **does** wrap around, requiring logical
               disjunction (OR) for checking inclusion.
        - For negative (counterclockwise) sectors, the logic is inverted
          accordingly.

        This method is used in `boundary()` to identify whether special points
        (e.g., where `sin` or `cos` reach extrema, i.e., where the
        :math:`y'`- or :math:`z'`- axis is intersected) fall within
        the sector.

        Examples
        --------
        The positive :math:`z'`-axis lies in the area of the circular
        sector.
        >>> from sstatics.core.preprocessing.geometry.objects import
        >>>     CircularSector
        >>> import numpy as np
        >>> c = CircularSector(center = (0,0), radius = 1, angle = np.pi/2,
        >>>                    start_angle = np.pi/4, positive = True
        >>>                    )
        >>> c._angle_in_sector(0)
        False
        >>> c._angle_in_sector(np.pi/2)
        True
        >>> c._angle_in_sector(np.pi)
        False
        """
        start = self.start_angle % (2 * np.pi)
        end = (start + self.angle) % (2 * np.pi)
        angle = angle % (2 * np.pi)
        if self.angle > 0:
            if start < end:
                return start <= angle <= end
            else:
                return angle >= start or angle <= end
        else:
            if end < start:
                return end <= angle <= start
            else:
                return angle >= end or angle <= start
