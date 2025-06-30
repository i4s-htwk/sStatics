
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

    >>> from sstatics.core.geometry.objects import Polygon
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
        >>> from sstatics.core.geometry.objects import Polygon

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
        >>> from sstatics.core.geometry.objects import Polygon
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

        >>> from sstatics.core.geometry.objects import Polygon
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
    center: Tuple[float, float]
    radius: float
    angle: float = 0
    start_angle: float = 0
    positive: bool = True

    def __post_init__(self):
        if not isinstance(self.center, tuple) or len(self.center) != 2:
            raise ValueError(
                "Center must be a tuple of two numeric values (x, y)."
            )
        if not all(isinstance(coord, (int, float)) for coord in self.center):
            raise ValueError(
                "Center coordinates must be numeric (float or int)."
            )
        if not isinstance(self.radius, (int, float)) or self.radius <= 0:
            raise ValueError("Radius must be a positive number.")
        if not isinstance(self.angle, (int, float)) or self.angle == 0:
            raise ValueError("Angle must be a non-zero number (in radians).")
        if not isinstance(self.positive, bool):
            raise ValueError(
                "The 'positive' attribute must be a boolean value."
            )
        if not isinstance(self.start_angle, (int, float)):
            raise ValueError("Start angle must be a number (in radians).")

    def convert_to_polygon(self, num_points=100):
        cx, cy = self.center
        points = [(cx, cy)]
        for i in range(num_points + 1):
            theta = self.start_angle + i * self.angle / num_points
            x = cx + self.radius * np.cos(theta)
            y = cy + self.radius * np.sin(theta)
            points.append((x, y))
        points.append((cx, cy))
        return Polygon(points, positive=self.positive)

    @property
    def area(self):
        # TODO: calc area of a CircularSector
        return 0.5 * self.radius ** 2 * abs(self.angle)

    @property
    def static_moment(self):
        # TODO: calc static moment of a CircularSector
        return [0, 0]

    @property
    def center_of_mass_y(self):
        return self.static_moment[1] / self.area

    @property
    def center_of_mass_z(self):
        return self.static_moment[0] / self.area

    def mom_of_int_steiner(self, center):
        return self.area * center ** 2

    @property
    def mom_of_int_z(self) -> float:
        # TODO: calc mon_of_int_z of a CircularSector
        return 0

    @property
    def mom_of_int_y(self) -> float:
        # TODO: calc mon_of_int_y of a CircularSector
        return 0
