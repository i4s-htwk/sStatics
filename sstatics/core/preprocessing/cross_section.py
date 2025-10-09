import numpy as np

from dataclasses import dataclass
from typing import List, Optional, Union


from sstatics.core.preprocessing.geometry.objects import (
    CircularSector, Polygon
)
from sstatics.core.preprocessing.geometry.operation import (
    PolygonMerge, SectorToPolygonHandler
)


@dataclass(eq=False)
class CrossSection:

    def __init__(
        self,
        mom_of_int: Optional[float] = None,
        area: Optional[float] = None,
        height: Optional[float] = None,
        width: Optional[float] = None,
        shear_cor: Optional[float] = None,
        geometry: Optional[List[Union[Polygon, CircularSector]]] = None
    ):
        r"""Initializes a CrossSection either by geometry or mechanical
        properties.

        Parameters
        ----------
        mom_of_int : Optional[float], optional
            Moment of inertia.
        area : Optional[float], optional
            Cross-sectional area.
        height : Optional[float], optional
            Cross-sectional height.
        width : Optional[float], optional
            Cross-sectional width.
        shear_cor : Optional[float], optional
            Shear correction factor (default 1.0).
        geometry : Optional[List[Union[Polygon, CircularSector]]], optional
            List of geometric shapes defining the cross-section.

        Raises
        ------
        ValueError
            If both or neither geometry and mechanical properties are provided,
            or if mechanical properties are invalid.

        Examples
        --------
        Initialize via geometry:

        >>> cs = CrossSection(geometry=[Polygon([...]), CircularSector(...)])

        Initialize via mechanical properties:

        >>> cs = CrossSection(mom_of_int=1.5, area=2.0, height=0.3,
        >>>                     width=0.1,  shear_cor=0.9)
        """
        self.mechanics_given = all(x is not None for x in (mom_of_int,
                                                           area, height,
                                                           width, shear_cor))
        self.geometry_given = bool(geometry)

        if self.geometry_given and self.mechanics_given:
            raise ValueError(
                "Either define geometry OR mechanical properties, not both.")
        if not self.geometry_given and not self.mechanics_given:
            raise ValueError(
                "Either geometry or mechanical properties must be given.")

        self._mom_of_int = mom_of_int
        self._area = area
        self._height = height
        self._width = width
        # TODO: is this default value correct?
        self._shear_cor = shear_cor if shear_cor is not None else 1.0

        if self.mechanics_given:
            if self._mom_of_int <= 0:
                raise ValueError("mom_of_int must be > 0.")
            if self._area <= 0:
                raise ValueError("area must be > 0.")
            if self._height <= 0:
                raise ValueError("height must be > 0.")
            if self._width <= 0:
                raise ValueError("width must be > 0.")
            if self._shear_cor <= 0:
                raise ValueError("shear_cor must be > 0.")
            if self._area > self._height * self._width:
                raise ValueError(f"area must be ≤ height × width = "
                                 f"{self._height * self._width}")
            self.polygon = None
            self.circular_sector = None
        else:
            self.geometry = geometry
            self._process_geometry()

    def _process_geometry(self):
        """
        Process input geometry: separate polygons and circular sectors,
        convert circular sectors that overlap with polygons and merge
        polygons into a unified shape.

        Raises
        ------
            ValueError
                If a single negative polygon is used as cross-section.

        Notes
        -----
            Circular sectors that overlap with polygons are converted
            internally to polygons for computation purposes.
        """
        poly = [e for e in self.geometry if isinstance(e, Polygon)]
        circ = [e for e in self.geometry if isinstance(e, CircularSector)]

        if circ:
            poly, circ = SectorToPolygonHandler(poly, circ)()

        self.circular_sector = circ

        if len(poly) == 1 and not self.circular_sector:
            if not poly[0].positive:
                raise ValueError(
                    "A single negative polygon cannot be used as a "
                    "cross-section.")
            self.polygon = poly[0]
        else:
            positive = [p for p in poly if p.positive]
            negative = [p for p in poly if not p.positive]
            self.polygon = PolygonMerge(positive, negative)()

    @property
    def mom_of_int(self) -> float:
        """
        Returns the moment of inertia of the cross-section.

        Returns
        -------
            float
                Moment of inertia (Iyy), either from input or calculated.

        Examples
        --------
        >>> cs = CrossSection(geometry=[Polygon(points=[(0, 0), (2, 0), (2,
        1), (0, 1), (0, 0)])])
        >>> I = cs.mom_of_int
        0.16666666666666663
        """
        if self._mom_of_int is not None:
            return self._mom_of_int
        return self._calc_mom_of_int()

    @property
    def area(self) -> float:
        """
        Returns the cross-sectional area.

        Returns
        -------
            float
                Cross-sectional area.

        Examples
        --------
        >>> cs = CrossSection(geometry=[Polygon(points=[(0, 0), (2, 0), (2,
        1),  (0, 1), (0, 0)])])
        >>> A = cs.area
        2.0
        """
        if self._area is not None:
            return self._area
        return self._calc_area()

    @property
    def height(self) -> float:
        """
        Returns the height of the cross-section.

        Returns
        -------
            float
                Cross-section height.

        Notes
        -----
            Calculated from polygon geometry; circular sectors are ignored.

        Examples
        --------
        >>> cs = CrossSection(geometry=[Polygon(points=[(0, 0), (2, 0), (2,
        1),  (0, 1), (0, 0)])])
        >>> h = cs.height
        1.0
        """
        if self._height is not None:
            return self._height
        return self._calc_height()

    @property
    def width(self) -> float:
        """
        Returns the width of the cross-section.

        Returns
        -------
            float
                Cross-section width.

        Notes
        -----
            Calculated from polygon geometry; circular sectors are ignored.

        Examples
        --------
        >>> cs = CrossSection(geometry=[Polygon(points=[(0, 0), (2, 0), (2,
        1),  (0, 1), (0, 0)])])
        >>> w = cs.width
        2.0
        """
        if self._width is not None:
            return self._width
        return self._calc_width()

    @property
    def shear_cor(self) -> float:
        """
        Returns the shear correction factor.

        Returns
        -------
            float
                Shear correction factor (default 1.0).

        Notes
        -----
            Currently a fixed value; automatic calculation is not implemented.

        Examples
        --------
        >>> cs = CrossSection(mom_of_int=..., area=..., height=...,
        width=..., shear_cor=0.85)
        >>> sc = cs.shear_cor
        float(0.85)
        """
        # TODO:
        return self._shear_cor

    @property
    def static_moment(self) -> tuple:
        """
        Returns the static moments (first moments of area) about z and y axes.

        Returns
        -------
            tuple of floats
                (S_z, S_y) static moments.

        Notes
        -----
            Includes contributions from polygons and circular sectors.

        Examples
        --------
        >>> cs = CrossSection(geometry=[Polygon(points=[(0, 0), (2, 0), (2,
        1),  (0, 1), (0, 0)])])
        >>> sz, sy = cs.static_moment
        (np.float64(1.0), np.float64(2.0))
        """
        if self.polygon is None:
            return (self._width * self._height ** 2 / 2,
                    self._height * self._width ** 2 / 2)
        else:
            sm = self.polygon.static_moment
            if self.circular_sector:
                for c in self.circular_sector:
                    sm = tuple(s + c_s for s, c_s in zip(sm, c.static_moment))
            return sm

    @property
    def center_of_mass_y(self) -> float:
        """
        Returns the y-coordinate of the centroid of the cross-section.

        Returns
        -------
        float
            Centroid y-coordinate.

        Examples
        --------
        >>> cs = CrossSection(geometry=[Polygon(points=[(0, 0), (2, 0), (2,
        1),  (0, 1), (0, 0)])])
        >>> cy = cs.center_of_mass_y
        1
        """
        return self.static_moment[1] / self.area

    @property
    def center_of_mass_z(self) -> float:
        """
        Returns the z-coordinate of the centroid of the cross-section.

        Returns
        -------
        float
            Centroid z-coordinate.

        Examples
        --------
        >>> cs = CrossSection(geometry=[Polygon(points=[(0, 0), (2, 0), (2,
        1),  (0, 1), (0, 0)])])
        >>> cz = cs.center_of_mass_z
        0.5
        """
        return self.static_moment[0] / self.area

    def _calc_area(self) -> float:
        """
        Calculates the total cross-sectional area from geometry.

        Returns
        -------
            float
                Cross-sectional area.
        """
        area = self.polygon.area
        if self.circular_sector:
            for c in self.circular_sector:
                area += c.area if c.positive else -c.area
        return area

    def _mom_of_int_polygon(self) -> float:
        r"""
        Calculates moment of inertia of polygonal parts including centroid
        offset correction.

        Returns
        -------
            float
                Polygon moment of inertia.
        """
        dy = self.center_of_mass_z - self.polygon.center_of_mass_z
        return self.polygon.iyy + self.polygon.mom_of_int_steiner(dy)

    def _mom_of_int_circular_sectors(self) -> float:
        """
        Calculates moment of inertia of circular sectors including centroid
        offset correction.

        Returns
        -------
            float
                Circular sectors moment of inertia.
        """
        if self.circular_sector:
            return sum(
                c.mom_of_int_y + c.mom_of_int_steiner(
                    self.center_of_mass_z - c.center_of_mass_z)
                for c in self.circular_sector
            )
        return 0

    def _calc_mom_of_int(self) -> float:
        """
        Calculates total moment of inertia from all geometry components.

        Returns
        -------
            float
                Total moment of inertia.
        """
        return self._mom_of_int_polygon() + self._mom_of_int_circular_sectors()

    def _calc_height(self) -> float:
        r"""
        Calculates height of the cross-section from polygon geometry.

        Returns
        -------
            float
                Height.

        Notes
        -----
        Circular sectors currently not included.
        """
        return self.polygon.height

    def _calc_width(self) -> float:
        r"""
        Calculates width of the cross-section from polygon geometry.

        Returns
        -------
            float
                Width.

        Notes
        -----
            Circular sectors currently not included.
        """
        return self.polygon.width

    @property
    def z_min(self):
        r"""
        Calculates the smallest z-value of the cross-section.

        Returns
        -------

        Notes
        -----

        """

        if self.geometry_given:
            z_coords = []
            for shape in self.geometry:
                if isinstance(shape, CircularSector):
                    shape = shape.convert_to_polygon()

                z_coords.extend(shape.z)
            return min(z_coords)

        else:
            return (0)

    @property
    def y_min(self):
        r"""
        Calculates the smallest y-value of the cross-section.
        """
        if self.geometry_given:
            y_coords = []
            for shape in self.geometry:
                if isinstance(shape, CircularSector):
                    shape = shape.convert_to_polygon()
                y_coords.extend(shape.y)
            return min(y_coords)
        else:
            return (0)

    def height_disc(self, disc):
        r"""
        Discetize the height of the cross-section.
        Creates evenly spaced height coordinates across the cross-section
        height.

        Parameters
        ----------
        disc : int
            Number of subdivisions along the cross-section height.

        Returns
        -------
        numpy.ndarray
        ( n ) array with linearly spaced heights from 0 to the cross-section
        height `self.height`, inclusive.
        Where n is n_disc + 1.
        """
        n_disc = disc
        h = self.height
        return np.linspace(0, h, n_disc + 1)

    @property
    def rectangle_check(self):
        '''
        Checks if the given cross-section is rectangular with edges parallel
        to the coordinate axes.

        Returns True if:
            - the cross-section is defined by mechanical properties (always
            rectangular),
        or if:
            - it consists of exactly one Polygon,
            - the Polygon has exactly 4 distinct corner points (excluding the
            closing point),
            - the corner points form a rectangle in correct edge order
            (condition A or B).
        Returns False otherwise.
        '''

        if self.geometry_given:
            if len(self.geometry) != 1:
                # a rectangle should be given as 1 geometry
                return False

            geom = self.geometry[0]

            if not isinstance(geom, Polygon):
                # a rectangle cannot contain a circular sector
                return False

            pts = list(geom.points)

            corner_pts = pts[:-1]   # deletes the last entry of the pts-list
            # which is the same as the first one

            if len(corner_pts) != 4:
                # corner_pts-list must be exactly 4 points long
                return False

            if len(set(corner_pts)) != 4:
                # checks, if no points appear twice by converting into a
                # set and checking if it is still 4 points long
                return False

            z = [p[0] for p in corner_pts]
            y = [p[1] for p in corner_pts]

            condition_a = (
                    (z[0] == z[1]) and
                    (z[2] == z[3]) and
                    (y[0] == y[3]) and
                    (y[1] == y[2])
            )

            condition_b = (
                    (z[0] == z[3]) and
                    (z[1] == z[2]) and
                    (y[0] == y[1]) and
                    (y[2] == y[3])
            )

            if not (condition_a or condition_b):
                return False

            return True

        else:
            # self.mechanics_given
            return True
