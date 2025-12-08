
from dataclasses import dataclass
from typing import List, Optional, Union

from sstatics.core.preprocessing.geometry.objects import (
    CircularSector, Polygon
)
from sstatics.core.preprocessing.geometry.operation import (
    PolygonMerge, SectorToPolygonHandler
)
from sstatics.core.logger_mixin import LoggerMixin


@dataclass(eq=False)
class CrossSection(LoggerMixin):

    # noinspection PyMissingConstructor
    def __init__(
        self,
        mom_of_int: Optional[float] = None,
        area: Optional[float] = None,
        height: Optional[float] = None,
        width: Optional[float] = None,
        shear_cor: Optional[float] = None,
        geometry: Optional[List[Union[Polygon, CircularSector]]] = None,
        debug: bool = False
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
        self.debug = debug
        self.logger.debug("Starting CrossSection initialization.")
        mechanics_given = all(x is not None for x in
                              (mom_of_int, area, height, width, shear_cor))
        geometry_given = bool(geometry)

        self.logger.debug(
            f"Mechanics provided: {mechanics_given}, "
            f"Geometry provided: {geometry_given}"
        )

        # ---- Validation -----------------------------------------------------
        if geometry_given and mechanics_given:
            self.logger.error(
                "Both geometry and mechanical properties provided."
            )
            raise ValueError(
                "Either define geometry OR mechanical properties, not both."
            )

        if not geometry_given and not mechanics_given:
            self.logger.error(
                "Neither geometry nor mechanical properties provided."
            )
            raise ValueError(
                "Either geometry or mechanical properties must be given."
            )

        # ---- Mechanical properties path -------------------------------------
        self._mom_of_int = mom_of_int
        self._area = area
        self._height = height
        self._width = width
        self._shear_cor = shear_cor if shear_cor is not None else 1.0

        if mechanics_given:
            self.logger.debug(
                f"Validating mechanical properties: "
                f"I={self._mom_of_int}, A={self._area}, "
                f"h={self._height}, w={self._width}, k={self._shear_cor}"
            )

            if self._mom_of_int <= 0:
                self.logger.error("Invalid moment of inertia.")
                raise ValueError("mom_of_int must be > 0.")
            if self._area <= 0:
                self.logger.error("Invalid area.")
                raise ValueError("area must be > 0.")
            if self._height <= 0:
                self.logger.error("Invalid height.")
                raise ValueError("height must be > 0.")
            if self._width <= 0:
                self.logger.error("Invalid width.")
                raise ValueError("width must be > 0.")
            if self._shear_cor <= 0:
                self.logger.error("Invalid shear correction factor.")
                raise ValueError("shear_cor must be > 0.")
            if self._area > self._height * self._width:
                self.logger.error(
                    f"Area larger than height × width "
                    f"({self._height * self._width})."
                )
                raise ValueError(
                    f"area must be ≤ height × width = "
                    f"{self._height * self._width}"
                )

            self.logger.info(
                "CrossSection initialized using mechanical properties."
            )

            self.polygon = None
            self.circular_sector = None
        else:
            # ---- Geometry path ----------------------------------------------
            self.geometry = geometry
            self.logger.debug("Initializing CrossSection from geometry...")
            self._process_geometry()
            self.logger.info(
                "CrossSection successfully created from geometry.")

    def _process_geometry(self):
        """
        Process input geometry: separate polygons and circular sectors,
        convert circular sectors that overlap with polygons and merge
        polygons into a unified shape.

        Raises
        ------
        ValueError
            If a single negative circular sector is used as cross-section.
        ValueError
            If a single negative polygon is used as cross-section.

        Notes
        -----
        Circular sectors that overlap with polygons are converted
        internally to polygons for computation purposes.
        """
        self.logger.debug(
            "Processing geometry: separating polygons and circular sectors..."
        )
        poly = [e for e in self.geometry if isinstance(e, Polygon)]
        circ = [e for e in self.geometry if isinstance(e, CircularSector)]

        self.logger.debug(
            f"Found {len(poly)} polygons and {len(circ)} circular sectors."
        )

        # Convert circular sectors that overlap with polygons
        if circ:
            self.logger.debug("Checking circular sector overlaps...")
            poly, circ = SectorToPolygonHandler(
                poly, circ, debug=self.debug
            )()
            self.logger.debug(
                f"After SectorToPolygonHandler: {len(poly)} polygons, "
                f"{len(circ)} remaining circular sectors."
            )

        self.circular_sector = circ
        self.polygon = poly

        # ---- Validation -----------------------------------------------------
        if not self.polygon:
            c_positive = sum(1 for c in circ if c.positive)

            self.logger.debug(
                f"No polygon present. Positive circular sectors count: "
                f"{c_positive}"
            )

            if c_positive == 0:
                self.logger.error(
                    "No positive circular sector in geometry."
                )
                raise ValueError(
                    "At least one positive circular sector is required."
                )

            # Single polygon only
        elif len(poly) == 1 and not self.circular_sector:
            self.logger.debug("Single polygon geometry.")
            if not poly[0].positive:
                self.logger.error("Single polygon is negative.")
                raise ValueError(
                    "A single negative polygon cannot be used as a "
                    "cross-section."
                )

            self.logger.debug("Single positive polygon accepted.")
            self.polygon = poly[0]

            # Polygon merge case
        elif self.polygon:
            self.logger.debug(
                f"Merging polygons: {len(poly)} polygons found."
            )
            positive = [p for p in poly if p.positive]
            negative = [p for p in poly if not p.positive]
            self.logger.debug(
                f"{len(positive)} positive polygons, "
                f"{len(negative)} negative polygons."
            )

            self.polygon = PolygonMerge(
                positive, negative, debug=self.debug
            )()
            self.logger.debug("Polygon merge completed.")

        else:
            self.logger.error(
                "Unsupported combination of polygons and circular sectors."
            )
            raise ValueError(
                "This definition of geometry consisting of polygons and "
                "circular sectors was not taken into account."
            )

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
        Calculated from polygon and circular sector geometry.

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
        Calculated from polygon and circular sector geometry.

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
        # TODO:...
        return self._shear_cor

    @property
    def static_moment(self) -> tuple:
        """
        Returns the static moments (first moments of area) about y and z axes
        if a geometry is defined.

        Returns
        -------
        tuple of floats
            (S_y, S_z) static moments.

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
        if self.polygon or self.circular_sector:
            sm = [0, 0]
            if self.polygon:
                sm = tuple(value * (1 if self.polygon.positive else -1)
                           for value in self.polygon.static_moment)
            if self.circular_sector:
                for c in self.circular_sector:
                    factor = 1 if c.positive else -1
                    signed_moment = tuple(factor * x for x in c.static_moment)
                    sm = tuple(s + c_s for s, c_s in zip(sm, signed_moment))
            return sm
        else:
            raise ValueError("Either a polygon or a circular sector must be"
                             " defined.")

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
        area = 0.0
        if self.polygon:
            area += self.polygon.area * (
                1 if self.polygon.positive else -1)
        if self.circular_sector:
            for c in self.circular_sector:
                area += c.area * (1 if c.positive else -1)
        return area

    def _mom_of_int_polygon(self) -> float:
        """
        Calculates moment of inertia of polygonal parts including centroid
        offset correction.

        Returns
        -------
        float
            Polygon moment of inertia.
        """
        if self.polygon:
            dy = self.center_of_mass_z - self.polygon.center_of_mass_z
            sign = 1 if self.polygon.positive else -1
            return ((self.polygon.iyy + self.polygon.mom_of_int_steiner(dy))
                    * sign)
        return 0.0

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
            mom_total = 0.0
            for c in self.circular_sector:
                mom = (c.mom_of_int_y +
                       c.mom_of_int_steiner(
                           self.center_of_mass_z - c.center_of_mass_z
                       )
                       )
                mom_total += mom if c.positive else -mom
            return mom_total
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

    def boundary(self):
        """
        Determines the overall geometric boundaries of the cross-section in
        both the y- and z-directions.

        Returns
        -------
        tuple of list of float
            ([y_min, y_max], [z_min, z_max]) representing the horizontal and
            vertical extents of the cross-section.

        Notes
        -----
        - The boundaries are computed based on all geometric components
          (polygon and circular sectors) that make up the cross-section.
        - If a polygon is present, its vertex coordinates are included
          directly.
        - For each circular sector, the corresponding boundary extents are
          obtained from its ``boundary()`` method.
        - Only circular sectors marked as *positive* (i.e., material present)
          are considered for determining the outer limits.
        - If neither polygons nor circular sectors are defined, the method
          returns zero boundaries.
        """
        y_vals, z_vals = [], []

        if self.polygon:
            y_vals.extend(self.polygon.y)
            z_vals.extend(self.polygon.z)

        if self.circular_sector:
            for c in self.circular_sector:
                if c.positive:
                    yb, zb = c.boundary()
                    y_vals.extend(yb)
                    z_vals.extend(zb)

        if not y_vals or not z_vals:
            return ([0.0, 0.0],
                    [0.0, 0.0])

        return ([min(y_vals), max(y_vals)],
                [min(z_vals), max(z_vals)])

    def _calc_width(self) -> float:
        """
        Calculates the total width of the cross-section in the y-direction.

        Returns
        -------
        float
            The horizontal extent (width) of the entire cross-section.
        """
        yb, _ = self.boundary()
        return yb[1] - yb[0]

    def _calc_height(self) -> float:
        """
        Calculates the total height of the cross-section in the z-direction.

        Returns
        -------
        float
            The vertical extent (height) of the entire cross-section.
        """
        _, zb = self.boundary()
        return zb[1] - zb[0]
