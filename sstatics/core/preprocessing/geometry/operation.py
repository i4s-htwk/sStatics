
from itertools import combinations

from typing import List, Tuple

from shapely.geometry import MultiPolygon, Polygon as ShapelyPolygon
from shapely.ops import unary_union
from shapely.set_operations import difference

from sstatics.core.logger_mixin import LoggerMixin
from sstatics.core.preprocessing.geometry.objects import (
    CircularSector, Polygon
)


class PolygonMerge(LoggerMixin):
    r"""
    Represents a combination of multiple positive and negative polygons,
    and provides methods to compute their geometric difference.

    This class is used to merge a list of "positive" polygons and subtract
    one or more "negative" polygons (holes or cut-outs) from the result.
    Internally, it uses geometric union and difference operations based
    on Shapely.

    Parameters
    ----------
    positive : list of :any:`Polygon`, optional
        A list of polygons that define the base area to be merged. At least
        one polygon must be provided. All elements must be instances of
        the custom `Polygon` class.
    negative : list of :any:`Polygon`, optional
        A list of polygons to be subtracted (cut out) from the merged
        positive geometry. These are treated as holes or exclusions.
        Defaults to an empty list.

    Raises
    ------
    ValueError
        If the `positive` list is empty.
    TypeError
        If any element of `positive` or `negative` is not an instance
        of `Polygon`.

    Attributes
    ----------
    positive : list of  :any:`Polygon`
        The list of input polygons to be merged.
    negative : list of  :any:`Polygon`
        The list of polygons to subtract from the merged positive area.
    unary_pos : shapely.geometry.Polygon or MultiPolygon
        Union geometry of all positive polygons.
    unary_neg : shapely.geometry.Polygon or MultiPolygon or None
        Union geometry of all negative polygons, or None if empty.
    """

    # noinspection PyMissingConstructor
    def __init__(self,
                 positive: List[Polygon] = None,
                 negative: List[Polygon] = None,
                 debug: bool = False):
        # allow LoggerMixin to initialize first (wrapped automatically)
        _ = debug  # PyCharm quiet

        self.logger.debug("Initializing PolygonMerge...")

        self.positive = positive or []
        self.negative = negative or []

        self.logger.debug(
            "Input contains %d positive and %d negative polygons.",
            len(self.positive), len(self.negative)
        )

        # ---- Validation -----------------------------------------------------
        if not all(isinstance(p, Polygon) for p in self.positive):
            self.logger.error(
                "Invalid type detected in 'positive' list. Elements: %s",
                [type(p).__name__ for p in self.positive]
            )
            raise TypeError(
                "All elements in 'positive' must be Polygon instances."
            )

        if not all(isinstance(p, Polygon) for p in self.negative):
            self.logger.error(
                "Invalid type detected in 'negative' list. Elements: %s",
                [type(p).__name__ for p in self.negative]
            )
            raise TypeError(
                "All elements in 'negative' must be Polygon instances."
            )

        # ---- Compute unary unions -------------------------------------------
        if self.positive:
            self.logger.debug("Computing unary union of positive polygons...")
            self.unary_pos = unary_union([p.polygon for p in self.positive])
            self.logger.debug(
                "Unary positive union created with geometry type: %s",
                self.unary_pos.geom_type
            )
        else:
            self.unary_pos = None
            self.logger.debug("No positive polygons provided.")

        if self.negative:
            self.logger.debug("Computing unary union of negative polygons...")
            self.unary_neg = unary_union([p.polygon for p in self.negative])
            self.logger.debug(
                "Unary negative union created with geometry type: %s",
                self.unary_neg.geom_type
            )
        else:
            self.unary_neg = None
            self.logger.debug("No negative polygons provided.")

        self.logger.debug("PolygonMerge initialized successfully.")

    def difference(self) -> Polygon:
        r"""
        Computes the geometric difference between the union of the
        positive polygons and the union of the negative polygons.

        Returns
        -------
         :any:`Polygon`
            A new `Polygon` instance representing the result of the
            difference operation.

        Raises
        ------
        NotImplementedError
            If the resulting geometry is a `MultiPolygon`, which is
            currently not supported.
        """
        self.logger.debug("Executing difference computation...")

        positive_flag = True

        # Case 1: positive exists
        if self.unary_pos:
            result = self.unary_pos
            self.logger.debug(
                "Starting with positive union (%s).",
                self.unary_pos.geom_type
            )

            if self.unary_neg:
                self.logger.debug(
                    "Negative union present → applying difference operation..."
                )
                result = difference(result, self.unary_neg)
            else:
                self.logger.debug("No negative union → skipping subtraction.")

        # Case 2: only negative exists
        elif self.unary_neg:
            result = self.unary_neg
            positive_flag = False
            self.logger.debug(
                "No positive union. Using negative union directly."
            )

        else:
            self.logger.error("Neither positive nor negative polygons exist.")
            raise ValueError("Cannot compute difference without any input.")

        # Multipolygon check --------------------------------------------------
        if isinstance(result, MultiPolygon):
            self.logger.error(
                "Difference resulted in unsupported MultiPolygon. "
                "Number of sub-geometries: %d",
                len(result.geoms)
            )
            raise NotImplementedError(
                "Difference resulted in a MultiPolygon, "
                "which is not yet handled."
            )

        self.logger.debug(
            "Difference successful. Result polygon has %d exterior points "
            "and %d holes.",
            len(result.exterior.coords),
            len(result.interiors)
        )

        return Polygon(
            points=list(result.exterior.coords),
            holes=[list(interior.coords) for interior in result.interiors],
            positive=positive_flag,
        )

    def __call__(self) -> Polygon:
        r"""
        Allows the object to be called like a function to compute
        the difference.

        Returns
        -------
        Polygon
            The resulting `Polygon` after applying the difference
            between positive and negative areas.
        """
        self.logger.debug("__call__ invoked → computing difference()")
        return self.difference()


class SectorToPolygonHandler(LoggerMixin):
    r"""
    Handles the interaction between regular polygons and circular sectors.

    This class is responsible for checking overlaps between circular sectors
    and existing polygons. If an overlap is detected, the circular sector is
    converted into a polygon and added to the list of polygons. Otherwise,
    it is kept as a remaining circular sector.

    Parameters
    ----------
    polygons : list of Polygon
        A list of existing polygonal shapes. Each element must be an instance
        of the custom `Polygon` class.
    circular : list of CircularSector
        A list of circular sectors to be tested against the polygons.
        Each element must be an instance of `CircularSector`.

    Raises
    ------
    TypeError
        If any element in `polygons` is not a `Polygon`, or any element in
        `circular` is not a `CircularSector`.

    Attributes
    ----------
    polygons : list of Polygon
        The original list of polygon shapes.
    circular_sector : list of CircularSector
        The original list of circular sectors to be tested and potentially
        converted.
    """

    # noinspection PyMissingConstructor
    def __init__(self,
                 polygons: List[Polygon],
                 circular: List[CircularSector],
                 debug: bool = False):
        # allow LoggerMixin to initialize first (wrapped automatically)
        _ = debug  # PyCharm quiet
        self.logger.debug("Initializing SectorToPolygonHandler...")

        self.polygons = polygons
        self.circular_sector = circular

        self.logger.debug(
            "Received %d polygons and %d circular sectors.",
            len(self.polygons), len(self.circular_sector)
        )

        # ---------------- VALIDATION ----------------
        if not all(isinstance(p, Polygon) for p in self.polygons):
            types = [type(p).__name__ for p in self.polygons]
            self.logger.error(
                "Invalid element in 'polygons': Types = %s", types
            )
            raise TypeError(
                "All elements in 'polygons' must be instances of Polygon."
            )

        if not all(
                isinstance(cs, CircularSector) for cs in self.circular_sector):
            types = [type(cs).__name__ for cs in self.circular_sector]
            self.logger.error(
                "Invalid element in 'circular': Types = %s", types
            )
            raise TypeError(
                "All elements in 'circular' must be CircularSector instances."
            )

        self.logger.debug("SectorToPolygonHandler successfully initialized.")

    @staticmethod
    def bounding_boxes_overlap(a: ShapelyPolygon, b: ShapelyPolygon) -> bool:
        r"""
        Checks whether the bounding boxes of two Shapely polygons overlap.

        Parameters
        ----------
        a : shapely.geometry.Polygon
            First polygon.
        b : shapely.geometry.Polygon
            Second polygon.

        Returns
        -------
        bool
            True if the bounding boxes overlap, False otherwise.
        """
        a_minx, a_miny, a_maxx, a_maxy = a.bounds
        b_minx, b_miny, b_maxx, b_maxy = b.bounds
        return (
            a_minx <= b_maxx and
            a_maxx >= b_minx and
            a_miny <= b_maxy and
            a_maxy >= b_miny
        )

    @staticmethod
    def bounding_boxes_overlap_circ(a: CircularSector, b: CircularSector) \
            -> bool:
        [a_miny, a_maxy], [a_minz, a_maxz] = a.boundary()
        [b_miny, b_maxy], [b_minz, b_maxz] = b.boundary()
        return (
                a_miny <= b_maxy and
                a_maxy >= b_miny and
                a_minz <= b_maxz and
                a_maxz >= b_minz
        )

    @staticmethod
    def identical(a: CircularSector, b: CircularSector) -> bool:
        [a_miny, a_maxy], [a_minz, a_maxz] = a.boundary()
        [b_miny, b_maxy], [b_minz, b_maxz] = b.boundary()
        return (
                a_miny == b_miny and
                a_maxy == b_maxy and
                a_minz == b_minz and
                a_maxz == b_maxz and
                a.center == b.center
        )

    def execute(self) -> Tuple[List[Polygon], List[CircularSector]]:
        r"""
        Converts intersecting circular sectors to polygons and separates
        non-intersecting ones.

        Each circular sector is checked for intersection with the existing
        polygons. If it intersects any polygon (based on bounding box and
        area of geometric intersection), it is converted to a polygon and
        added to the polygon list. Otherwise, it remains as a circular sector.

        Returns
        -------
        tuple of (list of Polygon, list of CircularSector)
            - A list of polygons including the original ones and those
              converted from intersecting sectors.
            - A list of circular sectors that did not intersect any polygon.
        """
        self.logger.debug("Starting sector-to-polygon processing...")

        # simple pass-through
        if len(self.polygons) == 0 and len(self.circular_sector) == 1:
            self.logger.debug(
                "Early exit: 0 polygons and 1 circular sector "
                "(no interactions)."
            )
            return self.polygons, self.circular_sector

        polygons_extended = self.polygons.copy()
        circular_sector_remaining = []

        # Convert all circular sectors to polygons first if more than one
        # exists
        if len(self.circular_sector) > 1:
            self.logger.debug(
                "Preparing %d sectors for pre-checking via conversion...",
                len(self.circular_sector)
            )

            circ_to_poly = [i.convert_to_polygon() for i in
                            self.circular_sector]
            circ_to_poly_shapely = [i.polygon for i in circ_to_poly]

            combined, contained = \
                self.find_duplicates_overlaps_contained(circ_to_poly_shapely)

            if combined:
                self.logger.info(
                    "%d circular sectors flagged to be converted to "
                    "polygons: %s",
                    len(combined), combined
                )

            # Add combined sectors as polygons
            for i in combined:
                self.logger.debug(
                    "Sector %d overlaps/duplicates → converting to polygon.",
                    i
                )
                polygons_extended.append(circ_to_poly[i])

            # Remove processed ones
            self.circular_sector = [
                elem for i, elem in enumerate(self.circular_sector)
                if i not in combined
            ]

        # Check remaining circular sectors individually
        for i, sector in enumerate(self.circular_sector):
            sec_poly = sector.convert_to_polygon(num_points=100)

            intersects = any(
                self.bounding_boxes_overlap(sec_poly.polygon, sp.polygon)
                and sec_poly.polygon.overlaps(sp.polygon)
                for sp in polygons_extended
            )

            if intersects:
                self.logger.info(
                    "Circular sector %d intersects an existing polygon → "
                    "converted to polygon (100-point discretization).",
                    i
                )
                polygons_extended.append(sec_poly)
            else:
                self.logger.debug(
                    "Circular sector %d does not intersect any polygon.",
                    i
                )
                circular_sector_remaining.append(sector)

        self.logger.debug(
            "Processing complete: %d polygons, %d remaining sectors.",
            len(polygons_extended),
            len(circular_sector_remaining)
        )

        return polygons_extended, circular_sector_remaining

    def __call__(self) -> Tuple[List[Polygon], List[CircularSector]]:
        r"""
        Enables the object to be called directly, executing the overlap
        check and conversion process.

        Returns
        -------
        tuple of (list of Polygon, list of CircularSector)
            See `execute` method for details.
        """
        return self.execute()

    def find_duplicates_overlaps_contained(self, geometries):
        self.logger.debug(
            "Checking %d geometries for duplicates/overlaps/containment.",
            len(geometries))

        if len(geometries) == 1:
            self.logger.debug("Only one geometry → no duplicates.")
            return [], [0]

        duplicates = set()
        overlaps = set()
        contained = set()

        for i, j in combinations(range(len(geometries)), 2):
            g1, g2 = geometries[i], geometries[j]
            self.logger.debug("Comparing geometry %d with %d...", i, j)

            if g1.equals(g2):
                duplicates.add(i)
                duplicates.add(j)
                self.logger.debug("Geometries %d and %d are identical.", i, j)

            elif g1.contains(g2):
                contained.add(j)
                self.logger.debug("Geometry %d contains geometry %d.", i, j)

            elif g2.contains(g1):
                contained.add(i)
                self.logger.debug("Geometry %d contains geometry %d.", j, i)

            elif g1.intersects(g2) and g1.intersection(g2).area > 0:
                overlaps.add(i)
                overlaps.add(j)
                self.logger.debug(
                    "Geometries %d and %d partially overlap (area > 0).", i, j
                )

        overlaps -= duplicates
        overlaps -= contained

        combined = list(duplicates | overlaps)

        self.logger.debug("Duplicate indices: %s", list(duplicates))
        self.logger.debug("Overlap indices:   %s", list(overlaps))
        self.logger.debug("Contained indices: %s", list(contained))
        self.logger.debug("Combined result:   %s", combined)

        return combined, contained
