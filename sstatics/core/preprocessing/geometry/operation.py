
from typing import List, Tuple

from shapely.geometry import MultiPolygon, Polygon as ShapelyPolygon
from shapely.ops import unary_union
from shapely.set_operations import difference

from sstatics.core.preprocessing.geometry.objects import (
    CircularSector, Polygon
)


class PolygonMerge:
    r"""
    Represents a combination of multiple positive and negative polygons,
    and provides methods to compute their geometric difference.

    This class is used to merge a list of "positive" polygons and subtract
    one or more "negative" polygons (holes or cut-outs) from the result.
    Internally, it uses geometric union and difference operations based
    on Shapely.

    Parameters
    ----------
    positive : list of Polygon, optional
        A list of polygons that define the base area to be merged. At least
        one polygon must be provided. All elements must be instances of
        the custom `Polygon` class.
    negative : list of Polygon, optional
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
    positive : list of Polygon
        The list of input polygons to be merged.
    negative : list of Polygon
        The list of polygons to subtract from the merged positive area.
    unary_pos : shapely.geometry.Polygon or MultiPolygon
        Union geometry of all positive polygons.
    unary_neg : shapely.geometry.Polygon or MultiPolygon or None
        Union geometry of all negative polygons, or None if empty.
    """

    def __init__(self,
                 positive: List[Polygon] = None,
                 negative: List[Polygon] = None):
        self.positive = positive or []
        self.negative = negative or []

        if not self.positive:
            raise ValueError("At least one positive polygon is required.")

        if not all(isinstance(p, Polygon) for p in self.positive):
            raise TypeError(
                "All elements in 'positive' must be Polygon instances."
            )
        if not all(isinstance(p, Polygon) for p in self.negative):
            raise TypeError(
                "All elements in 'negative' must be Polygon instances."
            )

        self.unary_pos = unary_union([p.polygon for p in self.positive])
        self.unary_neg = unary_union([p.polygon for p in self.negative]) \
            if self.negative else None

    def difference(self) -> Polygon:
        r"""
        Computes the geometric difference between the union of the
        positive polygons and the union of the negative polygons.

        Returns
        -------
        Polygon
            A new `Polygon` instance representing the result of the
            difference operation.

        Raises
        ------
        NotImplementedError
            If the resulting geometry is a `MultiPolygon`, which is
            currently not supported.
        """
        result = self.unary_pos
        if self.unary_neg:
            result = difference(result, self.unary_neg)

        if isinstance(result, MultiPolygon):
            # TODO: handle multipolygons explicitly if needed
            raise NotImplementedError(
                "Difference resulted in a MultiPolygon, "
                "which is not yet handled.")

        return Polygon(
            points=list(result.exterior.coords),
            holes=[list(interior.coords) for interior in result.interiors]
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
        return self.difference()


class SectorToPolygonHandler:
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

    def __init__(self,
                 polygons: List[Polygon],
                 circular: List[CircularSector]):
        self.polygons = polygons
        self.circular_sector = circular

        if not all(isinstance(p, Polygon) for p in self.polygons):
            raise TypeError(
                "All elements in 'polygons' must be instances of Polygon.")

        if not all(isinstance(cs, CircularSector) for cs in
                   self.circular_sector):
            raise TypeError(
                "All elements in 'circular' must be CircularSector "
                "instances.")

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
        polygons_extended = self.polygons.copy()
        circular_sector_remaining = []

        for i in self.circular_sector:
            sec_as_polygon = i.convert_to_polygon(num_points=100)

            if any(
                self.bounding_boxes_overlap(sec_as_polygon.polygon,
                                            sp.polygon) and
                sec_as_polygon.polygon.intersection(
                    sp.polygon).area > 1e-10
                for sp in polygons_extended
            ):
                print(
                    f"Note: A circular sector {i} intersects a polygon. "
                    "It is discrete into 100 points and "
                    "treated as a polygon."
                )
                polygons_extended.append(sec_as_polygon)
            else:
                circular_sector_remaining.append(i)

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
