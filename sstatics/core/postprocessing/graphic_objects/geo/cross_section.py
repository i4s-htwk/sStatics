

from functools import cached_property
from sstatics.core.preprocessing import CrossSection
from sstatics.core.preprocessing.geometry import Polygon, PolygonMerge, \
    CircularSector
from sstatics.core.postprocessing.graphic_objects.geo.object_geo import \
    ObjectGeo
from sstatics.core.postprocessing.graphic_objects.utils.defaults import (
    DEFAULT_CROSS_SECTION_POSITIVE, DEFAULT_CROSS_SECTION_NEGATIVE,
    DEFAULT_CROSS_SECTION_POINT_STYLE_POSITIVE,
    DEFAULT_CROSS_SECTION_POINT_STYLE_NEGATIVE
)
from sstatics.core.postprocessing.graphic_objects.geo import PolygonGeo


class CrossSectionGeo(ObjectGeo):

    def __init__(
            self,
            cross_section: CrossSection,
            merged: bool = True,
            show_center_of_mass: bool = True,
            **kwargs
    ):
        self._validate_cross_section(
            cross_section, merged, show_center_of_mass
        )
        super().__init__(
            (cross_section.center_of_mass_y, cross_section.center_of_mass_z),
            **kwargs
        )
        self._cross_section = cross_section
        self._merged = merged
        self._show_center_of_mass = show_center_of_mass

    @cached_property
    def graphic_elements(self):
        """Generates the PolygonGeo objects for the cross-section.

        Depending on the `merged` flag:
        - merged=True: a single PolygonGeo is created
        - merged=False: each polygon or circular sector is represented as
          a separate PolygonGeo.

        If the cross-section only contains mechanical inputs, a rectangular
        cross-section will to be created based on the input height and width.

        Returns
        -------
        list[PolygonGeo]
            List of PolygonGeo objects for rendering.
        """
        cs = self.cross_section

        if cs.polygon or cs.circular_sector:
            return [self._merged_polygon_geo(
                cs)] if self.merged else self._unmerged_polygons_geo(cs)

        # Mechanical cross-section
        return [self._mechanical_polygon_geo(cs)]

    def _merged_polygon_geo(self, cs):
        """Creates a single merged PolygonGeo from all polygons and circular
        sectors, taking the sign of the polygons into account.

        Parameters
        ----------
        cs : CrossSection
            The cross-section object containing polygons and circular sectors.

        Returns
        -------
        :any:`PolygonGeo`
            The merged PolygonGeo object.
        """
        positive, negative = self._collect_polygons_by_sign(cs)
        merged_poly = PolygonMerge(positive=positive, negative=negative)()
        return self._polygon_to_geo(merged_poly)

    def _unmerged_polygons_geo(self, cs) -> list[PolygonGeo]:
        """Creates a PolygonGeo for each polygon or circular sector
        without geometric merging.

        Parameters
        ----------
        cs : CrossSection
            The cross-section object containing polygons and circular sectors.

        Returns
        -------
        list[PolygonGeo]
            List of PolygonGeo objects, one for each polygon or circular
            sector.
        """
        return [self._polygon_to_geo(p) for p in cs.geometry]

    def _collect_polygons_by_sign(self, cs):
        """Collects all polygons and circular sectors into positive and
        negative lists.

        Parameters
        ----------
        cs : CrossSection
            The cross-section object containing polygons and circular sectors.

        Returns
        -------
        tuple[list[Polygon], list[Polygon]]
            Lists of positive and negative polygons.
        """
        positive, negative = [], []

        if cs.polygon:
            self._add_by_sign(cs.polygon, positive, negative)

        for sector in cs.circular_sector:
            self._add_by_sign(sector, positive, negative)

        return positive, negative

    @staticmethod
    def _add_by_sign(polygon: Polygon, positive, negative):
        """Appends a polygon to the appropriate list based on its sign.

        Parameters
        ----------
        polygon : Polygon
            The polygon to classify.
        positive : list[Polygon]
            List of positive polygons.
        negative : list[Polygon]
            List of negative polygons.
        """
        (positive if polygon.positive else negative).append(polygon)

    def _mechanical_polygon_geo(self, cs) -> PolygonGeo:
        """Creates a PolygonGeo for a mechanical (rectangular) cross-section.

        Parameters
        ----------
        cs : CrossSection
            The cross-section object containing width and height.

        Returns
        -------
        PolygonGeo
            The PolygonGeo representing the mechanical cross-section.
        """
        poly = Polygon([(0, 0), (cs.width, 0), (cs.width, cs.height),
                        (0, cs.height), (0, 0)])
        return self._polygon_to_geo(poly)

    def _polygon_to_geo(self, geometry) -> PolygonGeo:
        """Converts a Polygon or CircularSector into a PolygonGeo with
        appropriate styles.

        Parameters
        ----------
        geometry : Polygon or CircularSector
            The geometry object to convert.

        Returns
        -------
        PolygonGeo
            The resulting PolygonGeo with line and point styles applied.
        """
        geometry = geometry.convert_to_polygon() if (
            isinstance(geometry, CircularSector)) else geometry

        line_style = DEFAULT_CROSS_SECTION_POSITIVE if (
            geometry.positive) else DEFAULT_CROSS_SECTION_NEGATIVE
        point_style = DEFAULT_CROSS_SECTION_POINT_STYLE_POSITIVE if (
            geometry.positive) else DEFAULT_CROSS_SECTION_POINT_STYLE_NEGATIVE

        return PolygonGeo(
            geometry,
            show_center=self._show_center_of_mass,
            line_style=line_style,
            point_style=point_style,
            text_style=self._text_style
        )

    @cached_property
    def text_elements(self):
        return []

    @staticmethod
    def _validate_cross_section(cross_section, merged, show_center_of_mass):
        if not isinstance(cross_section, CrossSection):
            raise TypeError(
                f'"cross_section" must be a CrossSection, '
                f'got {type(cross_section).__name__!r}'
            )

        if not isinstance(merged, bool):
            raise TypeError(
                f'"merged" must be a boolean, got '
                f'{type(merged).__name__!r}'
            )

        if not isinstance(show_center_of_mass, bool):
            raise TypeError(
                f'"show_center_of_mass" must be a boolean, got '
                f'{type(show_center_of_mass).__name__!r}'
            )

    @property
    def cross_section(self):
        return self._cross_section

    @property
    def merged(self):
        return self._merged

    @property
    def show_center_of_mass(self):
        return self._show_center_of_mass

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'origin={self._origin}, '
            f'cross_section={self._cross_section}, '
            f'merged={self._merged}, '
            f'show_center_of_mass={self._show_center_of_mass}, '
            f'line_style={self._line_style}, '
            f'point_style={self._point_style}, '
            f'text_style={self._text_style}, '
            f'Transform={self._transform})'
        )
