
from sstatics.core.preprocessing.geometry.objects import (
    CircularSector, Polygon
)
from sstatics.core.preprocessing.geometry.operation import PolygonMerge
from sstatics.core.preprocessing.cross_section import CrossSection

from sstatics.graphic_objects.utils import SingleGraphicObject
from sstatics.graphic_objects.geometry import PolygonGraphic


class CrossSectionGraphic(SingleGraphicObject):

    def __init__(
            self, cross_section: CrossSection, merged: bool = True,
            show_center_of_mass: bool = True, **kwargs
    ):
        if not isinstance(cross_section, CrossSection):
            raise TypeError(
                '"cross_section" has to be an instance of CrossSection'
            )
        super().__init__(
            cross_section.center_of_mass_y, cross_section.center_of_mass_z,
            **kwargs
        )
        self.cross_section = cross_section
        self.merged = merged
        self.show_center_of_mass = show_center_of_mass

    def _draw_poly(self, geometry: Polygon | CircularSector):
        geometry = (
            geometry.convert_to_polygon()
            if isinstance(geometry, CircularSector) else geometry
        )
        color = (
            (60, 225, 0, 0.1) if geometry.positive else (255, 0, 0, 0.1)
        )
        return PolygonGraphic(
            geometry, self.show_center_of_mass,
            scatter_options={'fillcolor': f'rgba{color}'},
            rotation=self.rotation, scale=self.scale
        ).traces

    @property
    def _mechanical_cross_section(self):
        cs = self.cross_section
        poly = Polygon([
            (0, 0), (cs.width, 0),
            (cs.width, cs.height), (0, cs.height),
            (0, 0)
        ])
        return self._draw_poly(poly)

    @property
    def traces(self):
        cs = self.cross_section
        if cs.polygon:
            if self.merged:
                geometry = PolygonMerge(
                    [cs.polygon, *[
                        c.convert_to_polygon()for c in cs.circular_sector
                    ]]
                )()
                return self._draw_poly(geometry)
            else:
                traces = []
                [traces.extend(self._draw_poly(g)) for g in cs.geometry]
                return traces
        return self._mechanical_cross_section
