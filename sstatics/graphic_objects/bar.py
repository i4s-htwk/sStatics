
import numpy as np
from functools import cached_property

from sstatics.core.bar import Bar
from sstatics.graphic_objects.utils import (
    MultiGraphicObject, EmptyGraphicObject
)
from sstatics.graphic_objects.geometry import Line
from sstatics.graphic_objects.hinges import (
    NormalForceHinge, ShearForceHinge, MomentHinge
)


# TODO: Konfiguration-Datei
HINGE_FACTORS = {
    NormalForceHinge: (f := 11 / 80, -f),
    ShearForceHinge: (f := 11 / 160, -2 * f),
    MomentHinge: (f := 11 / 80, -2 * f),
    EmptyGraphicObject: (0.0, 0.0),
}


class BaseBarLine(Line):

    scatter_options = Line.scatter_options

    @classmethod
    def from_two_points(cls, point1, point2, **kwargs):
        kwargs.setdefault('line', cls.scatter_options['line'])
        return super().from_points([point1, point2], **kwargs)


class BarLine(BaseBarLine):

    scatter_options = Line.scatter_options | {
        'line': dict(width=4),
    }


class TensileZone(BaseBarLine):

    scatter_options = Line.scatter_options | {
        'line': dict(dash='dash', width=1),
    }


class GraphicBar(MultiGraphicObject):

    def __init__(self, bar: Bar, base_scale=None, **kwargs):
        if not isinstance(bar, Bar):
            raise TypeError('"bar" has to be an instance of Bar')
        super().__init__(
            [(bar.node_i.x, bar.node_i.z), (bar.node_j.x, bar.node_j.z)],
            **kwargs
        )
        self.bar = bar
        self.node_i = bar.node_i
        self.node_j = bar.node_j
        self.x_i, self.z_i = self.node_i.x, self.node_i.z
        self.x_j, self.z_j = self.node_j.x, self.node_j.z
        self.base_scale = base_scale

    @cached_property
    def _base_scale(self):
        return self.base_scale if self.base_scale \
            else 0.08 * self.bar.length  # TODO: + 0.02

    @cached_property
    def tensile_translation(self):
        return np.array([
            np.sin(self.bar.inclination), np.cos(self.bar.inclination),
        ]) * 0.075 * self._base_scale

    def select_hinge(self, x, z, hinge_type):
        hinge_cases = {
            (False, False, False): EmptyGraphicObject,
            (False, False, True): MomentHinge,
            (False, True, False): ShearForceHinge,
            (False, True, True): [],
            (True, False, False): NormalForceHinge,
            (True, False, True): [],
            (True, True, False): [],
            (True, True, True): []
        }
        hinge = hinge_cases.get(hinge_type, EmptyGraphicObject)
        if hinge is EmptyGraphicObject:
            return hinge()
        return hinge(x, z, **self.scatter_kwargs)

    @cached_property
    def hinges(self):
        hinge_i = self.select_hinge(self.x_i, self.z_i, self.bar.hinge[0:3])
        hinge_j = self.select_hinge(self.x_j, self.z_j, self.bar.hinge[3:6])
        return hinge_i, hinge_j

    @cached_property
    def hinge_factors(self):
        hinge_i, hinge_j = self.hinges
        translation_i, stretch_i = (
            HINGE_FACTORS.get(type(hinge_i), (0.0, 0.0))
        )
        translation_j, stretch_j = (
            HINGE_FACTORS.get(type(hinge_j), (0.0, 0.0))
        )
        return (translation_i, translation_j), (stretch_i, stretch_j)

    @cached_property
    def hinge_translation_factors(self):
        return self.hinge_factors[0]

    def hinge_translation(self, translation_factor=1.0):
        return np.array([
            np.cos(self.bar.inclination), -np.sin(self.bar.inclination),
        ]) * self._base_scale * translation_factor

    @cached_property
    def create_hinges(self):
        hinge_i, hinge_j = self.hinges
        translation_i, translation_j = self.hinge_translation_factors
        yield (
            hinge_i, self.bar.inclination,
            self.hinge_translation(translation_i)
        )
        yield (
            hinge_j, self.bar.inclination + np.pi,
            self.hinge_translation(-translation_j)
        )

    @cached_property
    def bar_stretch_factors(self):
        return tuple(fac * self._base_scale for fac in self.hinge_factors[1])

    @property
    def traces(self):
        traces = []

        traces.extend(
            BarLine.from_two_points(
                (self.x_i, self.z_i), (self.x_j, self.z_j),
                **self.scatter_kwargs
            ).stretching(
                *self.bar_stretch_factors
            ).transform_traces(self.x_i, self.z_i, self.rotation, self.scale)
        )
        traces.extend(
            TensileZone.from_two_points(
                (self.x_i, self.z_i), (self.x_j, self.z_j),
                **self.scatter_kwargs
            ).stretching(
                *self.bar_stretch_factors
            ).transform_traces(
                self.x_i, self.z_i, self.rotation, self.scale,
                self.tensile_translation
            )
        )

        # TODO: hinges
        for hinge, rotation, translation in self.create_hinges:
            traces.extend(
                hinge.transform_traces(
                    hinge.x, hinge.z, rotation, self._base_scale, translation
                )
            )

        # TODO: deformation

        # TODO: line_load

        # TODO: temp

        # TODO: point_load

        return traces
