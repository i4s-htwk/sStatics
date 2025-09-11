
import numpy as np
from functools import cached_property

from sstatics.core.preprocessing.bar import Bar
from sstatics.graphic_objects.utils import MultiGraphicObject, transform
from sstatics.graphic_objects.geometry import LineGraphic, EllipseGraphic
from sstatics.graphic_objects.hinges import (
    NoHinge, NormalForceHinge, ShearForceHinge, MomentHinge, CombiHinge
)


# TODO: use?
class BaseBarLine(LineGraphic):

    scatter_options = LineGraphic.scatter_options

    @classmethod
    def from_two_points(cls, point1, point2, **kwargs):
        kwargs.setdefault('line', cls.scatter_options['line'])
        return super().from_points([point1, point2], **kwargs)


class BarLine(BaseBarLine):

    scatter_options = LineGraphic.scatter_options | {
        'line': dict(width=4),
    }


class TensileZone(BaseBarLine):

    scatter_options = LineGraphic.scatter_options | {
        'line': dict(dash='dash', width=1),
    }


class BarGraphic(MultiGraphicObject):

    def __init__(
            self, bar: Bar, bar_number=None, base_scale=None, max_dim=None,
            show_annotations: bool = True,
            **kwargs
    ):
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
        self.number = bar_number
        self.base_scale = base_scale
        self.max_dim = max_dim
        self.show_annotations = show_annotations

    @cached_property
    def _annotation_pos(self):
        x_mid = (self.x_i + self.x_j) / 2
        z_mid = (self.z_i + self.z_j) / 2
        x_orth_vec = -(self.z_j - self.z_i)
        z_orth_vec = self.x_j - self.x_i
        length = np.hypot(x_orth_vec, z_orth_vec)
        off_scale = 0.5 * self._base_scale
        return (
            x_mid - (x_orth_vec / length) * off_scale,
            z_mid - (z_orth_vec / length) * off_scale
        )

    @cached_property
    def _max_dim(self):
        return self.max_dim if self.max_dim \
            else max(abs(self.x_i - self.x_j), abs(self.z_i - self.z_j))

    @cached_property
    def _base_scale(self):
        return self.base_scale if self.base_scale \
            else 0.08 * self._max_dim  # TODO: + 0.02

    @cached_property
    def tensile_translation(self):
        return np.array([
            np.sin(self.bar.inclination), np.cos(self.bar.inclination),
        ]) * 0.1 * self._base_scale

    @cached_property
    def _unit_dir_vec(self):
        return np.array([
            np.cos(self.bar.inclination), -np.sin(self.bar.inclination),
        ])

    @property
    def _hinge_cases(self):
        hinge_cases = {
            (False, False, False): (NoHinge,),
            (False, False, True): (MomentHinge,),
            (False, True, False): (ShearForceHinge,),
            (False, True, True): (ShearForceHinge, MomentHinge),
            (True, False, False): (NormalForceHinge,),
            (True, False, True): (MomentHinge, NormalForceHinge),
            (True, True, False): (ShearForceHinge, NormalForceHinge),
            (True, True, True): (NoHinge,)  # TODO: noch verbessern
        }
        return (
            hinge_cases.get(self.bar.hinge[0:3], (NoHinge,)),
            hinge_cases.get(self.bar.hinge[3:6], (NoHinge,))
        )

    def _hinge_factors(self, hinge_type):
        if hinge_type is NoHinge:
            return 0.0, 0.0
        else:
            scaled_width = hinge_type.width * self._base_scale
            if hinge_type is NormalForceHinge:
                return 1 / 4 * scaled_width, 3 / 4 * scaled_width
            return 1 / 2 * scaled_width, 0.0

    @property
    def _hinge_translations(self):
        hinge_i, hinge_j = self._hinge_cases
        trans_i, _ = self._hinge_factors(hinge_i[0])
        trans_j, _ = self._hinge_factors(hinge_j[0])
        return (
            trans_i * self._unit_dir_vec,
            -trans_j * self._unit_dir_vec
        )

    @cached_property
    def _hinge_graphics(self):
        translation_i, translation_j = self._hinge_translations
        hinge_i, hinge_j = self._hinge_cases
        return (
            CombiHinge(
                self.x_i, self.z_i, *hinge_i,
                scatter_options=self.scatter_kwargs,
                rotation=self.bar.inclination, scale=self._base_scale,
                translation=translation_i),
            CombiHinge(
                self.x_j, self.z_j, *hinge_j,
                scatter_options=self.scatter_kwargs,
                rotation=self.bar.inclination + np.pi, scale=self._base_scale,
                translation=translation_j)
        )

    @property
    def _bar_stretch(self):
        hinge_i, hinge_j = self._hinge_graphics
        _, stretch_i = self._hinge_factors(hinge_i.get_last_hinge)
        _, stretch_j = self._hinge_factors(hinge_j.get_last_hinge)
        return (
            (-hinge_i.total_width + stretch_i),
            (-hinge_j.total_width + stretch_j)
        )

    @property
    def _annotations(self):
        if not self.show_annotations or self.number is None:
            return ()
        x, z = transform(
            self.x_i, self.z_i, *self._annotation_pos, self.rotation,
            self.scale
        )
        return (
            () if self.number is None
            else ((x, z, self.number),)
        )

    @property
    def traces(self):
        # TODO: use extra classes for BarLine and TensileZone
        traces = []
        traces.extend(
            LineGraphic.from_points(
                [(self.x_i, self.z_i), (self.x_j, self.z_j)],
                scatter_options={'line': dict(width=4)} | self.scatter_kwargs
            ).stretching(
                *self._bar_stretch
            ).transform_traces(self.x_i, self.z_i, self.rotation, self.scale)
        )
        traces.extend(
            LineGraphic.from_points(
                [(self.x_i, self.z_i), (self.x_j, self.z_j)],
                scatter_options=(
                    {'line': dict(dash='dash', width=1)} | self.scatter_kwargs
                )
            ).stretching(
                *self._bar_stretch
            ).transform_traces(
                self.x_i - self.tensile_translation[0],
                self.z_i - self.tensile_translation[1], self.rotation,
                self.scale, self.tensile_translation
            )
        )
        # TODO: create separate class for bar number
        if self.show_annotations:
            traces.extend(
                EllipseGraphic(
                    *self._annotation_pos, 0.25 * self._base_scale
                ).transform_traces(self.x_i, self.z_i, self.rotation,
                                   self.scale)
            )

        # traces.extend(
        #     BarLine.from_two_points(
        #         (self.x_i, self.z_i), (self.x_j, self.z_j),
        #         scatter_options=self.scatter_kwargs
        #     ).stretching(
        #         *self.bar_stretch_factors
        #     ).transform_traces(self.x_i, self.z_i, self.rotation, self.scale)
        # )
        # traces.extend(
        #     TensileZone.from_two_points(
        #         (self.x_i, self.z_i), (self.x_j, self.z_j),
        #         scatter_options=self.scatter_kwargs
        #     ).stretching(
        #         *self.bar_stretch_factors
        #     ).transform_traces(
        #         self.x_i, self.z_i, self.rotation, self.scale,
        #         self.tensile_translation
        #     )
        # )
        # TODO: optimise hinge-methods
        for h in self._hinge_graphics:
            traces.extend(
                h.transform_traces(
                    self.x_i, self.z_i, self.rotation, self.scale
                )
            )

        # TODO: deformation

        # TODO: line_load

        # TODO: temp

        # TODO: point_load

        return traces
