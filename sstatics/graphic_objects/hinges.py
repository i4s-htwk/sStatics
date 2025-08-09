
import numpy as np
from abc import ABC
import plotly.graph_objs as go

from sstatics.graphic_objects import (
    transform, SingleGraphicObject, LineGraphic, EllipseGraphic
)


class Hinge(SingleGraphicObject, ABC):

    scatter_options = SingleGraphicObject.scatter_options | {
        'line': dict(width=3),
    }

    def __init__(self, x, z, width, **kwargs):
        if width < 0:
            raise ValueError(
                '"width" has to be a number greater than or equal to zero.'
            )
        super().__init__(x, z, **kwargs)
        self.width = width


class NoHinge(Hinge):
    width = 0

    def __init__(self, x, z, width=width, **kwargs):
        super().__init__(x, z, width, **kwargs)

    @property
    def traces(self):
        return []


class NormalForceHinge(Hinge):
    width = 11 / 20

    def __init__(self, x, z, width=width, **kwargs):
        super().__init__(x, z, width, **kwargs)

    @property
    def traces(self):
        x_off, z_off = self.width, 11 / 60
        x = np.array(
            [self.x + 3 / 4 * x_off, -1 / 4 * x_off + self.x,
             -1 / 4 * x_off + self.x, self.x + 3 / 4 * x_off]
        )
        z = np.array([
            self.z - z_off, self.z - z_off,
            self.z + z_off, self.z + z_off
        ])
        x, z = transform(self.x, self.z, x, z, self.rotation, self.scale)
        return go.Scatter(x=x, y=z, **self.scatter_kwargs),


class ShearForceHinge(Hinge):
    width = 11 / 80

    def __init__(self, x, z, width=width, **kwargs):
        super().__init__(x, z, width=width, **kwargs)

    @property
    def traces(self):
        length, x_offset = 11 / 20, self.width / 2
        left_line = LineGraphic.from_center(
            self.x - x_offset, self.z, length,
            scatter_options=self.scatter_kwargs, rotation=np.pi / 2
        )
        left_line_traces = left_line.transform_traces(
            self.x, self.z, self.rotation, self.scale
        )
        right_line = LineGraphic.from_center(
            self.x + x_offset, self.z, length,
            scatter_options=self.scatter_kwargs, rotation=np.pi / 2
        )
        right_line_traces = right_line.transform_traces(
            self.x, self.z, self.rotation, self.scale
        )
        return *left_line_traces, *right_line_traces


class MomentHinge(Hinge, EllipseGraphic):
    width = 11 / 40

    scatter_options = Hinge.scatter_options | {
        'line': dict(width=2),
        'fill': 'toself',
        'fillcolor': 'white',
    }

    def __init__(self, x, z, width=width, **kwargs):
        super().__init__(x, z, a=width / 2, width=width, **kwargs)


class CombiHinge(SingleGraphicObject):

    def __init__(self, x, z, *hinges, **kwargs):
        hinge_types = tuple(
            type(h) if isinstance(h, Hinge) else h
            for h in hinges
        )
        if not all(
                isinstance(h, type) and issubclass(h, Hinge)
                for h in hinge_types
        ):
            raise ValueError(
                "each hinge must be an instance or a subclass of Hinge"
            )
        super().__init__(x, z, **kwargs)
        self.hinge_types = hinge_types

    @property
    def get_last_hinge(self):
        return self.hinge_types[-1]

    @staticmethod
    def _get_factor(hinge_type, is_prev):
        width = hinge_type.width
        if hinge_type is NormalForceHinge:
            factor = 3 / 4 if is_prev else 1 / 4
            return factor * width
        return 1 / 2 * width

    @property
    def total_width(self):
        return sum(h.width for h in self.hinge_types) * self.scale

    def _get_offset(self, i):
        if i == 0:
            return 0.0
        prev, curr = self.hinge_types[i - 1], self.hinge_types[i]
        return (
            self._get_factor(prev, is_prev=True) +
            self._get_factor(curr, is_prev=False)
        )

    @property
    def traces(self):
        traces = []
        x = self.x
        for i, hinge_cls in enumerate(self.hinge_types):
            x += self._get_offset(i)
            hinge_obj = hinge_cls(
                x, self.z, scatter_options=self.scatter_kwargs
            ).transform_traces(
                self.x, self.z, self.rotation, self.scale, self.tranlation
            )
            traces.extend(hinge_obj)
        return traces
