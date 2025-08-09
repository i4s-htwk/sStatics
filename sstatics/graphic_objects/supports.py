
import numpy as np
from abc import ABC

from sstatics.graphic_objects import (
    SingleGraphicObject, PointGraphic, LineGraphic, IsoscelesTriangleGraphic,
    Hatching
)


class Support(SingleGraphicObject, ABC):

    scatter_options = SingleGraphicObject.scatter_options | {
        'line': dict(width=2),
        'fill': 'toself',
        'fillcolor': 'white',
    }

    def __init__(self, x, z, width, **kwargs):
        if width <= 0:
            raise ValueError('"width" has to be a number greater than zero.')
        super().__init__(x, z, **kwargs)
        self.width = width
        self.offset = 11 / 10


class BaseLineHatchSupport(Support):

    def __init__(self, x, z, width=11/10, **kwargs):
        super().__init__(x, z, width, **kwargs)

    @property
    def traces(self):
        line = LineGraphic.from_center(
            self.x, self.z, self.width, scatter_options=self.scatter_kwargs,
            rotation=np.pi / 2,
        )
        line_traces = line.transform_traces(
            self.x, self.z, self.rotation, self.scale
        )
        hatching = Hatching(
            self.x - self.offset / 8, self.z, self.width, self.offset / 4,
            scatter_options=self.scatter_kwargs, rotation=np.pi / 2
        )
        hatching_traces = hatching.transform_traces(
            self.x, self.z, self.rotation, self.scale
        )
        return *line_traces, *hatching_traces


class BaseDoubleLineHatchSupport(Support):

    def __init__(self, x, z, width=11/10, **kwargs):
        super().__init__(x, z, width, **kwargs)

    @property
    def traces(self):
        line = LineGraphic.from_center(
            self.x, self.z, self.width, scatter_options=self.scatter_kwargs,
            rotation=np.pi / 2
        )
        line_traces = line.transform_traces(
            self.x, self.z, self.rotation, self.scale
        )
        base_support = BaseLineHatchSupport(
            self.x - self.offset / 4, self.z, self.width,
            scatter_options=self.scatter_kwargs
        )
        base_support_traces = base_support.transform_traces(
            self.x, self.z, self.rotation, self.scale
        )
        return *line_traces, *base_support_traces


FreeNode = PointGraphic
""" Alias of :py:class:`PointGraphic` to make the use case of
this class more clear. """


class RollerSupport(Support):

    def __init__(self, x, z, width=11/10, **kwargs):
        super().__init__(x, z, width, **kwargs)

    @property
    def traces(self):
        triangle = IsoscelesTriangleGraphic.from_width(
            self.x, self.z, self.width, scatter_options=self.scatter_kwargs
        )
        triangle_traces = triangle.transform_traces(
            self.x, self.z, self.rotation, self.scale
        )
        z_off = 2 / 3 * np.sqrt(4 - self.width ** 2)
        line = LineGraphic.from_center(
            self.x, self.z + z_off, self.width,
            scatter_options=self.scatter_kwargs
        )
        line_traces = line.transform_traces(
            self.x, self.z, self.rotation, self.scale
        )
        return *triangle_traces, *line_traces


class PinnedSupport(Support):

    def __init__(self, x, z, width=11/10, **kwargs):
        super().__init__(x, z, width, **kwargs)

    @property
    def traces(self):
        top_line = LineGraphic.from_center(
            self.x + 3 / 8 * self.offset, self.z - 3 / 8 * self.width,
            5 / 4 * self.offset, scatter_options=self.scatter_kwargs
        )
        top_line_traces = top_line.transform_traces(
            self.x, self.z, self.rotation, self.scale
        )
        bottom_line = LineGraphic.from_center(
            self.x + 3 / 8 * self.offset, self.z + 3 / 8 * self.width,
            5 / 4 * self.offset, scatter_options=self.scatter_kwargs
        )
        bottom_line_traces = bottom_line.transform_traces(
            self.x, self.z, self.rotation, self.scale
        )
        base_support = BaseDoubleLineHatchSupport(
            self.x - self.offset / 4, self.z, self.width,
            scatter_options=self.scatter_kwargs
        )
        base_support_traces = base_support.transform_traces(
            self.x, self.z, self.rotation, self.scale
        )
        return *top_line_traces, *bottom_line_traces, *base_support_traces


class FixedSupportUW(Support):

    def __init__(self, x, z, width=11/10, **kwargs):
        super().__init__(x, z, width, **kwargs)

    @property
    def traces(self):
        triangle = IsoscelesTriangleGraphic.from_width(
            self.x, self.z, self.width, scatter_options=self.scatter_kwargs
        )
        triangle_traces = triangle.transform_traces(
            self.x, self.z, self.rotation, self.scale
        )
        z_off = np.sqrt(4 - self.width ** 2) / 2
        hatching = Hatching(
            self.x, self.z + 7 / 6 * z_off, self.width, 1 / 3 * z_off,
            scatter_options=self.scatter_kwargs
        )
        hatching_traces = hatching.transform_traces(
            self.x, self.z, self.rotation, self.scale
        )
        return *triangle_traces, *hatching_traces


FixedSupportUPhi = BaseDoubleLineHatchSupport
""" Alias of :py:class:`BaseDoubleLineHatchSupport` to make the use case of
this class more clear. """


class FixedSupportWPhi(Support):

    def __init__(self, x, z, width=11/10, **kwargs):
        super().__init__(x, z, width, **kwargs)

    @property
    def traces(self):
        top_line = LineGraphic.from_center(
            self.x + 3 / 8 * self.offset, self.z - 3 / 8 * self.width,
            5 / 4 * self.offset, scatter_options=self.scatter_kwargs
        )
        top_line_traces = top_line.transform_traces(
            self.x, self.z, self.rotation, self.scale
        )
        bottom_line = LineGraphic.from_center(
            self.x + 3 / 8 * self.offset, self.z + 3 / 8 * self.width,
            5 / 4 * self.offset, scatter_options=self.scatter_kwargs
        )
        bottom_line_traces = bottom_line.transform_traces(
            self.x, self.z, self.rotation, self.scale
        )
        base_support = BaseLineHatchSupport(
            self.x - self.offset / 4, self.z, self.width,
            scatter_options=self.scatter_kwargs
        )
        base_support_traces = base_support.transform_traces(
            self.x, self.z, self.rotation, self.scale
        )
        return *top_line_traces, *bottom_line_traces, *base_support_traces


ChampedSupport = BaseLineHatchSupport
""" Alias of :py:class:`BaseLineHatchSupport` to make the use case of this
class more clear. """
