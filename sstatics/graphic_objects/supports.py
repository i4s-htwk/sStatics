
import numpy as np
from sstatics.graphic_objects import (
    GraphicObject, Line, IsoscelesTriangle, Hatching
)


class Support(GraphicObject):

    def __init__(self, x, z, width, **kwargs):
        if width <= 0:
            raise ValueError('"width" has to be a number greater than zero.')
        super().__init__(x, z, **kwargs)
        self.width = width * self.scale

    @property
    def offset(self):
        return 11 / 10 * self.scale


class BaseLineHatchSupport(Support):

    def __init__(self, x, z, width=11/10, **kwargs):
        super().__init__(x, z, width=width, **kwargs)

    @property
    def traces(self):
        line = Line(
            self.x, self.z, self.width, rotation=np.pi / 2,
            **self.scatter_kwargs
        )
        line_traces = line.rotate_traces(self.x, self.z, self.rotation)
        hatching = Hatching(
            self.x - self.offset / 8, self.z, self.width, self.offset / 4,
            spacing=.17 * self.scale, rotation=np.pi/2, **self.scatter_kwargs
        )
        hatching_traces = hatching.rotate_traces(self.x, self.z, self.rotation)
        return *line_traces, *hatching_traces


class BaseDoubleLineHatchSupport(Support):

    def __init__(self, x, z, width=11/10, **kwargs):
        super().__init__(x, z, width=width, **kwargs)

    @property
    def traces(self):
        line = Line(
            self.x, self.z, self.width, rotation=np.pi / 2,
            **self.scatter_kwargs
        )
        line_traces = line.rotate_traces(self.x, self.z, self.rotation)
        base_support = BaseLineHatchSupport(
            self.x - self.offset / 4, self.z, self.width / self.scale,
            scale=self.scale, **self.scatter_kwargs
        )
        base_support_traces = base_support.rotate_traces(
            self.x, self.z, self.rotation
        )
        return *line_traces, *base_support_traces


class RollerSupport(Support):

    def __init__(self, x, z, width=11/10, **kwargs):
        super().__init__(x, z, width=width, **kwargs)

    @property
    def traces(self):
        triangle = IsoscelesTriangle(
            self.x, self.z, width=self.width, scale=self.scale,
            **self.scatter_kwargs
        )
        triangle_traces = triangle.rotate_traces(self.x, self.z, self.rotation)
        z_offset = np.sqrt(self.scale ** 2 - (self.width / 2) ** 2) * 4 / 3
        line = Line(
            self.x, self.z + z_offset, self.width, **self.scatter_kwargs
        )
        line_traces = line.rotate_traces(self.x, self.z, self.rotation)
        return *triangle_traces, *line_traces


class PinnedSupport(Support):

    def __init__(self, x, z, width=11/10, **kwargs):
        super().__init__(x, z, width=width, **kwargs)

    @property
    def traces(self):
        top_line = Line(
            self.x + 3 / 8 * self.offset, self.z - 3 / 8 * self.width,
            5 / 4 * self.offset, **self.scatter_kwargs
        )
        top_line_traces = top_line.rotate_traces(
            self.x, self.z, self.rotation
        )
        bottom_line = Line(
            self.x + 3 / 8 * self.offset, self.z + 3 / 8 * self.width,
            5 / 4 * self.offset, **self.scatter_kwargs
        )
        bottom_line_traces = bottom_line.rotate_traces(
            self.x, self.z, self.rotation
        )
        base_support = BaseDoubleLineHatchSupport(
            self.x - self.offset / 4, self.z, self.width / self.scale,
            scale=self.scale, **self.scatter_kwargs
        )
        base_support_traces = base_support.rotate_traces(
            self.x, self.z, self.rotation
        )
        return *top_line_traces, *bottom_line_traces, *base_support_traces


class FixedSupportUW(Support):

    def __init__(self, x, z, width=11/10, **kwargs):
        super().__init__(x, z, width=width, **kwargs)

    @property
    def traces(self):
        triangle = IsoscelesTriangle(
            self.x, self.z, width=self.width, scale=self.scale,
            **self.scatter_kwargs
        )
        triangle_traces = triangle.rotate_traces(self.x, self.z, self.rotation)
        z_offset = np.sqrt(self.scale ** 2 - (self.width / 2) ** 2) * 7 / 6
        hatching = Hatching(
            self.x, self.z + z_offset, self.width, 2/7 * z_offset,
            spacing=.17 * self.scale, **self.scatter_kwargs
        )
        hatching_traces = hatching.rotate_traces(self.x, self.z, self.rotation)
        return *triangle_traces, *hatching_traces


FixedSupportUPhi = BaseDoubleLineHatchSupport
""" Alias of :py:class:`BaseDoubleLineHatchSupport` to make the use case of
this class more clear. """


class FixedSupportWPhi(Support):

    def __init__(self, x, z, width=11/10, **kwargs):
        super().__init__(x, z, width=width, **kwargs)

    @property
    def traces(self):
        top_line = Line(
            self.x + 3 / 8 * self.offset, self.z - 3 / 8 * self.width,
            5 / 4 * self.offset, **self.scatter_kwargs
        )
        top_line_traces = top_line.rotate_traces(
            self.x, self.z, self.rotation
        )
        bottom_line = Line(
            self.x + 3 / 8 * self.offset, self.z + 3 / 8 * self.width,
            5 / 4 * self.offset, **self.scatter_kwargs
        )
        bottom_line_traces = bottom_line.rotate_traces(
            self.x, self.z, self.rotation
        )
        base_support = BaseLineHatchSupport(
            self.x - self.offset / 4, self.z, self.width / self.scale,
            scale=self.scale, **self.scatter_kwargs
        )
        base_support_traces = base_support.rotate_traces(
            self.x, self.z, self.rotation
        )
        return *top_line_traces, *bottom_line_traces, *base_support_traces


ChampedSupport = BaseLineHatchSupport
""" Alias of :py:class:`BaseLineHatchSupport` to make the use case of this
class more clear. """
