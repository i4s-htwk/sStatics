
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


class RollerSupport(Support, IsoscelesTriangle):

    def __init__(self, x, z, width=4/3, **kwargs):
        super().__init__(x, z, width=width, **kwargs)
        self.angle = 2 * np.arctan(self.width / (2 * self.scale))

    @property
    def traces(self):
        line = Line(self.x, self.z + 4/3 * self.scale, scale=self.scale)
        line_traces = line.rotate_traces(self.x, self.z, self.rotation)
        return *line_traces, *super().traces


class FixedSupportUW(Support, IsoscelesTriangle):

    def __init__(self, x, z, width=4/3, **kwargs):
        super().__init__(x, z, width=width, **kwargs)
        self.angle = 2 * np.arctan(self.width / (2 * self.scale))

    @property
    def traces(self):
        hatching = Hatching(
            self.x, self.z + self.scale * 7/6,
            self.width / self.scale, 1/3, scale=self.scale
        )
        hatching_traces = hatching.rotate_traces(self.x, self.z, self.rotation)
        return *hatching_traces, *super().traces


class FixedSupportUPhi(Support):

    def __init__(self, x, z, width=4/3, **kwargs):
        super().__init__(x, z, width=width, **kwargs)

    @property
    def traces(self):
        line_1 = Line(self.x, self.z, scale=self.scale)
        line_traces_1 = line_1.rotate_traces(self.x, self.z, self.rotation)
        line_2 = Line(self.x, self.z + self.scale * 1/3, scale=self.scale)
        line_traces_2 = line_2.rotate_traces(self.x, self.z, self.rotation)
        hatching = Hatching(
            self.x, self.z + self.scale * 1/2,
            self.width / self.scale, 1/3, scale=self.scale
        )
        hatching_traces = hatching.rotate_traces(self.x, self.z, self.rotation)
        return *line_traces_1, *line_traces_2, *hatching_traces


class ChampedSupport(Support):

    def __init__(self, x, z, width=4/3, **kwargs):
        super().__init__(x, z, width=width, **kwargs)

    @property
    def traces(self):
        line = Line(self.x, self.z, scale=self.scale)
        line_traces = line.rotate_traces(self.x, self.z, self.rotation)
        hatching = Hatching(
            self.x, self.z + self.scale * 1/6,
            self.width / self.scale, 1/3, scale=self.scale
        )
        hatching_traces = hatching.rotate_traces(self.x, self.z, self.rotation)
        return *line_traces, *hatching_traces
