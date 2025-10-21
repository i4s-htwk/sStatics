
import numpy as np

from sstatics.core.preprocessing.loads import PointLoad

from sstatics.graphic_objects.utils import (
    SingleGraphicObject, transform
)
from sstatics.graphic_objects.diagram import StraightArrow, CurvedArrow


class PointLoadGraphic(SingleGraphicObject):
    BASE_ROTATION_X = np.pi / 2
    BASE_ROTATION_Z = 0
    BASE_ROTATION_PHI = np.pi / 4

    def __init__(self, x: float, z: float, load: PointLoad,
                 rotate_moment: float = 0, **kwargs):
        if not isinstance(load, PointLoad):
            raise TypeError('"load" must be NodePointLoad or BarPointLoad')
        super().__init__(x, z, **kwargs)
        self.load = load
        self.offset = 11 / 10
        self.rotate_moment = rotate_moment

    @property
    def _annotation_pos(self):
        return [
            (self.x - 3, self.z - self.offset / 2),
            (self.x - self.offset / 2, self.z - 3),
            (self.x + 3 * 0.9, self.z)
        ]

    @property
    def _load_rot(self):
        return [
            0 if self.load.x > 0 else np.pi,
            0 if self.load.z > 0 else np.pi
        ]

    def _annotation_trans(self, x, z, extra_rotation):
        return transform(
            self.x, self.z, x, z, rotation=self.load.rotation + extra_rotation,
            scale=self.scale
        )

    @property
    def _annotations(self):
        x, z = np.array(list(zip(*self._annotation_pos)))
        x[0], z[0] = self._annotation_trans(x[0], z[0], self._load_rot[0])
        x[1], z[1] = self._annotation_trans(x[1], z[1], self._load_rot[1])
        x[2], z[2] = self._annotation_trans(x[2], z[2], self.rotate_moment)

        return (
            (x[0], z[0], abs(float(self.load.x))),
            (x[1], z[1], abs(float(self.load.z))),
            (x[2], z[2], abs(float(self.load.phi))),
        )

    @property
    def traces(self):
        traces = []
        if self.load.x:
            traces.extend(self._x_load_traces)
        if self.load.z:
            traces.extend(self._z_load_traces)
        if self.load.phi:
            traces.extend(self._phi_load_traces)
        return traces

    @property
    def _x_load_traces(self):
        return StraightArrow(
            self.x, self.z, offset=self.offset,
            scatter_options=self.scatter_kwargs,
            rotation=(
                    self.BASE_ROTATION_X + self.load.rotation
                    + self._load_rot[0]
            )
        ).transform_traces(self.x, self.z, self.rotation, self.scale)

    @property
    def _z_load_traces(self):
        return StraightArrow(
            self.x, self.z, offset=self.offset,
            scatter_options=self.scatter_kwargs,
            rotation=(
                   self.BASE_ROTATION_Z + self.load.rotation
                   + self._load_rot[1]
            )
        ).transform_traces(self.x, self.z, self.rotation, self.scale)

    @property
    def _phi_load_traces(self):
        if self.load.phi > 0:
            angle_span = (np.pi / 2, 0)
        else:
            angle_span = (0, np.pi / 2)
        return CurvedArrow(
            self.x, self.z, angle_span=angle_span,
            scatter_options=self.scatter_kwargs,
            rotation=(np.pi / 4 + self.load.rotation + self.rotate_moment)
        ).transform_traces(self.x, self.z, self.rotation, self.scale)
