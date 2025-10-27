
from __future__ import annotations
import numpy as np
from sympy.core.cache import cached_property

from .geometry import ClosedCurveGeo
from ..geo.object_geo import ObjectGeo


class HatchGeo(ObjectGeo):

    def __init__(
            self,
            shape: ClosedCurveGeo,
            spacing: float = 0.2,
            angle: float = np.pi / 4,
            **kwargs
    ):
        self._validate_init(shape, spacing, angle)
        super().__init__(origin=shape.origin, **kwargs)
        self._shape = shape
        self._spacing = spacing
        self._angle = angle
        self._np_points = self._extract_polygon_points()

    @cached_property
    def graphic_elements(self):
        return self._generate_hatch_elements()

    @cached_property
    def text_elements(self):
        x0, z0 = self._origin
        return [(x0, z0, self._text, self._text_style)]

    def _generate_hatch_elements(self):
        x_min, z_min = self._np_points.min(axis=0)
        x_max, z_max = self._np_points.max(axis=0)
        diag = np.hypot(x_max - x_min, z_max - z_min)
        max_len = 5 * diag

        center = np.array([(x_min + x_max) / 2, (z_min + z_max) / 2])
        direction = np.array([np.cos(self._angle), np.sin(self._angle)])
        normal = np.array([-direction[1], direction[0]])

        offsets = np.arange(-diag + self._spacing / 2, diag, self._spacing)

        lines = []
        for offset in offsets:
            c = center + offset * normal
            p0 = c - direction * max_len
            p1 = c + direction * max_len

            l_segments = self._clip_line_to_polygon(p0, p1)
            for l_segment in l_segments:
                x = [p[0] for p in l_segment]
                z = [p[1] for p in l_segment]
                lines.append((x, z, self._line_style))

        return lines

    def _extract_polygon_points(self):
        all_points = []
        for x_arr, z_arr in self._shape.shape_coords:
            clean_points = [
                (x, z) for x, z in zip(x_arr, z_arr)
                if x is not None and z is not None
            ]
            all_points.extend(clean_points)
        return np.array(all_points)

    def _clip_line_to_polygon(self, p0, p1):
        intersections = []
        for i in range(len(self._np_points)):
            q0 = self._np_points[i]
            q1 = self._np_points[(i + 1) % len(self._np_points)]

            pt = self._line_segment_intersection(p0, p1, q0, q1)
            if pt is not None:
                t = self._projective_param(p0, p1, pt)
                intersections.append((t, pt))

        intersections.sort(key=lambda pair: pair[0])

        lines = []
        for i in range(0, len(intersections) - 1, 2):
            mid_x = (intersections[i][1][0] + intersections[i + 1][1][0]) / 2
            mid_y = (intersections[i][1][1] + intersections[i + 1][1][1]) / 2
            if self._point_in_polygon(mid_x, mid_y):
                lines.append([intersections[i][1], intersections[i + 1][1]])
        return lines

    @staticmethod
    def _projective_param(p0, p1, pt):
        d = p1 - p0
        if np.allclose(d, 0):
            return 0
        return np.dot(pt - p0, d) / np.dot(d, d)

    @staticmethod
    def _line_segment_intersection(p0, p1, q0, q1):
        r = p1 - p0
        s = q1 - q0

        cross_prod = np.cross(r, s)
        diff = q0 - p0

        if np.isclose(cross_prod, 0):
            return None

        t = np.cross(diff, s) / cross_prod
        u = np.cross(diff, r) / cross_prod

        if 0 <= u <= 1:
            return p0 + t * r
        return None

    def _point_in_polygon(self, x, y):
        num = len(self._np_points)
        j = num - 1
        inside = False
        for i in range(num):
            xi, zi = self._np_points[i]
            xj, zj = self._np_points[j]
            if ((zi > y) != (zj > y)) and \
                    (x < (xj - xi) * (y - zi) / ((zj - zi) + 1e-12) + xi):
                inside = not inside
            j = i
        return inside

    @staticmethod
    def _validate_init(shape, spacing, angle):
        if not isinstance(shape, ClosedCurveGeo):
            raise TypeError(
                f'"shape" must be a ClosedCurveGeo, got {shape!r}.'
            )

        if not isinstance(spacing, (int, float)):
            raise TypeError(f'"spacing" must be a number, got {spacing!r}.')

        if spacing <= 0:
            raise ValueError('"spacing" must be positive.')

        if not isinstance(angle, (int, float)):
            raise TypeError(f'"angle" must be a number, got {angle!r}.')

    @property
    def shape(self):
        return self._shape

    @property
    def spacing(self):
        return self._spacing

    @property
    def angle(self):
        return self._angle

    def __repr__(self):
        return (
            f"HatchGeo("
            f"shape={self._shape} "
            f"spacing={self._spacing} "
            f"angle={self._angle} "
            f"line_style={self._line_style}, "
            f"Transform={self._transform})"
        )
