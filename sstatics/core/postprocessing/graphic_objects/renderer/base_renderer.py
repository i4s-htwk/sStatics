
from abc import ABC, abstractmethod

import numpy as np

from .convert import convert_style
from ..geo.object_geo import ObjectGeo
from ..utils.defaults import PLOTLY, MPL, DEFAULT_NUMBER_OF_TEXT_RINGS, \
    DEFAULT_NUMBER_OF_TEXT_POSITIONS


class AbstractRenderer(ABC):

    _show_grid: bool
    _show_axis: bool

    def __init__(self, mode: str):
        self._validate(mode)
        self._mode = mode
        self._all_graphic_elements = []
        self._placed_text_bboxes = []
        self._scene_boundaries = None  # später automatisch berechnet
        self._scene_scale = None

    @abstractmethod
    def _layout(self):
        pass

    def add_objects(self, *obj: ObjectGeo):
        self._all_graphic_elements.clear()
        for o in obj:
            for x, z, style in self._iter_graphic_elements(o):
                self._all_graphic_elements.append((x, z, style))
                style = convert_style(style, self._mode)
                self.add_graphic(x, z, **style)

        self._update_scene_metrics()

        for o in obj:
            for (
                    x, z, text, style, preferred, rotation
            ) in self._iter_text_elements(o):
                x, z = self._find_optimal_text_position(
                    x, z, text, preferred, rotation
                )
                style = convert_style(style, self._mode)

                rotation = rotation if rotation else 0
                self.add_text(x, z, text, np.rad2deg(rotation), **style)

                tw, th = self._estimate_text_bbox(text)
                self._placed_text_bboxes.append((
                    x - tw / 2, x + tw / 2,  # xmin, xmax
                    z - th / 2, z + th / 2  # zmin, zmax
                ))

    @abstractmethod
    def add_graphic(self, x, z, **style):
        pass

    @abstractmethod
    def add_text(self, x, z, text, rotation, **style):
        pass

    @abstractmethod
    def figure(self):
        pass

    @abstractmethod
    def show(self):
        pass

    def _update_scene_metrics(self):
        """Berechnet Grenzen und Skalierung anhand aller gesammelten
        Graphic-Elements."""
        x_all, z_all = [], []
        for x_list, z_list, _ in self._all_graphic_elements:
            if x_list is None or z_list is None:
                continue
            if len(x_list) == 0 or len(z_list) == 0:
                continue
            x_all.extend(x for x in x_list if x is not None)
            z_all.extend(z for z in z_list if z is not None)

        if not x_all or not z_all:
            self._scene_boundaries = (0.0, 0.0, 0.0, 0.0)
            self._scene_scale = 0.05
            return

        x_min, x_max = min(x_all), max(x_all)
        z_min, z_max = min(z_all), max(z_all)
        self._scene_boundaries = (x_min, x_max, z_min, z_max)
        self._scene_scale = 0.021 * max(x_max - x_min, 2*(z_max - z_min))

    @property
    def scene_boundaries(self):
        """Gibt die aktuellen Szenen-Grenzen zurück."""
        if self._scene_boundaries is None:
            self._update_scene_metrics()
        return self._scene_boundaries

    @property
    def scene_scale(self):
        """Gibt den globalen Skalierungsfaktor der Szene zurück."""
        if self._scene_scale is None:
            self._update_scene_metrics()
        return self._scene_scale

    def _iter_graphic_elements(self, obj):
        for element in obj.graphic_elements:
            if hasattr(element, 'graphic_elements'):
                for x, z, style in self._iter_graphic_elements(element):
                    x, z = obj.transform(x, z)
                    yield x, z, style
            else:
                x, z, style = element
                x, z = obj.transform(x, z)
                yield x, z, style

    def _iter_text_elements(self, obj):
        for element in getattr(obj, 'text_elements', []):
            if not (4 <= len(element) <= 6):
                raise ValueError(
                    'Each element in text_elements must be a tuple of '
                    'length 4, 5 or 6.'
                )

            # unpack gracefully
            if len(element) == 4:
                x, z, text, style = element
                preferred_pos = None
                rotation = None
            elif len(element) == 5:
                x, z, text, style, preferred_pos = element
                rotation = None
            else:  # len == 6
                x, z, text, style, preferred_pos, rotation = element

            if text != ['']:
                x, z = obj.transform(x, z)
                yield x, z, text, style, preferred_pos, rotation

        # Rekursion bleibt gleich:
        for sub in getattr(obj, 'graphic_elements', []):
            if (hasattr(sub, 'text_elements')
                    or hasattr(sub, 'graphic_elements')):
                for (
                        x, z, text, style, preferred_pos, rotation
                ) in self._iter_text_elements(sub):
                    x, z = obj.transform(x, z)
                    yield x, z, text, style, preferred_pos, rotation

    def _find_optimal_text_position(self, x, z, text, preferred_pos, rotation):
        if preferred_pos is not None and preferred_pos.endswith('!'):
            variable_mode = False
            preferred_pos = preferred_pos.rstrip("!")
            n, m = 4, 2
        else:
            variable_mode = True
            n = DEFAULT_NUMBER_OF_TEXT_POSITIONS
            m = DEFAULT_NUMBER_OF_TEXT_RINGS

        offsets = (1 + 0.5 * np.arange(m)) * self.scene_scale
        angles = np.linspace(np.pi / 2, -3 / 2 * np.pi, n, endpoint=False)
        R, A = np.meshgrid(offsets, angles, indexing="ij")

        A_rot = A + (rotation if rotation else 0)
        x_try = x + R * np.cos(A_rot)
        z_try = z - R * np.sin(A_rot)

        ring_indices = np.arange(len(offsets))
        pos_indices = np.arange(len(angles))
        RI, PI = np.meshgrid(ring_indices, pos_indices, indexing="ij")

        positions = [
            (f"{RI.flatten()[i]},{PI.flatten()[i]}",
             (x_try.flatten()[i], z_try.flatten()[i]))
            for i in range(n * m)
        ]

        if preferred_pos is not None:
            try:
                idx = next(
                    i for i, p in enumerate(positions) if p[0] == preferred_pos
                )
            except StopIteration:
                raise ValueError('Value for "preferred_pos" is invalid.')

            if not variable_mode:
                _, (x_force, z_force) = positions[idx]
                return x_force, z_force
            positions = positions[idx:] + positions[:idx]

        for pos_code, (x_try, z_try) in positions:
            if not self._text_collision(x_try, z_try, text):
                return x_try, z_try

        return positions[0][1]

    def _estimate_text_bbox(self, text):
        """Schätzt die Text-Bounding-Box basierend auf Textlänge und Szene."""
        scene_scale = getattr(self, "scene_scale", 0.05)
        # text = getattr(obj, "_text", "")
        if not text:
            return scene_scale * 0.6, scene_scale * 0.4

        text_width = len(str(text)) * 0.6 * scene_scale
        text_height = 0.4 * scene_scale
        return text_width, text_height

    def _text_collision(self, x, z, text, margin=0.01):
        """Prüft, ob der Text (als Rechteck) mit Linien kollidiert."""
        text_width, text_height = self._estimate_text_bbox(text)
        # Bestimme Textrechteck
        x_min, x_max = x - text_width / 2, x + text_width / 2
        z_min, z_max = z - text_height / 2, z + text_height / 2

        for px_list, pz_list, _ in self._all_graphic_elements:
            coords = [(xi, zi) for xi, zi in zip(px_list, pz_list)
                      if xi is not None and zi is not None]
            if len(coords) < 2:
                continue
            for (x0, z0), (x1, z1) in zip(coords[:-1], coords[1:]):
                # Wenn die Linie das Rechteck schneidet → Kollision
                if self._line_intersects_rect(
                        x0, z0, x1, z1,
                        x_min - margin, x_max + margin,
                        z_min - margin, z_max + margin
                ):
                    return True

        for (
                xmin_old, xmax_old, zmin_old, zmax_old
        ) in self._placed_text_bboxes:
            if not (
                    x_max < xmin_old or x_min > xmax_old or
                    z_max < zmin_old or z_min > zmax_old
            ):
                return True

        return False

    # TODO: Berechnung funktioniert nicht + Flächenberechnung einbauen
    @staticmethod
    def _line_intersects_rect(x0, z0, x1, z1, xmin, xmax, zmin, zmax):
        """Prüft, ob eine Linie das Rechteck schneidet oder darin liegt."""
        # Fall 1: Beide Punkte komplett außerhalb in derselben Richtung →
        # kein Schnitt
        if (x0 < xmin and x1 < xmin) or (x0 > xmax and x1 > xmax) or \
                (z0 < zmin and z1 < zmin) or (z0 > zmax and z1 > zmax):
            return False

        # Fall 2: Einer der Punkte liegt im Rechteck → Schnitt
        if xmin <= x0 <= xmax and zmin <= z0 <= zmax:
            return True
        if xmin <= x1 <= xmax and zmin <= z1 <= zmax:
            return True

        # Fall 3: Prüfe Schnitt mit jeder Rechteckkante
        def line_intersect(xa, za, xb, zb, xc, zc, xd, zd):
            """Hilfsfunktion: prüft Schnitt zweier Segmente."""

            def ccw(x1, z1, x2, z2, x3, z3):
                return (z3 - z1) * (x2 - x1) > (z2 - z1) * (x3 - x1)

            return (ccw(xa, za, xc, zc, xd, zd) != ccw(xb, zb, xc, zc, xd,
                                                       zd)) and \
                (ccw(xa, za, xb, zb, xc, zc) != ccw(xa, za, xb, zb, xd, zd))

        rect_edges = [
            (xmin, zmin, xmax, zmin),  # unten
            (xmax, zmin, xmax, zmax),  # rechts
            (xmax, zmax, xmin, zmax),  # oben
            (xmin, zmax, xmin, zmin)  # links
        ]
        for (xa, za, xb, zb) in rect_edges:
            if line_intersect(x0, z0, x1, z1, xa, za, xb, zb):
                return True

        return False

    @staticmethod
    def _segments_intersect(x1, y1, x2, y2, x3, y3, x4, y4):
        """Standard-Segment-Schnitt-Test (orientiert an CCW-Test)."""

        def ccw(ax, ay, bx, by, cx, cy):
            return (cy - ay) * (bx - ax) > (by - ay) * (cx - ax)

        return (
                ccw(x1, y1, x3, y3, x4, y4) != ccw(x2, y2, x3, y3, x4, y4)
                and ccw(x1, y1, x2, y2, x3, y3) != ccw(x1, y1, x2, y2, x4, y4)
        )

    @staticmethod
    def _point_to_segment_distance(px, pz, x0, z0, x1, z1):
        """Kürzester Abstand zwischen Punkt und Liniensegment."""
        dx, dz = px - x0, pz - z0
        sx, sz = x1 - x0, z1 - z0
        seg_len_sq = sx ** 2 + sz ** 2
        if seg_len_sq == 0:
            return np.hypot(dx, dz)
        t = max(0, min(1, (dx * sx + dz * sz) / seg_len_sq))
        closest_x = x0 + t * sx
        closest_z = z0 + t * sz
        return np.hypot(px - closest_x, pz - closest_z)

    @staticmethod
    def _validate(mode):
        if not isinstance(mode, str):
            raise TypeError(
                f'mode must be a string, got {type(mode).__name__!r}'
            )
        if mode not in (PLOTLY, MPL):
            raise ValueError(
                f'Invalid mode {mode!r}. Expected one of: {PLOTLY!r}, {MPL!r}.'
            )

    @property
    def mode(self):
        return self._mode

    @property
    def all_graphic_elements(self):
        return self._all_graphic_elements

    @property
    def show_grid(self):
        return self._show_grid

    @property
    def show_axis(self):
        return self._show_axis
