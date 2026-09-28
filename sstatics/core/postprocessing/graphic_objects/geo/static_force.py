from functools import cached_property

import numpy as np

from sstatics.core.postprocessing.graphic_objects.geo.object_geo import \
    ObjectGeo
from sstatics.core.postprocessing.graphic_objects.geo.geometry import \
    OpenCurveGeo
from sstatics.core.postprocessing.graphic_objects.geo.text import TextGeo
from sstatics.core.postprocessing.graphic_objects.utils.utils import \
    round_value
from sstatics.core.postprocessing.graphic_objects.utils.defaults import (
    DEFAULT_STATE_LINE, DEFAULT_STATE_LINE_TEXT
)
from sstatics.core.postprocessing.graphic_objects.geo.system import SystemGeo


class StaticForceGeo(ObjectGeo):
    """
    Schnittkraft-/Momentenfläche als exakte Bezierkurve pro Stababschnitt: 
    Aus Wert und Steigung an beiden Enden werden per Hermite-zu-Bezier-Umrechnung die Kontrollpunkte berechnet. 
    Somit ist eine exakte Darstellung der Kurve (keine Annäherung) für Polynome bis Grad 3 (bei Momentenverlauf bei Trapezstreckenlast) gewährleistet.
    Zusätzlich wird wenn vorhanden die Extremstelle innerhalb des Segments analytisch bestimmt und ausgegeben.
    In TikZ wird die Fläche über \\draw ... controls (c1) and (c2) ... gezeichnet.
    """

    CLASS_STYLES = {
        'line': DEFAULT_STATE_LINE,
        'text': DEFAULT_STATE_LINE_TEXT
    }

    def __init__(
            self,
            segment_data: list[dict],
            global_scale: float,
            decimals: int = 2,
            sig_digits: int | None = None,
            scale_diagram: float = 3.0,
            show_text: bool = True,
            **kwargs
    ):
        """
        segment_data: Liste von dicts, ein Eintrag pro Mesh-Stababschnitt:
            - point_i, point_j: (x, z) Weltkoordinaten der Segmentenden
            - value_i, value_j: Schnittkraftwert an beiden Enden (z.B. M_i, M_j)
            - slope_i, slope_j: Steigung dValue/dx an beiden Enden (Momenten-Ableitung: dM/dx = V)
            - rotation: Neigungswinkel des Stabs in rad, z.B. bar.inclination
        """
        super().__init__(
            origin=(0.0, 0.0), global_scale=global_scale, **kwargs
        )
        self._segment_data = segment_data
        self._decimals = decimals
        self._sig_digits = sig_digits
        self._scale_diagram = scale_diagram
        self._show_text = show_text

    BEZIER_VALUE_COLUMN = {'moment': 2, 'shear': 1, 'normal': 0}        # Spalte in forces_disc aus welcher der Wert stammt
    
    BEZIER_SLOPE_SOURCE = {                                             # Woher wird die Steigung entnommen?
        'moment': ('forces_disc', 1),                                   # Steigung ist die Querkraft
        'shear': ('line_load', 1),                                      # Steigung analytisch aus der Linienlast
        'normal': ('line_load', 0),                                     # Steigung analytisch aus der Linienlast
    }
    
    @classmethod
    def supports_kind(cls, kind):
        """Gibt an first_order.py und ähnlichen Stellen weiter, ob für diese Ergebnisart eine Bezierkurven-Darstellung möglich ist."""
        return kind in cls.BEZIER_VALUE_COLUMN
    
    @staticmethod
    def _hermite_extremum(p0, p1, m0, m1):
        """
        Gleichungssystem aus der Bedingung, dass die Ableitung der Hermitekurve null ist. (A*t^2 + B*t + C = 0)
        Sucht t in (0,1) mit horizontaler Tangente der Hermitekurve.
        Dabei sind p0, p1 die Funktionswerte an den Enden und m0, m1 sind die skalierten Tangenten (Steigung * Segmentlänge).
        """
        a = 6 * (p0 - p1) + 3 * (m0 + m1)
        b = 6 * (p1 - p0) - 4 * m0 - 2 * m1
        c = m0

        roots = []
        if abs(a) < 1e-12:
            if abs(b) > 1e-12:
                roots = [-c / b]
        else:
            disc = b ** 2 - 4 * a * c
            if disc >= 0:
                sq = disc ** 0.5
                roots = [(-b + sq) / (2 * a), (-b - sq) / (2 * a)]

        valid = [t for t in roots if 0.0 < t < 1.0]
        return valid[0] if valid else None

    @staticmethod
    def _hermite_eval(t, p0, p1, m0, m1):
        """Wertet die Hermitekurve an der Stelle t aus (Standardbasis)."""
        h00 = 2 * t**3 - 3 * t**2 + 1
        h10 = t**3 - 2 * t**2 + t
        h01 = -2 * t**3 + 3 * t**2
        h11 = t**3 - t**2
        return h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1

    @classmethod
    def from_system(
            cls, system, diff, kind, bar_mesh_type='bars',
            decimals=2, sig_digits=None, scale_diagram=3.0, show_text=True
    ):
        """Baut aus System und Differentialgleichungs-Ergebnissen direkt ein Tuple (sys_geo, sf_geo) und kapselt alles, was für den Befehl der Ausgabe solution.plot erforderlich ist."""
        value_col = cls.BEZIER_VALUE_COLUMN[kind]
        source, idx = cls.BEZIER_SLOPE_SOURCE[kind]
        sys_geo = SystemGeo(system, mesh_type=bar_mesh_type)

        segment_data = []
        for bar, diff_i in zip(system.mesh, diff):
            forces = diff_i.forces_disc
            value_i = forces[0, value_col]
            value_j = forces[-1, value_col]

            if source == 'forces_disc':                 # Querkraft als Steigung
                slope_i = forces[0, idx]
                slope_j = forces[-1, idx]
            else:                                       # 'lineload' (analytisch aus der Streckenlast)
                slope_i = -bar.line_load[idx][0]
                slope_j = -bar.line_load[idx + 3][0]
                
            xi, zi = bar.node_i.x, bar.node_i.z
            xj, zj = bar.node_j.x, bar.node_j.z
            length = ((xj - xi) ** 2 + (zj - zi) ** 2) ** 0.5
                
            m0, m1 = slope_i * length, slope_j * length
            t_star = cls._hermite_extremum(value_i, value_j, m0, m1)    # Extremstelle berechnen
            extremum = None
            if t_star is not None:
                val_star = cls._hermite_eval(t_star, value_i, value_j, m0, m1)
                extremum = {'x': t_star * length, 'value': val_star}

            segment_data.append({
                'point_i': (bar.node_i.x, bar.node_i.z),'point_j': (bar.node_j.x, bar.node_j.z),
                'value_i': value_i, 'value_j': value_j,
                'slope_i': slope_i, 'slope_j': slope_j,
                'rotation': bar.inclination,
                'extremum': extremum,
            })

        sf_geo = cls(
            segment_data, global_scale=sys_geo.global_scale,
            decimals=decimals, sig_digits=sig_digits,
            scale_diagram=scale_diagram, show_text=show_text
        )
        return sys_geo, sf_geo
    
    @cached_property
    def _max_value(self):
        values = [
            v for seg in self._segment_data
            for v in (seg['value_i'], seg['value_j'])
        ]
        # Wenn Moment an Rand = 0:  Steigungen mit berücksichtigen 
        slope_contributions = [
            abs(seg[skey]) * self._segment_length(seg)      # auf Funktionswert umgerechnet, damit ein Vergleich überhaupt sinnvoll ist -> Steigung * Segmentlänge
            for seg in self._segment_data
            for skey in ('slope_i', 'slope_j')
        ]
        max_val = max(
            [abs(v) for v in values] + slope_contributions,
            default=0.0
        )
        return 1e-6 if np.isclose(max_val, 0) else max_val

    @cached_property
    def _scale_factor(self):
        return self._base_scale / self._max_value * self._scale_diagram

    @staticmethod   
    def _segment_length(seg):
        """Zentrale Längenberechnung, damit sowohl in graphic_elements als auch in text_elements keine Redundanz entsteht."""
        xi, zi = seg['point_i']
        xj, zj = seg['point_j']
        return float(np.hypot(xj - xi, zj - zi))

    def _segment_style(self, seg, index):               # 
        return {
            **self._line_style,
            'element_type': 'static_force',
            'sf_pivot': seg['point_i'],
            'sf_rotation_deg': float(np.degrees(seg['rotation'])),
            'sf_length': self._segment_length(seg),
            'sf_index': index,
            'sf_value_i': seg['value_i'],
            'sf_value_j': seg['value_j'],
            'sf_slope_i': seg['slope_i'],
            'sf_slope_j': seg['slope_j'],
            'sf_labels':(
                self._segment_label_data(seg) if self._show_text else []
            ),
        }
    
    def _segment_label_data(self, seg):                            # neu
        """Liefert je Beschriftungspunkt ein Tupel (Position entlang Stab, roher Wert, gerundeter Anzeigetext).
        Wird von TikzRenderer genutzt, um die Werte direkt in der \\scope-Umgebung der zugehörigen Fläche auszugeben."""
        length = self._segment_length(seg)
        labels = [
            (0.0, seg['value_i'], str(round_value(
                seg['value_i'], self._decimals, self._sig_digits
            ))),
            (length, seg['value_j'], str(round_value(
                seg['value_j'], self._decimals, self._sig_digits
            ))),
        ]
        extremum = seg.get('extremum')
        if extremum is not None:
            labels.append((
                extremum['x'], extremum['value'], str(round_value(
                    extremum['value'], self._decimals, self._sig_digits
                ))
            ))
        return labels

    @cached_property
    def graphic_elements(self):
        elements = []
        for i, seg in enumerate(self._segment_data):                # i als Index
            xi, zi = seg['point_i']
            xj, zj = seg['point_j']
            elements.append(OpenCurveGeo(
                [xi, xj], [zi, zj], line_style=self._segment_style(seg, i)
            ))
        return elements

    def _segment_text_elements(self, seg):
        """Ein TextGeo pro Stabende, dabei rückt das linke Ende nach unten rechts ein und das rechte Ende (j) nach unten links. 
        Somit werden die Textwerte an den Stabenden automatisch so verschoben, dass eine Kollision vermieden wird."""
        length = self._segment_length(seg)
        xi, zi = seg['point_i']
        k = self._scale_factor

        endpoints = (
            ('value_i', 0.0, 'below right'),
            ('value_j', length, 'below left'),
        )
        texts = [
            TextGeo(
                self._origin,
                insertion_points=[(pos, seg[key] * k)],
                texts=[str(round_value(
                    seg[key], self._decimals, self._sig_digits
                ))],
                rotation=seg['rotation'], post_translation=(xi, zi),
                text_style={**self._text_style, 'anchor': anchor, 'element_type': 'static_force'}
            )
            for key, pos, anchor in endpoints
        ]

        extremum = seg.get('extremum')
        if extremum is not None:
            texts.append(TextGeo(
                self._origin,
                insertion_points=[(extremum['x'], extremum['value'] * k)],
                texts=[str(round_value(
                    extremum['value'], self._decimals, self._sig_digits
                ))],
                rotation=seg['rotation'], post_translation=(xi, zi),
                text_style={**self._text_style, 'anchor': 'below', 'element_type': 'static_force'}
            ))
        return texts
    
    @cached_property
    def text_elements(self):
        if not self._show_text:
            return []
        return [
            text_geo
            for seg in self._segment_data
            for text_geo in self._segment_text_elements(seg)
        ]

    def __repr__(self):
        return f'{self.__class__.__name__}(segment_data={self._segment_data})'