from types import NoneType
from typing import Literal

from functools import cached_property

from sstatics.core.preprocessing import Bar, Node, System

from sstatics.core.postprocessing.graphic_objects.geo.dimensioning import DimensioningGeo, \
    DimensioningGeo                                                                   #neu      

from sstatics.core.postprocessing.graphic_objects.utils.defaults import \
    DEFAULT_BAR, DEFAULT_CIRCLE_TEXT, DEFAULT_LINE, DEFAULT_TEXT
from sstatics.core.postprocessing.graphic_objects.geo.object_geo import \
    ObjectGeo
from sstatics.core.postprocessing.graphic_objects.geo.bar import BarGeo
from sstatics.core.postprocessing.graphic_objects.geo.node import NodeGeo
from sstatics.core.postprocessing.graphic_objects.geo.hinge import (
    FullMomentHingeGeo
)


class SystemGeo(ObjectGeo):

    def __init__(
            self,
            system: System,
            mesh_type: Literal['bars', 'user_mesh', 'mesh'] = 'bars',
            show_load: bool = True,
            show_bar_text: bool = True,
            show_node_text: bool = True,
            show_load_text: bool = True,
            show_tensile_zone: dict[Bar: bool] | None = None,
            show_full_hinges: bool = True,
            show_dimensions: bool = True,           # ein- und ausschalten der Bemaßung
            decimals: int = 2,
            sig_digits: int | None = None,
            load_distances: dict | None = None,
            dimensions: list[tuple] | None = None,
            dimension_gap: float = 1.0,             # Abstand der Bemaßungslinien von den Objekten
            **kwargs
    ):
        self._validate_system(
            system, mesh_type, show_load, show_bar_text, show_node_text,
            show_load_text, show_full_hinges, show_tensile_zone, show_dimensions,
            decimals, sig_digits, load_distances
        )
        self._system = system
        self._mesh_type = mesh_type
        self._show_load = show_load
        self._show_bar_text = show_bar_text
        self._show_node_text = show_node_text
        self._show_load_text = show_load_text
        self._show_full_hinges = show_full_hinges
        self._show_tensile_zone = show_tensile_zone or {}
        self._show_dimensions = show_dimensions
        self._decimals = decimals
        self._sig_digits = sig_digits
        self._load_distances = load_distances
        self._dimensions = dimensions                   # neu
        self._dimension_gap = dimension_gap             # neu
        super().__init__(
            origin=(system.bars[0].node_i.x, system.bars[0].node_i.z), **kwargs
        )

    @cached_property
    def graphic_elements(self):
        return [
            *self._bar_elements,
            *self._node_elements,
            *self._full_hinge_elements,
            *self._dimensioning_elements          # neu hinzugefügt
        ]

    @staticmethod                       # für die Auflösung von Punkten (Node-Objekte oder rohe Tupel)
    def _resolve_point(point):
        """Erlaubt sowohl rohe (x, z)-Tupel als auch Node-Objekte als Bemaßungs-Endpunkte, damit man z.B. direkt node_1, node_2 statt (node_1.x, node_1.z) schreiben kann."""
        if hasattr(point, 'x') and hasattr(point, 'z'):
            return point.x, point.z
        return point

    @property                           # Liefert die grafischen Elemente für die Bemaßung
    def _dimensioning_elements(self):
        if not self._show_dimensions:
            return []
        
        specs = (
            self._auto_dimension_specs if self._dimensions is None    # None führt zu automatisch generieren
            else self._dimensions
        )
        
        elements = []
        for dim in specs:
            point_i, point_j, distance = dim[0], dim[1], dim[2]
            measure = dim[3] if len(dim) > 3 else None
            dim_type = dim[4] if len(dim) > 4 else None
            elements.append(DimensioningGeo(
                self._resolve_point(point_i), self._resolve_point(point_j),
                distance, measure=measure, dim_type=dim_type,
                line_style=self._line_style, text_style=self._text_style
            ))
        return elements
    
    @property
    def _auto_dimension_specs(self):
        """Erzeugt automatisch eine Bemaßungskette: 
        alle eindeutigen x-Koordinaten der Knoten (horizontal, unterhalb des Systems)
        alle eindeutigen z-Koordinaten (vertikal, links vom System)
        Jeweils als Kette aufeinanderfolgender Abschnitte.
        Keine schrägen Bemaßungen möglich.
        Wird nur benutzt, wenn "dimensions" nicht explizit gesetzt wurde (automatisch generiert, wenn None)."""
        xs = sorted({round(node.x, 4) for node in self._nodes})
        zs = sorted({round(node.z, 4) for node in self._nodes})

        gap = self._dimension_gap
        x_min = min(xs) if xs else 0.0
        z_max = max(zs) if zs else 0.0
        h_line = z_max + gap     # eine gemeinsame Linie UNTERHALB des ganzen Systems
        v_line = x_min - gap     # eine gemeinsame Linie LINKS vom ganzen System

        specs = []
        for x0, x1 in zip(xs, xs[1:]):
            specs.append(((x0, 0.0), (x1, 0.0), h_line, None, 1))
        for z0, z1 in zip(zs, zs[1:]):
            specs.append(((0.0, z0), (0.0, z1), v_line, None, 2))
        return specs

    @cached_property
    def text_elements(self):
        return []

    @property
    def _raw_graphic_elements(self):
        x, z = [], []
        for node in self._nodes:
            x.append(node.x)
            z.append(node.z)
        return [(x, z, self._line_style)]

    @property
    def _nodes(self):
        return self._system.nodes(mesh_type=self._mesh_type)

    @property
    def _bars(self):
        return getattr(self._system, self._mesh_type)

    @property
    def _max_line_load_value(self):
        return max(
            (
                max((abs(load.pi), abs(load.pj)))
                for bar in self._system.bars for load in bar.line_loads
            ),
            default=None
        )

    def _count_bars_and_hinges(self, node: Node):
        m, n = 0, 0
        for bar in self._bars:
            if bar.node_i == node:
                n += 1
                if (False, False, True) == bar.hinge[0:3]:
                    m += 1
            elif bar.node_j == node:
                n += 1
                if (False, False, True) == bar.hinge[3:6]:
                    m += 1
        return m, n

    @property
    def _find_nodes_with_full_hinges(self):
        if not self._show_full_hinges:
            return []

        nodes_with_full_hinge = []
        for node in self._nodes:
            m, n = self._count_bars_and_hinges(node)
            if (
                    (m + 1) == n
                    and n > 1
                    and node.u == 'free'
                    and node.w == 'free'
                    and node.phi == 'free'
            ):
                nodes_with_full_hinge.append(node)
        return nodes_with_full_hinge

    def _possible_same_load_text_pos(
            self, current_loads, compare_loads, current_i, compare_i
    ):
        pos_per_load = [False] * len(current_loads)
        distances = self._load_distances or {}

        for i, current in enumerate(current_loads):
            for compare in compare_loads:
                current_p = current.pi if current_i else current.pj
                compare_p = compare.pi if compare_i else compare.pj
                if (
                        current_p == compare_p and
                        current.direction == compare.direction and
                        current.coord == compare.coord and
                        current.length == compare.length and
                        distances.get(current, None)
                        == distances.get(compare, None)
                ):
                    pos_per_load[i] = True
                    break
        return pos_per_load

    @property
    def _show_line_load_list(self):
        bars = getattr(self._system, self._mesh_type)
        result = []
        checked_pairs = set()

        for current_bar in bars:
            loads = current_bar.line_loads
            show = [(True, True) for _ in loads]

            for compare_bar in bars:
                if current_bar == compare_bar:
                    continue

                if current_bar.inclination != compare_bar.inclination:
                    continue

                pair_key = frozenset({current_bar, compare_bar})
                if pair_key in checked_pairs:
                    continue

                if current_bar.node_i.same_location(compare_bar.node_i):
                    hide = self._possible_same_load_text_pos(
                        loads, compare_bar.line_loads, True, True
                    )
                    show = [
                        (not h and s0, s1)
                        for (s0, s1), h in zip(show, hide)
                    ]

                elif current_bar.node_i.same_location(compare_bar.node_j):
                    hide = self._possible_same_load_text_pos(
                        loads, compare_bar.line_loads, True, False
                    )
                    show = [
                        (not h and s0, s1)
                        for (s0, s1), h in zip(show, hide)
                    ]

                if current_bar.node_j.same_location(compare_bar.node_i):
                    hide = self._possible_same_load_text_pos(
                        loads, compare_bar.line_loads, False, True
                    )
                    show = [
                        (s0, not h and s1)
                        for (s0, s1), h in zip(show, hide)
                    ]

                elif current_bar.node_j.same_location(compare_bar.node_j):
                    hide = self._possible_same_load_text_pos(
                        loads, compare_bar.line_loads, False, False
                    )
                    show = [
                        (s0, not h and s1)
                        for (s0, s1), h in zip(show, hide)
                    ]

                if any(not all(v) for v in show):
                    checked_pairs.add(pair_key)

            result.append(show)
        return result

    @property
    def _show_line_load_values(self):
        if not self._show_load_text:
            return [
                [(False, False) for _ in bar.line_loads] for bar in self._bars
            ]
        return self._show_line_load_list

    @property
    def _bar_elements(self):
        return [
            BarGeo(
                bar, load_distances=self._load_distances,
                global_max_line_load=self._max_line_load_value,
                show_load=self._show_load,
                show_point_load_text=self._show_load_text,
                show_line_load_texts=self._show_line_load_values[i],
                text=(i + 1) if self._show_bar_text else '',
                show_tensile_zone=self._show_tensile_zone.get(bar, True),
                show_full_hinges=(
                    bar.node_i in self._find_nodes_with_full_hinges,
                    bar.node_j in self._find_nodes_with_full_hinges
                ),
                decimals=self._decimals, sig_digits=self._sig_digits,
                line_style=self._resolve_style(
                    bar, DEFAULT_BAR, self._line_style
                ),
                text_style=self._resolve_style(
                    bar, DEFAULT_CIRCLE_TEXT, self._text_style
                ),
                global_scale=self._base_scale
            ) for i, bar in enumerate(self._bars)
        ]

    @property
    def _node_elements(self):
        return [
            NodeGeo(
                node, load_distances=self._load_distances,
                show_load=self._show_load,
                show_load_text=self._show_load_text,
                text=(i + 1) if self._show_node_text else '',
                decimals=self._decimals, sig_digits=self._sig_digits,
                line_style=self._resolve_style(
                    node, DEFAULT_LINE, self._line_style
                ),
                text_style=self._resolve_style(
                    node, DEFAULT_TEXT, self._text_style
                ),
                scaling=self._base_scale
            ) for i, node in enumerate(self._nodes)
        ]

    @property
    def _full_hinge_elements(self):
        return [
            FullMomentHingeGeo((node.x, node.z), scaling=self._base_scale)
            for node in self._find_nodes_with_full_hinges
        ]

    @staticmethod
    def _validate_system(
            system, mesh_type, show_load, show_bar_text, show_node_text,
            show_load_text, show_full_hinges, show_tensile_zone, show_dimensions,       # show_dimensions hinzugefügt
            decimals, sig_digits, load_distances
    ):
        if not isinstance(system, System):
            raise TypeError(
                f'"system" must be a System, got {type(system).__name__!r}'
            )

        if not isinstance(mesh_type, str):
            raise TypeError(
                f'"mesh_type" must be a String, got '
                f'{type(mesh_type).__name__!r}'
            )

        if mesh_type not in ['bars', 'user_mesh', 'mesh']:
            raise ValueError(
                f'"mesh_type" must be "bars", "user_mesh" or "mesh", got '
                f'{mesh_type!r}'
            )

        if not isinstance(show_load, bool):
            raise TypeError(
                f'"show_load" must be a boolean, got '
                f'{type(show_load).__name__!r}'
            )

        if not isinstance(show_bar_text, bool):
            raise TypeError(
                f'"show_bar_text" must be a boolean, got '
                f'{type(show_bar_text).__name__!r}'
            )

        if not isinstance(show_node_text, bool):
            raise TypeError(
                f'"show_node_text" must be a boolean, got '
                f'{type(show_node_text).__name__!r}'
            )

        if not isinstance(show_load_text, bool):
            raise TypeError(
                f'"show_load_text" must be a boolean, got '
                f'{type(show_load_text).__name__!r}'
            )

        if not isinstance(show_tensile_zone, (dict, NoneType)):
            raise TypeError(
                f'"show_tensile_zone" must be a dict, got '
                f'{type(show_tensile_zone).__name__!r}'
            )

        if (
                isinstance(show_tensile_zone, dict)
                and not all(
                    isinstance(k, Bar) and isinstance(v, bool)
                    for k, v in show_tensile_zone.items()
                )
        ):
            raise TypeError(
                '"show_tensile_zone" must be a dict with Bar keys and bool '
                'values: dict[Bar, bool]'
            )

        if not isinstance(show_full_hinges, bool):
            raise TypeError(
                f'"show_full_hinges" must be a boolean, got '
                f'{type(show_full_hinges).__name__!r}'
            )

        if not isinstance(show_dimensions, bool):                          # Überprüfung, ob show_dimensions ein boolescher Wert ist
            raise TypeError(                                
                f'"show_dimensions" must be a boolean, got '
                f'{type(show_dimensions).__name__!r}'       
            )

        if not isinstance(decimals, int):
            raise TypeError(
                f'"decimals" must be int or None, '
                f'got {type(decimals).__name__!r}'
            )

        if not isinstance(sig_digits, (int, NoneType)):
            raise TypeError(
                f'"sig_digits" must be int or None, '
                f'got {type(sig_digits).__name__!r}'
            )

        if sig_digits is not None and sig_digits <= 0:
            raise ValueError('"sig_digits" has to be greater than zero.')

        if not isinstance(load_distances, (dict, NoneType)):
            raise TypeError(
                f'"load_distances" must be dict or None, '
                f'got {type(load_distances).__name__!r}'
            )

        if isinstance(load_distances, dict) and not all(
                isinstance(v, (int, float)) for v in load_distances.values()
        ):
            raise TypeError(
                'all values of load_distances must be int or float'
            )

    @property
    def system(self):
        return self._system

    @property
    def mesh_type(self):
        return self._mesh_type

    @property
    def show_load(self):
        return self._show_load

    @property
    def show_bar_text(self):
        return self._show_bar_text

    @property
    def show_node_text(self):
        return self._show_node_text

    @property
    def show_load_text(self):
        return self._show_load_text

    @property
    def show_tensile_zone(self):
        return self._show_tensile_zone

    @property
    def show_full_hinges(self):
        return self._show_full_hinges

    @property                             # Liefert, ob die Bemaßungen angezeigt werden sollen oder ausgeblendet werden.
    def show_dimensions(self):
        return self._show_dimensions

    @property
    def decimals(self):
        return self._decimals

    @property
    def sig_digits(self):
        return self._sig_digits

    @property
    def load_distances(self):
        return self._load_distances

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'origin={self._origin}, '
            f'system={self._system}, '
            f'mesh_type={self._mesh_type}, '
            f'show_load={self._show_load}, '
            f'show_bar_text={self._show_bar_text}, '
            f'show_node_text={self._show_node_text}, '
            f'show_load_text={self._show_load_text}, '
            f'show_tensile_zone={self._show_tensile_zone}, '
            f'show_full_hinges={self._show_full_hinges}, '
            f'show_dimensions={self._show_dimensions}, '                # neu hinzugefügt
            f'decimals={self._decimals}, '
            f'sig_digits={self._sig_digits}, '
            f'load_distances={self._load_distances}, '
            f'line_style={self._line_style}, '
            f'text_style={self._text_style}, '
            f'Transform={self._transform})'
        )
