from types import NoneType
from typing import Literal

from functools import cached_property

from sstatics.core.postprocessing.graphic_objects.utils.defaults import \
    DEFAULT_BAR, DEFAULT_CIRCLE_TEXT, DEFAULT_LINE, DEFAULT_TEXT
from sstatics.core.preprocessing.system import System

from sstatics.core.postprocessing.graphic_objects.geo.object_geo import \
    ObjectGeo
from sstatics.core.postprocessing.graphic_objects.geo.bar import BarGeo
from sstatics.core.postprocessing.graphic_objects.geo.node import NodeGeo


class SystemGeo(ObjectGeo):

    def __init__(
            self,
            system: System,
            mesh_type: Literal['bars', 'user_mesh', 'mesh'] = 'bars',
            show_load: bool = True,
            show_bar_text: bool = False,
            show_node_text: bool = True,
            show_load_text: bool = True,
            show_full_hinge: bool = True,
            decimals: int = 2,
            sig_digits: int | None = None,
            **kwargs
    ):
        self._validate_system(
            system, mesh_type, show_load, show_bar_text, show_node_text,
            show_load_text, decimals, sig_digits
        )
        self._system = system
        self._mesh_type = mesh_type
        self._show_load = show_load
        self._show_bar_text = show_bar_text
        self._show_node_text = show_node_text
        self._show_load_text = show_load_text
        self._show_full_hinge = show_full_hinge
        self._decimals = decimals
        self._sig_digits = sig_digits
        super().__init__(
            origin=(system.bars[0].node_i.x, system.bars[0].node_i.z), **kwargs
        )

    @cached_property
    def graphic_elements(self):
        return [
            *self._bar_elements,
            *self._node_elements
        ]

    @cached_property
    def text_elements(self):
        return []

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
    def _bar_elements(self):
        return [
            BarGeo(
                bar, show_load=self._show_load,
                show_load_text=self._show_load_text,
                text=(i + 1) if self._show_bar_text else '',
                decimals=self._decimals, sig_digits=self._sig_digits,
                line_style=self._resolve_style(
                    bar, DEFAULT_BAR, self._line_style
                ),
                text_style=self._resolve_style(
                    bar, DEFAULT_CIRCLE_TEXT, self._text_style
                ),
                global_scale=self._base_scale
            ) for i, bar in enumerate(getattr(self._system, self._mesh_type))
        ]

    @property
    def _node_elements(self):
        return [
            NodeGeo(
                node, show_load=self._show_load,
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

    @staticmethod
    def _validate_system(
            system, mesh_type, show_load, show_bar_text, show_node_text,
            show_load_text, decimals, sig_digits
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
    def decimals(self):
        return self._decimals

    @property
    def sig_digits(self):
        return self._sig_digits

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
            f'decimals={self._decimals}, '
            f'sig_digits={self._sig_digits}, '
            f'line_style={self._line_style}, '
            f'text_style={self._text_style}, '
            f'Transform={self._transform})'
        )
