
from functools import cached_property
from types import NoneType

import numpy as np

from sstatics.core.preprocessing.node import Node

from sstatics.core.postprocessing.graphic_objects.utils.defaults import (
    DEFAULT_SUPPORT, DEFAULT_LINE, DEFAULT_TEXT
)
from sstatics.core.postprocessing.graphic_objects.geo.object_geo import \
    ObjectGeo
from sstatics.core.postprocessing.graphic_objects.geo.constraint import (
    RollerSupportGeo, PinnedSupportGeo, FixedSupportUWGeo, FixedSupportUPhiGeo,
    FixedSupportWPhiGeo, ClampedSupportGeo, FreeNodeGeo,
    TranslationalSpringGeo, TorsionalSpringGeo
)
from sstatics.core.postprocessing.graphic_objects.geo.effect import (
    DisplacementGeo, PointLoadGeo
)


class NodeGeo(ObjectGeo):

    def __init__(
            self,
            node: Node,
            load_distances: dict | None = None,
            show_load: bool = True,
            show_load_text: bool = True,
            decimals: int = 2,
            sig_digits: int | None = None,
            **kwargs
    ):
        self._validate_node(
            node, load_distances, show_load, show_load_text, decimals,
            sig_digits
        )
        super().__init__(origin=(node.x, node.z), **kwargs)
        self._node = node
        self._load_distances = load_distances
        self._show_load = show_load
        self._show_load_text = show_load_text
        self._decimals = decimals
        self._sig_digits = sig_digits

    @cached_property
    def graphic_elements(self):
        return [
            *self._select_support,
            *self._select_spring,
            *self._load_elements,
            *self._displacement_elements
        ]

    @cached_property
    def text_elements(self):
        return []

    @property
    def _select_support(self):
        bits = self._support_bits
        rotation = self._node.rotation

        support_classes = {
            '100': RollerSupportGeo,
            '010': RollerSupportGeo,
            '001': PinnedSupportGeo,
            '110': FixedSupportUWGeo,
            '101': FixedSupportUPhiGeo,
            '011': FixedSupportWPhiGeo,
            '111': ClampedSupportGeo,
        }

        support = support_classes.get(bits, FreeNodeGeo)
        if support is FreeNodeGeo:
            return [support(
                self._origin, text=self._text, point_style=self._point_style
            )]
        if support is RollerSupportGeo and bits == '100':
            rotation -= np.pi / 2

        line_style = self._resolve_style(
            self._node, DEFAULT_SUPPORT, self._line_style
        )
        text_style = self._resolve_style(
            self._node, DEFAULT_TEXT, self._text_style
        )

        return [support(
            self._origin, text=self._text, line_style=line_style,
            text_style=text_style, rotation=rotation
        )]

    @property
    def _support_bits(self):
        return ''.join(
            str(int(s == 'fixed'))
            for s in (self._node.u, self._node.w, self._node.phi)
        )

    @property
    def _select_spring(self):
        elements = []
        springs = [
            (self._node.u, TranslationalSpringGeo, -np.pi / 2),
            (self._node.w, TranslationalSpringGeo, 0),
            (self._node.phi, TorsionalSpringGeo, 0),
        ]

        for val, cls, rot in springs:
            if isinstance(val, (int, float)):
                elements.append(cls(
                    self._origin,
                    line_style=self._resolve_style(
                        self._node, DEFAULT_SUPPORT, self._line_style
                    ),
                    rotation=rot+self._node.rotation
                ))
        return elements

    @property
    def _load_elements(self):
        scale = self._transform.scaling
        return [
            PointLoadGeo(
                self._origin, load=load,
                distance=(
                        self._load_distances[load] / scale if
                        self._load_distances and load in self._load_distances
                        else None
                ),
                show_text=self._show_load_text,
                decimals=self._decimals, sig_digits=self._sig_digits,
                line_style=self._resolve_style(
                    load, DEFAULT_LINE, self._line_style
                ),
                text_style=self._resolve_style(
                    load, DEFAULT_TEXT, self._text_style
                ),
            ) for load in self._node.loads
        ] if self._show_load else []

    @property
    def _displacement_elements(self):
        scale = self._transform.scaling
        return [
            DisplacementGeo(
                self._origin, displacement=displacement,
                distance=(
                        self._load_distances[displacement] / scale
                        if (
                                self._load_distances
                                and displacement in self._load_distances
                        )
                        else None
                ),
                show_text=self._show_load_text,
                decimals=self._decimals, sig_digits=self._sig_digits,
                line_style=self._resolve_style(
                    displacement, DEFAULT_LINE, self._line_style
                ),
                text_style=self._resolve_style(
                    displacement, DEFAULT_TEXT, self._text_style
                ),
            ) for displacement in self._node.displacements
        ] if self._show_load else []

    @staticmethod
    def _validate_node(
            node, load_distances, show_load, show_load_text, decimals,
            sig_digits
    ):
        if not isinstance(node, Node):
            raise TypeError(
                f'"node" must be a Node, got {type(node).__name__!r}'
            )

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

        if not isinstance(show_load, bool):
            raise TypeError(
                f'"show_load" must be a boolean, got '
                f'{type(show_load).__name__!r}'
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
    def node(self):
        return self._node

    @property
    def show_load(self):
        return self._show_load

    @property
    def load_distances(self):
        return self._load_distances

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
            f'node={self._node}, '
            f'load_distances={self._load_distances}, '
            f'show_load={self._show_load}, '
            f'show_load_text={self._show_load_text}, '
            f'decimals={self._decimals}, '
            f'sig_digits={self._sig_digits}, '
            f'line_style={self._line_style}, '
            f'text_style={self._text_style}, '
            f'Transform={self._transform})'
        )
