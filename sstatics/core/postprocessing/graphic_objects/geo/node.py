
from functools import cached_property

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
            show_load_text: bool = True,
            **kwargs
    ):
        self._validate_node(node, show_load_text)
        super().__init__(origin=(node.x, node.z), **kwargs)
        self._node = node
        self._show_load_text = show_load_text

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

        return [support(
            self._origin, text=self._text,
            line_style=self._resolve_style(
                self._node, DEFAULT_SUPPORT, self._line_style
            ),
            text_style=self._resolve_style(
                self._node, DEFAULT_TEXT, self._text_style
            ),
            rotation=rotation
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
        return [
            PointLoadGeo(
                self._origin, load=load, show_text=self._show_load_text,
                line_style=self._resolve_style(
                    load, DEFAULT_LINE, self._line_style
                ),
                text_style=self._resolve_style(
                    load, DEFAULT_TEXT, self._text_style
                ),
            ) for load in self._node.loads
        ]

    @property
    def _displacement_elements(self):
        return [
            DisplacementGeo(
                self._origin, displacement=displacement,
                show_text=self._show_load_text,
                line_style=self._resolve_style(
                    displacement, DEFAULT_LINE, self._line_style
                ),
                text_style=self._resolve_style(
                    displacement, DEFAULT_TEXT, self._text_style
                ),
            ) for displacement in self._node.displacements
        ]

    @staticmethod
    def _validate_node(node, show_load_text):
        if not isinstance(node, Node):
            raise TypeError(
                f'"node" must be a Node, got {type(node).__name__!r}'
            )

        if not isinstance(show_load_text, bool):
            raise TypeError(
                f'"show_load_text" must be a boolean, got '
                f'{type(show_load_text).__name__!r}'
            )

    @property
    def node(self):
        return self._node

    @property
    def show_load_text(self):
        return self._show_load_text

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'origin={self._origin}, '
            f'node={self._node}, '
            f'show_load_text={self._show_load_text}, '
            f'line_style={self._line_style}, '
            f'text_style={self._text_style}, '
            f'Transform={self._transform})'
        )
