
from functools import cached_property

import plotly.express as px
import numpy as np

from sstatics.core.preprocessing.poleplan import Poleplan, Chain, Pole
from sstatics.core.preprocessing.bar import Bar

from sstatics.graphic_objects.utils import SingleGraphicObject
from sstatics.graphic_objects.node import NodeGraphic
from sstatics.graphic_objects.bar import BarGraphic
from sstatics.graphic_objects.geometry import (LineGraphic, PointGraphic,
                                               EllipseGraphic)


class PoleplanGraphic(SingleGraphicObject):

    def __init__(
            self, poleplan: Poleplan,
            base_scale=None, show_annotations: bool = True, **kwargs
    ):
        if not isinstance(poleplan, Poleplan):
            raise TypeError('"poleplan" has to be an instance of Poleplan')
        super().__init__(
            poleplan.system.bars[0].node_i.x,
            poleplan.system.bars[0].node_i.z, **kwargs
        )
        self._max_dim = max(poleplan.system.max_dimensions)
        self.base_scale = base_scale
        self.boundary = poleplan.system.boundary

        self.bars = poleplan.system.bars
        self.nodes = poleplan.system.nodes()

        self.chains = poleplan.chains
        num_chains = len(self.chains)
        colors = px.colors.sample_colorscale(
            'Viridis', [i / (num_chains - 1) for i in range(num_chains)]
        )

        self._apole_lines = []
        self._chains = []
        self._pole = []

        for i, chain in enumerate(self.chains):
            self._chains.append(
                ChainGraphic(chain=chain,
                             chain_number=i+1,
                             all_bars=self.bars,
                             all_nodes=self.nodes,
                             color=colors[i],
                             show_annotations=show_annotations,
                             max_dim=self._max_dim,
                             base_scale=self._base_scale)
            )
            # TODO: hier alles in eigene Klassen aufteilen
            #  > erst in ChainGraphic > PoleGraphic > PoleLineGraphic
            if chain.stiff:
                continue

            for rPol in chain.relative_pole:
                line_dict = chain.apole_lines
                if rPol.x is not None:
                    # draw rPol as PointGraph

                    if line_dict is not False:
                        line = line_dict[rPol.node]
                        if not line[1] is None:
                            self._apole_lines.append(
                                LineGraphic.from_slope_intercept(
                                    m=line[0], n=line[1],
                                    boundary=self.boundary,
                                    scatter_options=self.scatter_kwargs | {
                                        'line': dict(width=1),
                                        'line_color': 'blue'
                                    }
                                )
                            )
                else:
                    line = rPol.line()
                    self._apole_lines.append(
                        LineGraphic.from_slope_intercept(
                            m=line[0], n=line[1],
                            boundary=self.boundary,
                            scatter_options=self.scatter_kwargs | {
                                'line': dict(width=1),
                                'line_color': 'blue'
                            }
                        )
                    )

            if chain.absolute_pole is not None:
                self._pole.append(
                    PoleGraphic(pole=chain.absolute_pole, pole_number=i + 1,
                                scale=self._base_scale))

        # TODO: displacement_figure in eigene Klasse DisplacementGraphic
        #  auslagern
        deform = poleplan.get_displacement_figure()

        self._displacement_figure = []

        for idx, bar in enumerate(self.bars):
            deform_vals = np.dot(
                bar.transformation_matrix(to_node_coord=False), deform[idx])

            points = [
                [bar.node_i.x + float(deform_vals[0]),
                 bar.node_i.z + float(deform_vals[1])],
                [bar.node_j.x + float(deform_vals[3]),
                 bar.node_j.z + float(deform_vals[4])]
            ]

            self._displacement_figure.append(
                LineGraphic.from_points(
                    points,
                    scatter_options=self.scatter_kwargs | {
                        'line': dict(width=4),
                        'line_color': 'red'
                    }
                )
            )

    @cached_property
    def _base_scale(self):
        return self.base_scale if self.base_scale \
            else 0.08 * self._max_dim  # TODO: + 0.02

    @property
    def annotations(self):
        annotations = []
        for ch in self._chains:
            annotations.extend(ch.annotations)
        for rp in self._pole:
            annotations.extend(rp.annotations)
        return tuple(annotations)

    @property
    def traces(self):
        traces = []

        for ch in self._chains:
            traces.extend(
                ch.transform_traces(self.x, self.z, self.rotation, self.scale)
            )

        for rp in self._pole:
            traces.extend(
                rp.transform_traces(self.x, self.z, self.rotation, self.scale)
            )

        for line in self._displacement_figure:
            traces.extend(
                line.transform_traces(self.x, self.z, self.rotation,
                                      self.scale)
            )

        for line in self._apole_lines:
            traces.extend(
                line.transform_traces(self.x, self.z, self.rotation,
                                      self.scale)
            )

        return traces


class ChainGraphic(SingleGraphicObject):

    def __init__(
            self, chain: Chain, chain_number: int, all_bars, all_nodes,
            color, show_annotations,
            base_scale=None, max_dim=None, **kwargs
    ):
        if not isinstance(chain, Chain):
            raise TypeError('"chain" has to be an instance of Chain')
        super().__init__(
            chain.nodes[0].x,
            chain.nodes[0].z, **kwargs
        )
        self.chain = chain
        self.nodes = chain.nodes
        self.bars = chain.bars
        self.all_bars = all_bars
        self.color = color
        self.base_scale = base_scale
        self.max_dim = max_dim
        self.number = chain_number

        self._node_chain = [
            NodeGraphic(
                node,
                node_number=all_nodes.index(node) + 1,
                scale=self.base_scale,
                scatter_options=self.scatter_kwargs,
                annotation_options=self.annotation_kwargs,
                show_annotations=show_annotations
            ) for node in self.nodes
        ]

        self._bar_chain = [
            BarChainGraphic(
                bar,
                bar_number=all_bars.index(bar) + 1,
                base_scale=self.base_scale,
                max_dim=self.max_dim,
                show_annotations=show_annotations,
                line_color=self.color,
                annotation_color=self.color,
                scatter_options=self.scatter_options,
                annotation_options=self.annotation_kwargs,
            ) for bar in self.bars
        ]

        if not chain.stiff:
            self._pole = [

            ]

    @cached_property
    def centroid(self):
        nodes = self.nodes
        x_coords = [node.x for node in nodes]
        z_coords = [node.z for node in nodes]
        return (sum(x_coords) / len(x_coords),
                sum(z_coords) / len(z_coords))

    # @property
    # def annotations(self):
    #     annotations = []
    #     for node in self._node_chain:
    #         annotations.extend(node.annotations)
    #     for bar in self._bar_chain:
    #         annotations.extend(bar.annotations)
    #     return tuple(annotations)

    @cached_property
    def _annotation_pos(self):
        # Mittelpunkt der Scheibe = Schwerpunkt aller Knoten
        x_c, z_c = self.centroid
        offset = 0.5 * self.base_scale
        return x_c, z_c - offset

    @property
    def _annotations(self):
        # TODO: wie bekomme ich die Annotation in den Plot entweder
        #  self._annotations oder self.annotations beides geht nicht!
        x, z = self._annotation_pos
        return ((x, z, self._int_to_roman(self.number)),)

    @staticmethod
    def _int_to_roman(number: int) -> str:
        val = [
            1000, 900, 500, 400,
            100, 90, 50, 40,
            10, 9, 5, 4,
            1
        ]
        syms = [
            "M", "CM", "D", "CD",
            "C", "XC", "L", "XL",
            "X", "IX", "V", "IV",
            "I"
        ]
        roman_num = ""
        i = 0
        while number > 0:
            for _ in range(number // val[i]):
                roman_num += syms[i]
                number -= val[i]
            i += 1
        return roman_num

    @property
    def traces(self):
        traces = []
        for node in self._node_chain:
            traces.extend(
                node.transform_traces(self.x, self.z, self.rotation,
                                      self.scale)
            )
        for bar in self._bar_chain:
            traces.extend(
                bar.transform_traces(self.x, self.z, self.rotation, self.scale)
            )
        traces.extend(
            EllipseGraphic(
                *self._annotation_pos, 0.25 * self.base_scale
            ).transform_traces(self.x, self.z, self.rotation,
                               self.scale))
        return traces

#
# class DisplacementGraphic(SingleGraphicObject):
#
#     def __init__(self):
#         pass


class BarChainGraphic(BarGraphic):

    def __init__(
            self,
            bar: Bar,
            bar_number: int | None = None,
            base_scale: float | None = None,
            max_dim: float | None = None,
            show_annotations: bool = True,
            *,
            line_color: str | tuple = None,
            annotation_color: str | tuple = None,
            scatter_options: dict | None = None,
            annotation_options: dict | None = None,
            **kwargs,
    ):

        scatter_opts = scatter_options.copy() if scatter_options else {}
        if line_color is not None:
            scatter_opts = self.scatter_options | {'line_color': line_color}

        annotation_options = annotation_options.copy() \
            if annotation_options else {}
        if annotation_color is not None:
            annotation_options = (self.annotation_options |
                                  {'line_color': line_color})

        super().__init__(
            bar,
            bar_number=bar_number,
            base_scale=base_scale,
            max_dim=max_dim,
            show_annotations=show_annotations,
            scatter_options=scatter_opts,
            **kwargs,
        )

        self._line_color = line_color
        self._annotation_color = annotation_color


class PoleGraphic(SingleGraphicObject):

    def __init__(self, pole: Pole, pole_number=None, **kwargs):
        if not isinstance(pole, Pole):
            raise TypeError('"pole" has to be an instance of Pole')
        super().__init__(pole.node.x, pole.node.z, **kwargs)
        self.node = pole.node
        self.number = pole_number

    @property
    def _annotations(self):
        if self.number is not None:
            d = 0.3 * self.scale
            return ((self.x - d, self.z - d, f'({self.number})'),)
        return ()

    @property
    def traces(self):
        traces = []

        traces.extend(
            PointGraphic(x=self.x, z=self.z).transform_traces(
                self.x, self.z, self.node.rotation, self.scale
            )
        )

        return traces
