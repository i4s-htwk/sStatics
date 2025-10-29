
import plotly.graph_objs as go
from functools import cached_property

from .base_renderer import AbstractRenderer
from sstatics.core.postprocessing.graphic_objects.utils.defaults import PLOTLY


class PlotlyRenderer(AbstractRenderer):

    def __init__(
            self,
            show_axis: bool = True,
            show_grid: bool = False,
    ):
        super().__init__(mode=PLOTLY)
        self._show_axis = show_axis
        self._show_grid = show_grid

    @cached_property
    def _layout(self):
        return go.Layout(
            template='simple_white',
            xaxis=dict(
                visible=self._show_axis,
                showgrid=self._show_grid
            ),
            yaxis=dict(
                visible=self._show_axis,
                showgrid=self._show_grid,
                autorange='reversed',
                scaleanchor='x',
                scaleratio=1
            ),
        )

    @cached_property
    def figure(self):
        return go.Figure(layout=self._layout)

    def add_graphic(self, x, z, **style):
        self.figure.add_trace(go.Scatter(x=x, y=z, **style))

    def add_text(self, x, z, text, **style):
        self.figure.add_trace(go.Scatter(x=[x], y=[z], text=[text], **style))

    def show(self, *args, **kwargs):
        self.figure.show(renderer='browser', *args, **kwargs)
