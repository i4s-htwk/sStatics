
import plotly.graph_objs as go
from functools import cached_property

from .base_renderer import AbstractRenderer
from sstatics.core.postprocessing.graphic_objects.utils.defaults import (
    PLOTLY, DEFAULT_LAYOUT_X, DEFAULT_LAYOUT_Y,
)
from .convert import convert_mpl_to_plotly_layout


class PlotlyRenderer(AbstractRenderer):

    def __init__(
            self,
            show_axis: bool = True,
            show_grid: bool = False,
            x_opts: dict | None = None,
            y_opts: dict | None = None,
    ):
        super().__init__(mode=PLOTLY)
        self._show_axis = show_axis
        self._show_grid = show_grid
        self._x_opts, self._y_opts = self._set_opts(x_opts, y_opts)

    @staticmethod
    def _set_opts(x_opts, y_opts):
        x_opts = DEFAULT_LAYOUT_X.copy() if x_opts is None else x_opts.copy()
        y_opts = DEFAULT_LAYOUT_Y.copy() if y_opts is None else y_opts.copy()
        return convert_mpl_to_plotly_layout(x_opts, y_opts)

    @cached_property
    def _layout(self):
        return go.Layout(
            template='simple_white',
            xaxis=dict(
                visible=self._show_axis,
                showgrid=self._show_grid,
                **self._x_opts
            ),
            yaxis=dict(
                visible=self._show_axis,
                showgrid=self._show_grid,
                **self._y_opts
            ),
        )

    @cached_property
    def figure(self):
        return go.Figure(layout=self._layout)

    def add_graphic(self, x, z, **style):
        self.figure.add_trace(go.Scatter(x=x, y=z, **style))

    def add_text(self, x, z, text, **style):
        self.figure.add_trace(go.Scatter(x=[x], y=[z], text=text, **style))

    def show(self, *args, **kwargs):
        self.figure.show(renderer='browser', *args, **kwargs)
