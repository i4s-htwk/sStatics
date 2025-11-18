
from functools import cache, cached_property

import matplotlib.pyplot as plt

from .base_renderer import AbstractRenderer
from sstatics.core.postprocessing.graphic_objects.utils.defaults import (
    MPL, DEFAULT_LAYOUT_X, DEFAULT_LAYOUT_Y
)
from .convert import convert_plotly_to_mpl_layout


class MplRenderer(AbstractRenderer):

    def __init__(
            self,
            show_axis: bool = True,
            show_grid: bool = False,
            x_opts: dict | None = None,
            y_opts: dict | None = None,
            **kwargs
    ):
        super().__init__(mode=MPL)
        self._show_axis = show_axis
        self._show_grid = show_grid
        self._x_opts, self._y_opts = self._set_opts(x_opts, y_opts)
        self._fig, self._ax = plt.subplots(**kwargs)
        self._z_order = 0
        self._layout()

    @staticmethod
    def _set_opts(x_opts, y_opts):
        x_opts = DEFAULT_LAYOUT_X.copy() if x_opts is None else x_opts.copy()
        y_opts = DEFAULT_LAYOUT_Y.copy() if y_opts is None else y_opts.copy()
        return convert_plotly_to_mpl_layout(x_opts, y_opts)

    @cache
    def _layout(self):
        if not self._show_axis:
            self._ax.axis('off')
        self._ax.grid(self._show_grid)

        if 'aspect' in self._y_opts:
            self._ax.set_aspect('equal', adjustable='datalim')

        if lim := self._x_opts.get("xlim"):
            self._ax.set_xlim(lim)
        if self._y_opts.get("invert_yaxis", False):
            self._ax.invert_yaxis()
        if lim := self._y_opts.get("ylim"):
            self._ax.set_ylim(lim)

        self._fig.tight_layout()

    @cached_property
    def figure(self):
        return self._fig

    def add_graphic(self, x, y, **style):
        style = self._fix_z_order(style)
        if style.pop('fill', False):
            self._ax.fill(x, y, **style)
        else:
            self._ax.plot(x, y, **style)

    def add_text(self, x, y, text, **style):
        style = self._fix_z_order(style)
        self._ax.text(x, y, *text, **style)

    def show(self, *args, **kwargs):
        plt.show(*args, **kwargs)

    def _fix_z_order(self, style):
        style = style.copy()
        z = getattr(self, '_z_order', 0)
        self._z_order = z + 1
        style.setdefault('zorder', z)
        return style
