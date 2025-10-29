
from functools import cache, cached_property

import matplotlib.pyplot as plt

from .base_renderer import AbstractRenderer
from sstatics.core.postprocessing.graphic_objects.utils.defaults import MPL


class MplRenderer(AbstractRenderer):

    def __init__(
            self,
            show_axis: bool = True,
            show_grid: bool = False,
            **kwargs
    ):
        super().__init__(mode=MPL)
        self._show_axis = show_axis
        self._show_grid = show_grid
        self._fig, self._ax = plt.subplots(**kwargs)
        self._z_order = 0
        self._layout()

    @cache
    def _layout(self):
        if not self._show_axis:
            self._ax.axis('off')
        self._ax.grid(self._show_grid)
        self._ax.set_aspect('equal', adjustable='datalim')
        self._ax.invert_yaxis()
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
        self._ax.text(x, y, text, **style)

    def show(self, *args, **kwargs):
        plt.show(*args, **kwargs)

    def _fix_z_order(self, style):
        style = style.copy()
        z = getattr(self, "_z_order", 0)
        self._z_order = z + 1
        style.setdefault("zorder", z)
        return style
