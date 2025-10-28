from functools import cache

from ..geo.object_geo import ObjectGeo
from .plotly_renderer import PlotlyRenderer
from .mpl_renderer import MplRenderer
from sstatics.core.postprocessing.graphic_objects.utils.defaults import (
    DEFAULT_MODE, PLOTLY, MPL
)


class ObjectRenderer:

    def __init__(
            self,
            *objects: (
                    ObjectGeo | list[ObjectGeo] | tuple[ObjectGeo] | str
            ),
    ):
        self._groups = []
        self._parse_objects(objects)

    def _parse_objects(self, objects):
        pending_objects = []
        for obj in objects:

            if isinstance(obj, str) and obj in (PLOTLY, MPL):
                for po in pending_objects:
                    self._groups.append((
                        po if isinstance(po, (list, tuple)) else [po],
                        obj
                    ))
                pending_objects = []

            elif isinstance(obj, ObjectGeo):
                pending_objects.append(obj)

            elif isinstance(obj, (list, tuple)):
                if not all(isinstance(o, ObjectGeo) for o in obj):
                    raise TypeError(f'Invalid list element in {obj!r}')
                pending_objects.append(obj)

            else:
                raise TypeError(
                    f'Invalid argument: {obj!r}: expected ObjectModel, '
                    f'list[ObjectModel], or mode string.'
                )

        for po in pending_objects:
            self._groups.append((
                po if isinstance(po, (list, tuple)) else [po],
                DEFAULT_MODE
            ))

    @cache
    def show(self, show_axis=True, show_grid=False):
        for renderer in self._render(show_axis, show_grid):
            renderer.show()

    @cache
    def figure(self, **kwargs):
        figs = []
        for renderer in self._render(**kwargs):
            figs.append(renderer.figure)
        return figs

    @cache
    def _render(self, show_axis=True, show_grid=False):
        figures = []
        for objs, mode in self._groups:
            renderer = self._make_renderer(mode, show_axis, show_grid)
            renderer.add_objects(*objs)
            figures.append(renderer)
        return figures

    @staticmethod
    def _make_renderer(mode, show_axis, show_grid):
        if mode == PLOTLY:
            return PlotlyRenderer(show_axis=show_axis, show_grid=show_grid)
        elif mode == MPL:
            return MplRenderer(show_axis=show_axis, show_grid=show_grid)
        else:
            raise ValueError(f'mode must be PLOTLY or MPL, got {mode!r}')

    @property
    def groups(self):
        return self._groups
