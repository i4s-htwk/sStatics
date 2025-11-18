
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
        self._patch_mpl_figure_repr()

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

    def show(
            self, show_axis=True, show_grid=False,
            x_opts: dict | None = None, y_opts: dict | None = None
    ):
        for renderer in self._render(
                show_axis, show_grid, x_opts, y_opts
        ):
            renderer.show()

    @cache
    def figure(self, show_axis=True, show_grid=False):
        figs = []
        for renderer in self._render(show_axis, show_grid):
            figs.append(renderer.figure)
        return figs

    def _render(
            self, show_axis=True, show_grid=False,
            x_opts: dict | None = None, y_opts: dict | None = None
    ):
        figures = []
        for objs, mode in self._groups:
            renderer = self._make_renderer(
                mode, show_axis, show_grid, x_opts, y_opts
            )
            renderer.add_objects(*objs)
            figures.append(renderer)
        return figures

    @staticmethod
    def _make_renderer(mode, show_axis, show_grid, x_opts, y_opts):
        renderer_cases = {
            PLOTLY: PlotlyRenderer,
            MPL: MplRenderer
        }
        case = renderer_cases.get(mode)
        if case is None:
            raise ValueError(f'mode must be PLOTLY or MPL, got {mode!r}')
        return case(
            show_axis=show_axis, show_grid=show_grid,
            x_opts=x_opts, y_opts=y_opts
        )

    @property
    def groups(self):
        return self._groups

    @staticmethod
    def _describe_mpl_figure(fig):
        info = {'axes': []}
        for ax in fig.axes:
            ax_info = {
                'title': ax.get_title(),
                'xlim': ax.get_xlim(),
                'ylim': ax.get_ylim(),
                'lines': [],
                'patches': [],
                'texts': []
            }

            for line in ax.lines:
                ax_info['lines'].append({
                    'xdata': line.get_xdata().tolist(),
                    'ydata': line.get_ydata().tolist(),
                    'style': {
                        'color': line.get_color(),
                        'linewidth': line.get_linewidth(),
                        'linestyle': line.get_linestyle()
                    }
                })

            for patch in ax.patches:
                ax_info['patches'].append({
                    'type': type(patch).__name__,
                    'facecolor': patch.get_facecolor(),
                    'edgecolor': patch.get_edgecolor(),
                    'alpha': patch.get_alpha(),
                })

            for text in ax.texts:
                ax_info['texts'].append({
                    'text': text.get_text(),
                    'position': text.get_position(),
                    'color': text.get_color(),
                    'fontsize': text.get_fontsize()
                })

            info['axes'].append(ax_info)
        return info

    @staticmethod
    def _patch_mpl_figure_repr():
        import matplotlib.figure

        if not hasattr(matplotlib.figure.Figure, '_patched_repr'):
            def _custom_fig_repr(self):
                from pprint import pformat
                return pformat(ObjectRenderer._describe_mpl_figure(self))

            matplotlib.figure.Figure.__repr__ = _custom_fig_repr
            matplotlib.figure.Figure._patched_repr = True
