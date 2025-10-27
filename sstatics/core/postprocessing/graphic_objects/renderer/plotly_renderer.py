
import plotly.graph_objs as go
from functools import cached_property

from .base_renderer import AbstractRenderer
from .convert import convert_style
from sstatics.core.postprocessing.graphic_objects.utils.defaults import PLOTLY


# TODO: getter
class PlotlyRenderer(AbstractRenderer):

    def __init__(
            self,
            show_axis: bool = True,
            show_grid: bool = False,
    ):
        self._validate(show_axis, show_grid)
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

    # TODO: make it better
    def add_objects(self, *obj):
        for o in obj:
            for x, z, style in self._iter_graphic_elements(o):
                x, z = o.transform(x, z)
                style = convert_style(style, PLOTLY)
                self.add_graphic(x, z, **style)

            for x, z, text, style in self._iter_text_elements(o):
                x, z = o.transform(x, z)
                style = convert_style(style, PLOTLY)
                self.add_text(x, z, text, **style)

    def _iter_graphic_elements(self, obj, current_transform=None):
        for element in obj.graphic_elements:
            if hasattr(element, "graphic_elements"):
                yield from self._iter_graphic_elements(
                    element, current_transform=obj.transform
                )
            else:
                x, z, style = element
                x, z = obj.transform(x, z)

                if current_transform:
                    x, z = current_transform(x, z)

                yield x, z, style

    def _iter_text_elements(self, obj, parent_transform=None):
        """
        Rekursive Generatorfunktion, um alle Textelemente eines Geo-Objekts
        (und seiner Unterobjekte) zu durchlaufen.

        Liefert Tupel (x, z, text, style) zurück, wobei alle Transformationen
        korrekt kombiniert sind.
        """
        # 1. Eigene Textelemente durchlaufen
        for element in getattr(obj, "text_elements", []):
            # Das kann ein einfaches Tupel (x, z, text, style) sein
            if isinstance(element, tuple) and len(element) == 4:
                x, z, text, style = element
                x, z = obj.transform(x, z)
                if parent_transform:
                    x, z = parent_transform(x, z)
                yield x, z, text, style
            # oder evtl. Unterobjekte enthalten (selten)
            elif hasattr(element, "text_elements"):
                yield from self._iter_text_elements(
                    element, parent_transform=obj.transform
                )

        # 2. Unterobjekte durchsuchen (z. B. CenterGeo, HatchGeo etc.)
        for sub in getattr(obj, "graphic_elements", []):
            if hasattr(sub, "text_elements"):
                yield from self._iter_text_elements(
                    sub, parent_transform=obj.transform
                )

    def add_graphic(self, x, z, **style):
        self.figure.add_trace(go.Scatter(x=x, y=z, **style))

    def add_text(self, x, z, text, **style):
        self.figure.add_trace(go.Scatter(x=[x], y=[z], text=[text], **style))

    def show(self, *args, **kwargs):
        self.figure.show(renderer='browser', *args, **kwargs)

    @staticmethod
    def _validate(show_axis, show_grid):
        pass
