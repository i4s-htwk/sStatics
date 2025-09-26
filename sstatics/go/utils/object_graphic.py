
from __future__ import annotations

import abc
from typing import Any

import plotly.graph_objs as go

from sstatics.go.utils.style import DEFAULT_LINE, DEFAULT_TEXT
from sstatics.go.utils.transform import Transform


class Figure(go.Figure):

    def __init__(
            self,
            show_axis: bool = True,
            show_grid: bool = False,
            **kwargs
    ):
        layout = go.Layout(
            template='simple_white',
            xaxis=dict(
                visible=show_axis,
                showgrid=show_grid
            ),
            yaxis=dict(
                visible=show_axis,
                showgrid=show_grid,
                autorange='reversed',
                scaleanchor='x',
                scaleratio=1
            ),
        )
        kwargs['layout'] = kwargs.get('layout', layout)
        super().__init__(**kwargs)


class ObjectGraphic(abc.ABC):

    def __init__(
            self,
            origin: tuple[float, float] = (0.0, 0.0),
            rotation: float = 0.0,
            scaling: float = 1.0,
            translation: tuple[float, float] = (0.0, 0.0),
            line_style: dict[str, Any] | None = None,
            text_style: dict[str, Any] | None = None
    ):
        if line_style is None:
            line_style = {}
        if text_style is None:
            text_style = {}

        self._transform = Transform(
            origin=origin, rotation=rotation, scaling=scaling,
            translation=translation
        )
        self._origin = self._transform.origin
        self._line_style = self._deep_style_merge(DEFAULT_LINE, line_style)
        self._text_style = self._deep_style_merge(DEFAULT_TEXT, text_style)

    @abc.abstractmethod
    def poly_lines(self):
        pass

    @abc.abstractmethod
    def text_objects(self):
        pass

    @property
    def traces(self):
        if hasattr(self, '_traces'):
            return self._traces
        traces = []
        for x, z, style in self.poly_lines():
            x, z = self._transform.apply(x, z)
            traces.append(go.Scatter(x=x, y=z, **style))

        for x, z, text, style in self.text_objects():
            x, z = self._transform.apply(x, z)
            traces.append(go.Scatter(
                x=[x], y=[z], text=[text], **style
            ))

        return traces

    def figure(self, show_axis: bool = True, show_grid: bool = False):
        fig = Figure(
            data=self.traces,
            show_axis=show_axis,
            show_grid=show_grid
        )
        return fig

    def show(
            self,
            show_axis: bool = True,
            show_grid: bool = False,
            *args, **kwargs
    ):
        fig = self.figure(show_axis, show_grid)
        fig.show(*args, **kwargs)

    # def save_as_image(
    #         self,
    #         path: str = None,
    #         show_axis: bool = True,
    #         show_grid: bool = False,
    #         **kwargs
    # ):
    #     if not path:
    #         path = filedialog.asksaveasfilename(
    #             defaultextension='.png',
    #             filetypes=[('PNG files', '*.png'),
    #                        ('JPEG files', '*.jpg'),
    #                        ('SVG files', '*.svg'),
    #                        ('PDF files', '*.pdf')]
    #         )
    #     if path:
    #         DEFAULT_SAVE.update(kwargs)
    #         (self.figure(show_axis, show_grid)
    #          .write_image(path, **DEFAULT_SAVE))
    #
    #     sys.exit(0)
    #
    # def save_as_code(
    #         self,
    #         path: str = None,
    #         show_axis: bool = True,
    #         show_grid: bool = False,
    #         **kwargs
    # ):
    #     if not path:
    #         path = filedialog.asksaveasfilename(
    #             defaultextension='.tex',
    #             filetypes=[('LaTex TikZ files', '*.tex')]
    #         )
    #     if path:
    #         DEFAULT_SAVE.update(kwargs)
    #         fig = self.figure(show_axis, show_grid)
    #         code = tikzplotly.get_tikz_code(fig)
    #         with open(path, 'w', encoding='utf-8') as f:
    #             f.write(code)

    @staticmethod
    def _deep_style_merge(base: dict, override: dict) -> dict:
        result = dict(base)
        for k, v in (override or {}).items():
            if (
                k in result and isinstance(result[k], dict)
                and isinstance(v, dict)
            ):
                result[k] = ObjectGraphic._deep_style_merge(result[k], v)
            else:
                result[k] = v
        return result

    @property
    def transform(self):
        return self._transform

    @property
    def origin(self):
        return self._origin

    @property
    def line_style(self):
        return self._line_style

    @property
    def text_style(self):
        return self._text_style


class CombinedGraphic(ObjectGraphic):

    def __init__(
            self,
            objects: list[ObjectGraphic] | tuple[ObjectGraphic] | ObjectGraphic
    ):
        if isinstance(objects, ObjectGraphic):
            objects = objects,
        self._validate(objects)
        super().__init__()
        self._objects = objects

    def poly_lines(self):
        lines = []
        for o in self._objects:
            lines.extend(o.poly_lines())
        return lines

    def text_objects(self):
        texts = []
        for o in self._objects:
            texts.extend(o.text_objects())
        return texts

    @staticmethod
    def _validate(objects):
        if not (
            isinstance(objects, list) or
            isinstance(objects, tuple)
        ):
            raise TypeError(
                f'objects must be a list or tuple, got {objects.__class__!r}'
            )
        if not all(isinstance(v, ObjectGraphic) for v in objects):
            raise TypeError(
                f'objects must contain ObjectGraphic, got {objects!r}'
            )

    @property
    def objects(self):
        return self._objects


class Rectangle(ObjectGraphic):

    def __init__(
            self,
            origin: tuple[float, float],
            width: float,
            height: float,
            text: str = '',
            **kwargs
    ):
        super().__init__(origin=origin, **kwargs)
        self._width = width
        self._height = height
        self._text = text

    def poly_lines(self):
        x_off, z_off = self._width / 2, self._height / 2
        x0, z0 = self.origin
        x = x0 - x_off, x0 + x_off, x0 + x_off, x0 - x_off, x0 - x_off
        z = z0 - z_off, z0 - z_off, z0 + z_off, z0 + z_off, z0 - z_off
        return [(x, z, self._line_style)]

    def text_objects(self):
        x0, z0 = self.origin
        return [(x0, z0, self._text, self._text_style)]
