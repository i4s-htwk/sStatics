
import abc

import numpy as np
import plotly.graph_objs as go


def translate(ox, oz, x, z, translation=(0, 0)):
    x_shift = ox + (x - ox) + translation[0]
    z_shift = oz + (z - oz) + translation[1]
    return x_shift, z_shift


# TODO: 3 functions but only one used?
def rotate(ox, oz, x, z, rotation=0):
    # always used for annotations in CoordinateSystem
    x_rot = ox + np.cos(rotation) * (x - ox) + np.sin(rotation) * (z - oz)
    z_rot = oz - np.sin(rotation) * (x - ox) + np.cos(rotation) * (z - oz)
    return x_rot, z_rot


def scaling(ox, oz, x, z, scale=1):
    x_scale = ox + (x - ox) * scale
    z_scale = oz + (z - oz) * scale
    return x_scale, z_scale


def transform(ox, oz, x, z, rotation=0, scale=1, translation=(0, 0)):
    x, z = rotate(ox, oz, x, z, rotation)
    x, z = scaling(ox, oz, x, z, scale)
    x, z = translate(ox, oz, x, z, translation)
    return x, z


class Figure(go.Figure):

    def __init__(self, show_grid=False, **kwargs):
        layout = go.Layout(
            template='simple_white', yaxis_autorange='reversed',
            yaxis_scaleanchor='x', yaxis_scaleratio=1,
            xaxis=dict(showgrid=show_grid),
            yaxis=dict(showgrid=show_grid),
        )
        kwargs['layout'] = kwargs.get('layout', layout)
        super().__init__(**kwargs)


class MultiGraphicObject(abc.ABC):

    # TODO: find a better solution to pass down options to customize an object
    # TODO: each instance should be customizable
    # TODO: default values should be the intended appearance
    # TODO: solution: style-file oder class(es)?
    scatter_options = {
        'mode': 'lines',
        'line_color': 'black',
        'showlegend': False,
        'hoverinfo': 'skip',
    }
    annotation_options = {
        'font': {'size': 20, 'family': 'Times New Roman'},
        'showarrow': False,
        'textangle': 0,
    }

    def __init__(
            self, points, scatter_options=None, annotation_options=None,
            rotation=0, scale=1, translation=(0, 0)
    ):
        if scale <= 0:
            raise ValueError('"scale" has to be a number greater than zero.')
        if not (
            isinstance(points, list) and
            all(
                isinstance(p, (list, tuple)) and len(p) == 2 for p in points
            )
        ):
            raise ValueError(
                '"points" has to be a list of (x, z) tuples or [x, z] lists.'
            )
        self.points = points
        self.rotation = rotation
        self.scale = scale
        self.tranlation = translation
        self.scatter_kwargs = (
                self.scatter_options | (scatter_options or {})
        )
        self.annotation_kwargs = (
                self.annotation_options | (annotation_options or {})
        )

    @property
    def _annotations(self):
        """Override this to return a sequence of (x, y, text)."""
        return ()

    @property
    def annotations(self):
        return tuple(
            go.layout.Annotation(x=x, y=y, text=text, **self.annotation_kwargs)
            for x, y, text in self._annotations
        )

    @property
    @abc.abstractmethod
    def traces(self):
        pass

    def transform_traces(
            self, ox, oz, rotation=0, scale=1.0, translation=(0, 0)
    ):
        traces = []
        for trace in self.traces:
            x, z = np.array(trace.x), np.array(trace.y)
            x, z = transform(ox, oz, x, z, rotation, scale, translation)
            traces.append(trace.update(x=x, y=z))
        return tuple(traces)

    def show(self, show_grid=False, *args, **kwargs):
        fig = Figure(data=self.traces, show_grid=show_grid)
        for annotation in self.annotations:
            fig.add_annotation(annotation)
        fig.show(renderer='browser', *args, **kwargs)


class SingleGraphicObject(MultiGraphicObject):
    def __init__(self, x, z, **kwargs):
        super().__init__([(x, z)], **kwargs)
        self.x = x
        self.z = z


class EmptyGraphicObject(SingleGraphicObject):
    def __init__(self):
        super().__init__(0, 0)

    @property
    def traces(self):
        return []
