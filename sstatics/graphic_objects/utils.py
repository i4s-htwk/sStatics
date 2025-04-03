
import abc

import numpy as np
import plotly.graph_objs as go


def rotate(ox, oz, x, z, rotation=0):
    x_rot = ox + np.cos(rotation) * (x - ox) - np.sin(rotation) * (z - oz)
    z_rot = oz + np.sin(rotation) * (x - ox) + np.cos(rotation) * (z - oz)
    return x_rot, z_rot


class Figure(go.Figure):

    def __init__(self, **kwargs):
        layout = go.Layout(
            template='simple_white', yaxis_autorange='reversed',
            yaxis_scaleanchor='x', yaxis_scaleratio=1
        )
        kwargs['layout'] = kwargs.get('layout', layout)
        super().__init__(**kwargs)


class GraphicObject(abc.ABC):

    # TODO: find a better solution to pass down options to customize an object
    # TODO: each instance should be customizable
    # TODO: default values should be the intended appearance
    scatter_options = {
        'mode': 'lines',
        'line_color': 'black',
        'showlegend': False,
        'hoverinfo': 'skip',
    }

    def __init__(self, x, z, rotation=0, scale=1, **scatter_kwargs):
        if scale <= 0:
            raise ValueError('"scale" has to be a number greater than zero.')
        self.x = x
        self.z = z
        self.rotation = rotation
        self.scale = scale
        self.scatter_kwargs = self.scatter_options | scatter_kwargs

    @property
    def annotations(self):
        return ()

    @property
    @abc.abstractmethod
    def traces(self):
        pass

    def rotate_traces(self, ox, oz, rotation=0):
        traces = []
        for trace in self.traces:
            x, z = rotate(
                ox, oz, np.array(trace.x), np.array(trace.y), rotation
            )
            traces.append(trace.update(x=x, y=z))
        return tuple(traces)

    def show(self, *args, **kwargs):
        fig = Figure(data=self.traces)
        for annotation in self.annotations:
            fig.add_annotation(annotation)
        fig.show(*args, **kwargs)
