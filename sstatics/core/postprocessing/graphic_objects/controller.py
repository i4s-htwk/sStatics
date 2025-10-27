
from sstatics.core.preprocessing.geometry import Polygon
from .geo.geometry import PolygonGeo
from .renderer import ObjectRenderer
from .utils.defaults import PLOTLY, MPL


class ControllerGraphic:

    def __init__(
            self,
            *objects,
    ):
        for obj in objects:
            if isinstance(obj, str) and obj not in (PLOTLY, MPL):
                raise ValueError(
                    f'mode in objects must be PLOTLY or MPL, got {obj!r}'
                )
        self._objects = list(objects)
        self._extra_models = []

    def show(self, show_axis=True, show_grid=False):
        renderer = ObjectRenderer(self.models)
        renderer.show(show_axis, show_grid)

    def add_models(self, *models):
        self._extra_models.append(*models)

    def _build_models(self):
        models = []
        for obj in self._objects:
            if isinstance(obj, (list, tuple)):
                for o in obj:
                    models.append(self._core_to_model(o))
            else:
                models.append(self._core_to_model(obj))

        return models

    @staticmethod
    def _core_to_model(o):
        if isinstance(o, Polygon):
            return PolygonGeo(o)

        else:
            raise ValueError(
                f'Invalid argument: {o!r}, expected an object from the '
                f'classes of core, a list of them, or mode string'
            )

    @property
    def models(self):
        return self._build_models() + self._extra_models
