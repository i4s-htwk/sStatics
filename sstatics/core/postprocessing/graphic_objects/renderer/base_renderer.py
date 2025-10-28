
from abc import ABC, abstractmethod

from ..geo.object_geo import ObjectGeo


class AbstractRenderer(ABC):

    _show_grid: bool
    _show_axis: bool

    @abstractmethod
    def _layout(self):
        pass

    @abstractmethod
    def add_objects(self, *obj: ObjectGeo):
        pass

    @abstractmethod
    def add_graphic(self, x, z, **style):
        pass

    @abstractmethod
    def add_text(self, x, z, text, **style):
        pass

    @abstractmethod
    def figure(self):
        pass

    @abstractmethod
    def show(self):
        pass

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
        for element in getattr(obj, "text_elements", []):
            if isinstance(element, tuple) and len(element) == 4:
                x, z, text, style = element
                x, z = obj.transform(x, z)
                if parent_transform:
                    x, z = parent_transform(x, z)
                yield x, z, text, style
            elif hasattr(element, "text_elements"):
                yield from self._iter_text_elements(
                    element, parent_transform=obj.transform
                )

        for sub in getattr(obj, "graphic_elements", []):
            if hasattr(sub, "text_elements"):
                yield from self._iter_text_elements(
                    sub, parent_transform=obj.transform
                )

    @property
    def show_grid(self):
        return self._show_grid

    @property
    def show_axis(self):
        return self._show_axis
