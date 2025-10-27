
from abc import ABC, abstractmethod

from ..geo.object_geo import ObjectGeo


class AbstractRenderer(ABC):

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
