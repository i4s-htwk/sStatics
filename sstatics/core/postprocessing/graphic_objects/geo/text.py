
from functools import cached_property

from sstatics.core.postprocessing.graphic_objects.geo.object_geo import \
    ObjectGeo


class TextGeo(ObjectGeo):

    def __init__(
            self,
            transform_position: tuple[float, float],
            insertion_points: list[tuple[float, float]] | tuple[float, float],
            texts: str | list[str],
            **kwargs
    ):
        super().__init__(origin=transform_position, text=texts, **kwargs)
        self._validate_text(insertion_points, texts)
        self._insertion_points = (
            insertion_points if isinstance(insertion_points, list)
            else [insertion_points]
        )

    @cached_property
    def graphic_elements(self):
        return []

    @cached_property
    def text_elements(self):
        elements = []
        for pos, text in zip(self._insertion_points, self._text):
            elements.append((*pos, [text], self._text_style))
        return elements

    @staticmethod
    def _validate_text(insertion_points, texts):
        if not isinstance(insertion_points, (tuple, list)):
            raise TypeError(
                f'"insertion_points" must be tuple or list, '
                f'got {type(insertion_points).__name__}'
            )

        if isinstance(insertion_points, tuple):
            if not all(isinstance(v, (int, float)) for v in insertion_points):
                raise TypeError(
                    'Every element of the tuple "insertion_points" must be '
                    'int or float.'
                )

        if isinstance(insertion_points, list):
            if not all(isinstance(v, tuple) for v in insertion_points):
                raise TypeError(
                    '"Every element of the list "insertion_points" must be a '
                    'tuple.'
                )

            if (not all(
                    isinstance(v, (int, float))
                    for tup in insertion_points for v in tup
            )):
                raise TypeError(
                    'Every value inside the tuples of "insertion_points" must '
                    'be int or float.'
                )

            if len(texts) != len(insertion_points):
                raise ValueError(
                    '"texts" must have the same length as "insertion_points"'
                )

    @property
    def transform_position(self):
        return self._origin

    @property
    def insertion_points(self):
        return self._insertion_points

    @property
    def texts(self):
        return self._text

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'transform_position={self._origin}, '
            f'insertion_points={self._insertion_points}, '
            f'texts={self._text}, '
            f'text_style={self._text_style}, '
            f'Transform={self._transform})'
        )
