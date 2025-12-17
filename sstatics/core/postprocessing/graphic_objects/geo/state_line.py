import copy

import numpy as np
from functools import cached_property
from types import NoneType

from sstatics.core.postprocessing.graphic_objects.geo.object_geo import \
    ObjectGeo
from sstatics.core.postprocessing.graphic_objects.utils.defaults import (
    DEFAULT_STATE_LINE, DEFAULT_STATE_LINE_TEXT
)
from sstatics.core.postprocessing.graphic_objects.geo import (
    OpenCurveGeo, TextGeo
)


class StateLineGeo(ObjectGeo):
    CLASS_STYLES = {
        'line': DEFAULT_STATE_LINE,
        'text': DEFAULT_STATE_LINE_TEXT
    }

    def __init__(
            self,
            state_line_data: list[dict],
            global_scale: float,
            decimals: int = 2,
            sig_digits: int | None = None,
            scale_state_line: float = 3.0,
            show_maximum: bool = False,
            show_text: bool = True,
            show_connecting_line: bool = False,
            **kwargs
    ):
        """
        state_lines: list of dicts with keys:
            - x: np.ndarray
            - z: np.ndarray
            - translation: (float, float)
            - rotation: float
        """
        self._validate_state_line(
            state_line_data, global_scale, decimals, sig_digits,
            scale_state_line, show_maximum, show_text, show_connecting_line
        )
        super().__init__(
            origin=(0.0, 0.0), global_scale=global_scale, **kwargs
        )
        self._state_line_data = state_line_data
        self._decimals = decimals
        self._sig_digits = sig_digits
        self._scale_state_line = scale_state_line
        self._show_maximum = show_maximum
        self._show_text = show_text
        self._show_connecting_line = show_connecting_line

    @cached_property
    def graphic_elements(self):
        return [
            *self._profile_line_elements
        ]

    @cached_property
    def text_elements(self):
        elements = []
        if not self._show_text:
            return []
        for line, process in zip(
                self._state_line_data, self._process_state_lines
        ):
            text_idx = {0, len(line['z']) - 1}
            if self._show_maximum:
                text_idx.add(np.argmax(np.abs(line['z'])))
            points_text = [
                (
                    (process['x'][i], process['z'][i]),
                    self._round_value(line['z'][i])
                )
                for i in text_idx
            ]
            points, texts = zip(*points_text)
            elements.append(TextGeo(
                self._origin, insertion_points=list(points), texts=list(texts),
                rotation=line['rotation'],
                post_translation=line['translation'],
                text_style=self._text_style
            ))
        return elements

    @cached_property
    def _max_value(self):
        all_z = np.concatenate(
            [np.asarray(line['z'], dtype=float)
             for line in self._state_line_data]
        )
        max_val = np.max(np.abs(all_z))
        return 1e-6 if np.isclose(max_val, 0) else max_val

    @cached_property
    def _process_state_lines(self):
        state_lines = copy.deepcopy(self._state_line_data)
        scale_factor = (
                self._base_scale / self._max_value * self._scale_state_line
        )

        for line in state_lines:
            z = np.array(line['z']) * scale_factor
            line['z'] = z
        return state_lines

    def _round_value(self, value):
        if self._sig_digits is not None:
            value = float(f"{value:.{self._sig_digits}g}")
        else:
            value = round(value, self._decimals)
        return 0.0 if np.isclose(value, 0.0, atol=1e-8) else value

    def _add_points(self, x: np.ndarray, z: np.ndarray):
        x = [x[0], *x, x[-1]]
        z = [0.0, *z, 0.0]
        if self._show_connecting_line:
            x.append(x[0])
            z.append(0.0)
        return x, z

    @property
    def _profile_line_elements(self):
        return [
            OpenCurveGeo(
                *self._add_points(line['x'], line['z']),
                rotation=line['rotation'],
                post_translation=line['translation'],
                line_style=self._line_style
            ) for line in self._process_state_lines
        ]

    @staticmethod
    def _validate_state_line(
            state_line_data, global_scale, decimals, sig_digits,
            scale_state_line, show_maximum, show_text, show_connecting_line
    ):
        if not isinstance(state_line_data, list):
            raise TypeError(
                f'state_line_data must be a list, '
                f'got {type(state_line_data).__name__}.'
            )

        if len(state_line_data) == 0:
            raise ValueError('state_line_data must not be empty.')

        for i, item in enumerate(state_line_data):

            if not isinstance(item, dict):
                raise TypeError(
                    f'Each state line must be a dict, '
                    f'got {type(item).__name__} at index {i}.'
                )

            required_keys = {'x', 'z', 'translation', 'rotation'}
            missing = required_keys - item.keys()
            if missing:
                raise KeyError(
                    f'State line at index {i} is missing keys: '
                    f'{sorted(missing)}.'
                )

            x = item['x']
            z = item['z']
            translation = item['translation']
            rotation = item['rotation']

            if not isinstance(x, (list, tuple, np.ndarray)):
                raise TypeError(
                    f'x must be a list, tuple or ndarray, '
                    f'got {type(x).__name__} at index {i}.'
                )

            if not isinstance(z, (list, tuple, np.ndarray)):
                raise TypeError(
                    f'z must be a list, tuple or ndarray, '
                    f'got {type(z).__name__} at index {i}.'
                )

            x_array = np.asarray(x)
            z_array = np.asarray(z)

            if not np.issubdtype(x_array.dtype, np.number):
                raise TypeError(
                    f'All x values must be numeric at index {i}.'
                )

            if not np.issubdtype(z_array.dtype, np.number):
                raise TypeError(
                    f'All z values must be numeric at index {i}.'
                )

            if x_array.shape != z_array.shape:
                raise ValueError(
                    f'x and z must have the same length at index {i}.'
                )

            if x_array.size < 2:
                raise ValueError(
                    f'A state line requires at least two points at index {i}.'
                )

            if not isinstance(translation, (tuple, list)):
                raise TypeError(
                    f'translation must be a tuple or list, '
                    f'got {type(translation).__name__} at index {i}.'
                )

            if len(translation) != 2:
                raise ValueError(
                    f'translation must have length 2 '
                    f'at index {i}.'
                )

            if not all(isinstance(v, (int, float)) for v in translation):
                raise TypeError(
                    f'translation values must be numbers '
                    f'at index {i}.'
                )

            if not isinstance(rotation, (int, float)):
                raise TypeError(
                    f'rotation must be a number, '
                    f'got {type(rotation).__name__} at index {i}.'
                )

        if not isinstance(global_scale, (int, float)):
            raise TypeError(
                f'"global_scal" must be int or float, got '
                f'{type(global_scale).__name__}'
            )

        if not isinstance(decimals, int):
            raise TypeError(
                f'"decimals" must be int or None, '
                f'got {type(decimals).__name__!r}'
            )

        if not isinstance(sig_digits, (int, NoneType)):
            raise TypeError(
                f'"sig_digits" must be int or None, '
                f'got {type(sig_digits).__name__!r}'
            )

        if sig_digits is not None and sig_digits <= 0:
            raise ValueError('"sig_digits" has to be greater than zero.')

        if not isinstance(scale_state_line, (int, float)):
            raise TypeError(
                f'"scale_state_line" must be int or float, '
                f'got {type(scale_state_line).__name__!r}'
            )

        if not isinstance(show_maximum, bool):
            raise TypeError(
                f'"show_maximum" must be a boolean, got '
                f'{type(show_maximum).__name__!r}'
            )

        if not isinstance(show_text, bool):
            raise TypeError(
                f'"show_text" must be a boolean, got '
                f'{type(show_text).__name__!r}'
            )

        if not isinstance(show_connecting_line, bool):
            raise TypeError(
                f'"show_connecting_line" must be a boolean, got '
                f'{type(show_connecting_line).__name__!r}'
            )

    @property
    def state_line_data(self):
        return self._state_line_data

    @property
    def decimals(self):
        return self._decimals

    @property
    def sig_digits(self):
        return self._sig_digits

    @property
    def scale_state_line(self):
        return self._scale_state_line

    @property
    def show_maximum(self):
        return self._show_maximum

    @property
    def show_text(self):
        return self._show_text

    @property
    def show_connecting_line(self):
        return self._show_connecting_line

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'state_line_data={self._state_line_data}, '
            f'decimals={self._decimals}, '
            f'sig_digits={self._sig_digits}, '
            f'scale_state_line={self._scale_state_line}, '
            f'show_maximum={self._show_maximum}, '
            f'show_text={self._show_text}, '
            f'show_connecting_line={self._show_connecting_line}, '
            f'line_style={self._line_style}, '
            f'text_style={self._text_style}, '
            f'Transform={self._transform})'
        )
