import copy

import numpy as np
from functools import cached_property
from types import NoneType

from sstatics.core.postprocessing.graphic_objects.geo.object_geo import \
    ObjectGeo
from sstatics.core.postprocessing.graphic_objects.geo.geometry import (
    OpenCurveGeo
)
from sstatics.core.postprocessing.graphic_objects.utils.defaults import (
    DEFAULT_STATE_LINE
)


class StateLineGeo(ObjectGeo):
    CLASS_STYLES = {
        'line': DEFAULT_STATE_LINE
    }

    def __init__(
            self,
            state_line_data: list[dict],
            decimals: int | None = None,
            sig_digits: int | None = None,
            scale_state_line: float = 3.0,
            show_maximum: bool = False,
            show_text: bool = True,
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
            state_line_data, decimals, sig_digits, scale_state_line,
            show_maximum, show_text
        )
        super().__init__(origin=(0.0, 0.0), **kwargs)
        self._state_line_data = state_line_data
        self._decimals = decimals
        self._sig_digits = sig_digits
        self._scale_state_line = scale_state_line
        self._show_maximum = show_maximum
        self._show_text = show_text

    @cached_property
    def graphic_elements(self):
        return [
            *self._boundary_line_elements,
            *self._profile_line_elements
        ]

    @cached_property
    def text_elements(self):
        return []

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
            x = np.array(line['x'])
            z = np.array(line['z']) * scale_factor
            line['x'] = x
            line['z'] = z
        return state_lines

    def _vertical_boundary(self, x, z, line):
        return OpenCurveGeo(
            [x, x], [0.0, z],
            origin=self._origin,
            rotation=line['rotation'], post_translation=line['translation'],
            line_style=self._line_style
        )

    @property
    def _boundary_line_elements(self):
        elements = []
        for line in self._process_state_lines:
            elements.extend([
                self._vertical_boundary(0.0, line['z'][0], line),
                self._vertical_boundary(line['x'][-1], line['z'][-1], line),
            ])
        return elements

    def _round_value(self, value):
        if self._decimals is not None:
            round_value = np.round(value, self._decimals)
        elif self._sig_digits is not None:
            round_value = float(f"{value:.{self._sig_digits}g}")
        else:
            round_value = np.round(value, 2)
        return (
            0.0 if np.isclose(round_value, 0.0, atol=1e-8) else round_value
        )

    @cached_property
    def _text_values(self):
        return [
            [self._round_value(line['z'][0]), self._round_value(line['z'][-1])]
            for line in self._state_line_data
        ]

    @property
    def _profile_line_elements(self):
        return [
            OpenCurveGeo(
                list(line['x']), list(line['z']), origin=self._origin,
                text=self._text_values[i] if self._show_text else '',
                rotation=line['rotation'],
                post_translation=line['translation'],
                line_style=self._line_style, text_style=self._text_style
            ) for i, line in enumerate(self._process_state_lines)
        ]

    @staticmethod
    def _validate_state_line(
            state_line_data, decimals, sig_digits, scale_state_line,
            show_maximum, show_text
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

        if not isinstance(decimals, (int, NoneType)):
            raise TypeError(
                f'"decimals" must be int or None, '
                f'got {type(decimals).__name__!r}'
            )

        if not isinstance(sig_digits, (int, NoneType)):
            raise TypeError(
                f'"sig_digits" must be int or None, '
                f'got {type(sig_digits).__name__!r}'
            )

        if decimals is not None and sig_digits is not None:
            raise ValueError(
                'Specify only one of "decimals" or "sig_digits", not both.'
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

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'state_line_data={self._state_line_data}, '
            f'decimals={self._decimals}, '
            f'sig_digits={self._sig_digits}, '
            f'scale_state_line={self._scale_state_line}, '
            f'show_maximum={self._show_maximum}, '
            f'show_text={self._show_text}, '
            f'line_style={self._line_style}, '
            f'text_style={self._text_style}, '
            f'Transform={self._transform})'
        )
