
from typing import Sequence

import numpy as np


class Transform:
    """Represents a 2D geometric transformation that can be applied to
    coordinates.

    This class encapsulates the core transformations applied to 2D points:
    rotation around an origin, uniform scaling, and translation. By
    centralizing these operations, all graphic objects can consistently
    transform their coordinates without duplicating logic.

    Parameters
    ----------
    origin : tuple[float, float], default=(0.0, 0.0)
        The pivot point for rotation and scaling.
    rotation : float, default=0.0
        The rotation angle in radians, counterclockwise.
    scaling : float, default=1.0
        Uniform scaling factor, must be greater than zero.
    translation : tuple[float, float], default=(0.0, 0.0)
        Translation vector applied after rotation and scaling.

    Raises
    ------
    TypeError
        If the types of the parameters are invalid.
    ValueError
        If `scaling` is less than or equal to zero.
    """
    def __init__(
            self,
            origin: tuple[float, float] = (0.0, 0.0),
            rotation: float = 0.0,
            scaling: float = 1.0,
            translation: tuple[float, float] = (0.0, 0.0)
    ):
        self._validate(origin, rotation, scaling, translation)
        self._origin = origin
        self._rotation = rotation
        self._scaling = scaling
        self._translation = np.array(translation, dtype=float)

    def apply(
            self,
            x: float | int | Sequence[float],
            z: float | int | Sequence[float]
    ):
        """Apply the transformation to a set of 2D coordinates.

        The operations are applied in the following order:
        1. Rotation around the origin.
        2. Scaling relative to the origin.
        3. Translation.

        Parameters
        ----------
        x : array-like
            x-coordinates to transform.
        z : array-like
            z-coordinates to transform.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Transformed coordinates as two numpy arrays:
            (x_transformed, z_transformed).

        Notes
        -----
        This method ensures consistent handling of coordinates for all graphic
        objects. Conversion to numpy arrays happens internally to allow
        vectorized computations.

        Example
        -------
        >>> t = Transform(
        >>>     origin=(1, 1), rotation=np.pi/2, scaling=2, translation=(3, 0)
        >>> )
        >>> x, z = t.apply([0, 1], [0, 1])  # x = [2. 4.], z = [3. 1.]
        """
        x = np.array(x, dtype=float)
        z = np.array(z, dtype=float)

        x, z = self._rotate(x, z)
        x, z = self._scale(x, z)
        x, z = self._translate(x, z)
        return x, z

    def _rotate(self, x, z):
        """Rotate points around the defined origin.

        Parameters
        ----------
        x : :any:`numpy.array`
            The x-coordinates of the points to be rotated.
        z : :any:`numpy.array`
            The z-coordinates of the points to be rotated.

        Returns
        -------
        tuple[:any:`numpy.array`, :any:`numpy.array`]
            The rotated x- and z-coordinates.

        Notes
        -----
        The rotation is applied counterclockwise using the standard 2D rotation
        matrix. The origin of rotation is defined by :py:attr:`_origin`.
        """
        c, s = np.cos(self._rotation), np.sin(self._rotation)
        ox, oz = self._origin[0], self._origin[1]
        x_rot = ox + c * (x - ox) + s * (z - oz)
        z_rot = oz - s * (x - ox) + c * (z - oz)
        return x_rot, z_rot

    def _scale(self, x, z):
        """Scale points relative to the defined origin.

        Parameters
        ----------
        x : :any:`numpy.array`
            The x-coordinates of the points to be scaled.
        z : :any:`numpy.array`
            The z-coordinates of the points to be scaled.

        Returns
        -------
        tuple[:any:`numpy.array`, :any:`numpy.array`]
            The scaled x- and z-coordinates.

        Notes
        -----
        Scaling is applied relative to the origin defined in
        :py:attr:`_origin`. This ensures that the origin itself remains
        invariant under scaling.
        """
        ox, oz = self._origin[0], self._origin[1]
        x_scaled = ox + (x - ox) * self._scaling
        z_scaled = oz + (z - oz) * self._scaling
        return x_scaled, z_scaled

    def _translate(self, x, z):
        """Translate points by the defined translation vector.

        Parameters
        ----------
        x : :any:`numpy.array`
            The x-coordinates of the points to be translated.
        z : :any:`numpy.array`
            The z-coordinates of the points to be translated.

        Returns
        -------
        tuple[:any:`numpy.array`, :any:`numpy.array`]
            The translated x- and z-coordinates.

        Notes
        -----
        The translation is applied after rotation and scaling, shifting all
        coordinates by the specified offset in :py:attr:`_translation`.
        """
        x_shifted = x + self._translation[0]
        z_shifted = z + self._translation[1]
        return x_shifted, z_shifted

    @staticmethod
    def _validate(origin, rotation, scaling, translation):
        """Validate parameters for the Transform constructor.

        Raises
        ------
        TypeError
            If the types of the parameters are incorrect.
        ValueError
            If `scaling` <= 0.
        """
        if not isinstance(origin, tuple) or len(origin) != 2:
            raise TypeError(
                f'origin must be a tuple of length 2, got {origin!r}.'
            )

        if not all(isinstance(v, (int, float)) for v in origin):
            raise TypeError(f'origin must contain numbers, got {origin!r}.')

        if not isinstance(rotation, (int, float)):
            raise TypeError(f'rotation must be a number, got {rotation!r}.')

        if not isinstance(scaling, (int, float)):
            raise TypeError(f'scaling must be a number, got {scaling!r}.')

        if scaling < 0:
            raise ValueError(f'scaling must not be negative, got {scaling!r}.')

        if not isinstance(translation, tuple) or len(translation) != 2:
            raise TypeError(
                f'translation must be a tuple of length 2, '
                f'got {translation!r}.'
            )

        if not all(isinstance(v, (int, float)) for v in translation):
            raise TypeError(
                f'translation must contain numbers, got {translation!r}.'
            )

    @property
    def origin(self):
        return self._origin

    @property
    def rotation(self):
        return self._rotation

    @property
    def scaling(self):
        return self._scaling

    @property
    def translation(self):
        return self._translation

    def __call__(self, x, z):
        """Apply the transformation by calling the instance directly.

        This is a shorthand for :meth:`apply`, enabling syntax like:
        >>> transform = Transform(rotation=np.pi / 2)
        >>> x_new, z_new = transform(x, z)

        Parameters
        ----------
        x : float | Sequence[float]
            The x-coordinates to transform.
        z : float | Sequence[float]
            The z-coordinates to transform.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The transformed coordinates.
        """
        return self.apply(x, z)

    def __repr__(self):
        """Return a concise string representation for debugging.

        Returns
        -------
        str
            A developer-friendly string showing key transformation parameters,
            e.g.:
            ``Transform(
            origin=(0, 0), rotation=1.571, scaling=1.0, translation=(0, 0)
            )``.
        """
        return (
            f'Transform(origin={self._origin}, rotation={self._rotation:.3f}, '
            f'scaling={self._scaling}, translation={tuple(self._translation)})'
        )
