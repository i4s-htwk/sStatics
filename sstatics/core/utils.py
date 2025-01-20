
import numpy as np


def transformation_matrix(alpha: float):
    r"""Create a 3x3 rotation matrix.

    Parameters
    ----------
    alpha : :any:`float`
        Rotation angle.

    Returns
    -------
    :any:`numpy.array`
        A 3x3 matrix for rotating 3x1 vectors.

    Notes
    -----
    The resulting matrix has the form

    .. math::
        \left(\begin{array}{c}
        \cos(\alpha) & \sin(\alpha) & 0 \\
        -\sin(\alpha) & \cos(\alpha) & 0 \\
        0 & 0 & 1
        \end{array}\right).

    Examples
    --------
    >>> import numpy
    >>> import sstatics
    >>> m = sstatics.transformation_matrix(numpy.pi)
    >>> m
    array([[-1, 0, 0],
           [0, -1, 0],
           [0, 0, 1]])
    >>> m @ numpy.array([[1], [2], [3]])
    array([[-1], [-2], [3]])
    """
    return np.array([
        [np.cos(alpha), np.sin(alpha), 0],
        [-np.sin(alpha), np.cos(alpha), 0],
        [0, 0, 1]
    ])
