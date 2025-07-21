
import numpy as np


def transformation_matrix(alpha: float):
    r"""Create a 3x3 rotation matrix.

    Parameters
    ----------
    alpha : :any:`float`
        Rotation angle in rad.

    Returns
    -------
    :any:`numpy.array`
        A 3x3 matrix for rotating 3x1 vectors.

    Notes
    -----
    The resulting matrix has the form

    .. math::
        \left(\begin{array}{ccc}
        \cos(\alpha) & \sin(\alpha) & 0 \\
        -\sin(\alpha) & \cos(\alpha) & 0 \\
        0 & 0 & 1
        \end{array}\right).

    Examples
    --------
    >>> import numpy
    >>> import sstatics.core
    >>> m = sstatics.core.transformation_matrix(numpy.pi)
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


def get_intersection_point(line_1, line_2):
    m1, n1 = line_1
    m2, n2 = line_2

    # === 1. Identische Geraden (unendlich viele Lösungen) ===
    if m1 == m2 and n1 == n2:
        print(' -> z0 = z1 identische Geraden!')
        return float('inf'), float('inf')

    # === 2. Parallele Geraden (kein Schnittpunkt) ===
    if m1 == m2:
        print(' -> z0 und z1 sind parallele Geraden')
        return None, None

    # === 3. Eine Gerade ist vertikal ===
    if m1 is None:
        return n1, m2 * n1 + n2
    if m2 is None:
        return n2, m1 * n2 + n1

    if np.isclose(m1, m2, atol=1e-9) and np.isclose(n1, n2, atol=1e-9):
        print(' -> z0 = z1 identische Geraden!')
        return float('inf'), float('inf')

    # === 4. Eine Gerade ist horizontal ===
    if m1 == 0:
        return (n1 - n2) / m2, n1
    if m2 == 0:
        return (n2 - n1) / m1, n2

    # === 5. Allgemeiner Fall: Matrizenschreibweise ===
    matrix = np.array([[-m1, 1], [-m2, 1]])
    b = np.array([n1, n2])

    x, z = np.linalg.solve(matrix, b)
    return x, z


def validate_point_on_line(line, point, debug=False, epsilon=1e-9):
    m, n = line
    x, z = point

    if m is None:  # Vertikale Gerade
        result = abs(x - n) < epsilon
    else:
        z_calc = m * x + n
        result = abs(z - z_calc) < epsilon

    if debug:
        print(f"      -> {'JA' if result else 'NEIN'}, Punkt liegt "
              f"{'auf' if result else 'nicht auf'} der Geraden")
    return result


def get_angle(point, center, displacement: float = 1):
    r = point - center

    # Länge des Vektors
    length = np.linalg.norm(r)

    # Bestimme das Vorzeichen
    if np.all(center == 0):
        sign = np.sign(r[0, 0])
    else:
        sign = np.sign(np.dot(r.T, center)).item()

    if displacement == 1:
        return sign / length
    else:
        return displacement / length
