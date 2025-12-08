
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
    >>> from sstatics.core.utils import transformation_matrix
    >>> m = transformation_matrix(numpy.pi)
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


def get_differential_equation(
    system,
    deform_list: list[np.ndarray],
    forces_list: list[np.ndarray],
    idx: int | None = None,
    n_disc: int = 10
):
    r"""Create one or multiple :class:`DifferentialEquation` instances.

    This function constructs differential equation objects for bar
    elements based on deformation and force data. If an index is provided,
    only the corresponding bar is processed; otherwise, a list of differential
    equations for all bars in the system is returned.

    Parameters
    ----------
    system : :any:`object`
        Structural system containing a mesh with bar elements. The function
        expects a ``System`` instance to provide iterable bar objects.
    deform_list : list of :any:`numpy.ndarray`
        List of deformation vectors associated with each bar. Each entry must
        correspond to one bar in ``system.mesh.bars``.
    forces_list : list of :any:`numpy.ndarray`
        List of internal force result vectors for each bar. Each entry must
        match the corresponding deformation entry and bar index.
    idx : :any:`int`, optional
        If provided, only the bar with this index is processed and a single
        :class:`DifferentialEquation` instance is returned.
        Default is ``None``.
    n_disc : :any:`int`, default=10
        Number of discretization points used for constructing the differential
        equation along the bar.

    Returns
    -------
    :class:`DifferentialEquation` or :any:`list`
        If ``index`` is given, a single differential equation object is
        returned. Otherwise, a list containing one instance per bar is created.

    Examples
    --------
    Compute differential equations for all bars:

    >>> from sstatics.core.utils import get_differential_equation
    >>> eq_list = get_differential_equation(system, deform_list, forces_list)
    >>> len(eq_list)
    5

    Compute the differential equation for a single bar:

    >>> eq = get_differential_equation(system, deform_list, forces_list, idx=2)
    >>> type(eq)
    <class 'sstatics.core.postprocessing.results.DifferentialEquation'>
    """
    from sstatics.core.postprocessing.results import DifferentialEquation

    if system.__class__.__name__ != "System":
        raise TypeError("`system` must be a System instance.")

    bars = system.mesh
    n_bars = len(bars)

    if idx is not None:
        if not isinstance(idx, int):
            raise TypeError(
                f"`index` must be an int or None, got {type(idx)}.")

        if not (-n_bars <= idx < n_bars):
            raise IndexError(
                f"Index {idx} is out of valid range [-{n_bars}, "
                f"{n_bars - 1}].")

    if len(deform_list) != n_bars:
        raise ValueError(
            f"`deform_list` must have {n_bars} entries, got "
            f"{len(deform_list)}.")

    if len(forces_list) != n_bars:
        raise ValueError(
            f"`forces_list` must have {n_bars} entries, "
            f"got {len(forces_list)}.")

    for i, d in enumerate(deform_list):
        if d.shape not in [(6, 1), (6,)]:
            raise ValueError(
                f"deform_list[{i}] must have shape (6,1) or (6,), "
                f"got {d.shape}.")

    for i, f in enumerate(forces_list):
        if f.shape not in [(6, 1), (6,)]:
            raise ValueError(
                f"forces_list[{i}] must have shape (6,1) or (6,), "
                f"got {f.shape}.")

    if idx is not None:
        return DifferentialEquation(
            bar=bars[idx],
            deform=deform_list[idx],
            forces=forces_list[idx],
            n_disc=n_disc,
        )

    return [
        DifferentialEquation(
            bar=bars[i],
            deform=deform_list[i],
            forces=forces_list[i],
            n_disc=n_disc,
        )
        for i in range(len(bars))
    ]
