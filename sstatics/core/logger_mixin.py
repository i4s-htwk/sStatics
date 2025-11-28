
import numpy as np

import logging
from tabulate import tabulate

from typing import Any


class LoggerMixin:
    """
    A mixin class providing a configurable logger to any subclass.

    This mixin automatically initializes a class-specific logger using
    Python's built-in `logging` module. It supports both normal and debug
    modes and can integrate seamlessly with `dataclasses` via automatic
    post-initialization.

    Parameters
    ----------
    *args : Any
        Positional arguments passed to the parent class (if any).
    debug : bool, optional
        Enables debug-level logging output if True. Default is False.
    **kwargs : Any
        Additional keyword arguments passed to the parent class (if any).

    Attributes
    ----------
    logger : logging.Logger
        A logger instance configured for the specific subclass.
    """

    _formatter = logging.Formatter(
        fmt=(
            "%(asctime)s [%(levelname)s] %(name)s.%(funcName)s():%(lineno)d "
            "- %(message)s"
        ),
        datefmt="%H:%M:%S",
    )

    def __init__(self, *args: Any, debug: bool = False, **kwargs: Any):
        """
        Initialize the mixin logger and attach it to the subclass instance.

        Parameters
        ----------
        *args : Any
            Positional arguments passed to the parent class (if any).
        debug : bool, optional
            Enables debug-level logging output if True. Default is False.
        **kwargs : Any
            Additional keyword arguments passed to the parent class (if any).
        """
        # ---- Logger setup ----
        self._logger = logging.getLogger(
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )
        self._logger.propagate = False

        # Add a NullHandler if no handlers are present
        if not self._logger.handlers:
            self._logger.addHandler(logging.NullHandler())

        # Default to WARNING level
        self._logger.setLevel(logging.WARNING)

        # Enable debug logging if requested
        if debug:
            # Ensure a StreamHandler exists only once
            if not any(isinstance(h, logging.StreamHandler)
                       for h in self._logger.handlers):
                sh = logging.StreamHandler()
                sh.setFormatter(self._formatter)
                self._logger.addHandler(sh)
            self._logger.setLevel(logging.DEBUG)

        self._logger.debug(
            "Instantiated %s(debug=%s)", self.__class__.__name__, debug
        )
        # ----------------------

    @property
    def logger(self) -> logging.Logger:
        """
        Returns the logger instance associated with this object.

        Returns
        -------
        logging.Logger
            The configured logger.
        """
        if not hasattr(self, "_logger"):
            # fallback in case init was not called
            LoggerMixin.__init__(self)
        return self._logger

    # Automatic subclass hook
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        has_debug = "debug" in getattr(cls, "__annotations__", {})

        # ---------- Case 1: Dataclass → wrap __post_init__ ----------
        orig_post = getattr(cls, "__post_init__", None)
        if orig_post is not None and has_debug:
            def wrapped_post(self, *a, **k):
                LoggerMixin.__init__(self, debug=getattr(self, "debug", False))
                return orig_post(self, *a, **k)

            cls.__post_init__ = wrapped_post
            return  # already handled; no need to touch __init__

        # ---------- Case 2: Normal class with its own __init__ ----------
        orig_init = getattr(cls, "__init__", None)

        # Avoid wrapping LoggerMixin.__init__ itself
        if orig_init is not LoggerMixin.__init__:

            def wrapped_init(self, *a, **k):
                # call mixin init BEFORE user init
                debug_value = k.get("debug", False)
                LoggerMixin.__init__(self, debug=debug_value)

                # then call the original __init__
                if orig_init is not None:
                    return orig_init(self, *a, **k)

            cls.__init__ = wrapped_init


def table_bar(list_of_vec_lists, mapping, header_lists, decimals: int = 6):
    n_bars = len(list_of_vec_lists[0])

    header = (["Bar nr.", "Node nr."] +
              [h for group in header_lists for h in group])

    data = []
    for bar_idx in range(n_bars):

        row_i = [bar_idx, mapping[bar_idx][0]]
        row_j = ["", mapping[bar_idx][1]]

        for vec_list in list_of_vec_lists:
            vec = np.asarray(vec_list[bar_idx]).flatten()
            row_i.extend(vec[:3])
            row_j.extend(vec[3:6])

        data += [row_i, row_j]

    return tabulate(data, headers=header, tablefmt="grid",
                    floatfmt=f".{decimals}f")


def table_node_bar_index(bars, nodes):
    mapping = {}

    for bar_idx, bar in enumerate(bars):
        i = nodes.index(bar.node_i)
        j = nodes.index(bar.node_j)

        mapping[bar_idx] = [i, j]

    return mapping


def table_node(vecs, columns_list, decimals: int = 6):
    flat = [np.asarray(v).flatten() for v in vecs]
    k = len(columns_list[0])
    n_nodes = len(flat[0]) // k

    header = ["Node nr."] + [h for cols in columns_list for h in cols]

    data = []
    for node in range(n_nodes):
        row = [node + 1]
        for fl in flat:
            row.extend(fl[node*k:(node+1)*k])
        data.append(row)

    return tabulate(data, headers=header, tablefmt="grid",
                    floatfmt=f".{decimals}f")


def table_matrix(matrix, column_names=None, decimals=6):
    matrix = np.asarray(matrix)
    n_rows, n_cols = matrix.shape

    if column_names is None:
        column_names = [str(i) for i in range(n_cols)]

    headers = column_names

    data = []
    for i in range(n_rows):
        label = "Sum" if i == n_rows - 1 else i + 1
        data.append([label] + matrix[i].tolist())

    return tabulate(data, headers=headers, tablefmt="grid",
                    floatfmt=f".{decimals}f")
