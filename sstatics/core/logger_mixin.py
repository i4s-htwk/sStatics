import logging
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
        return self._logger

    # ---------------------------------------------------------
    # Automatic subclass hook
    # ---------------------------------------------------------
    def __init_subclass__(cls, **kwargs: Any):
        """
        Automatically called when a subclass of LoggerMixin is defined.

        If the subclass defines a `debug` attribute (e.g., as a dataclass
        field), this method automatically injects a `__post_init__` wrapper
        to ensure that the mixin logger is initialized with the correct
        debug value.

        Parameters
        ----------
        **kwargs : Any
            Keyword arguments passed by Python during subclass creation.
        """
        super().__init_subclass__(**kwargs)

        # Check if subclass defines a 'debug' field or annotation.
        has_debug = "debug" in getattr(cls, "__annotations__", {})

        if not has_debug:
            # If no debug field exists, no modification is needed.
            return

        # Save the original __post_init__ method (if defined)
        orig_post = getattr(cls, "__post_init__", None)

        def wrapped_post(self, *a, **k):
            """
            Wrapper function automatically injected into subclasses.

            Initializes the LoggerMixin using the subclass's 'debug' field,
            and then calls the original `__post_init__` method (if present).
            """
            # 1 Initialize LoggerMixin with the subclass's 'debug' value
            LoggerMixin.__init__(self, debug=getattr(self, "debug", False))

            # 2 Execute subclass-specific post-initialization logic
            if orig_post is not None:
                orig_post(self, *a, **k)

        # Replace or wrap the subclass's __post_init__ method
        cls.__post_init__ = wrapped_post
