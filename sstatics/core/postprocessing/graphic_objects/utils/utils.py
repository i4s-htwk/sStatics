import numpy as np


def is_clockwise(x: list[float], z: list[float]):
    x, z = np.asarray(x), np.asarray(z)
    area = np.sum((x[1:] - x[:-1]) * (z[1:] + z[:-1]))
    area += (x[0] - x[-1]) * (z[0] + z[-1])
    return area > 0


def round_value(value, decimals, sig_digits):
    if sig_digits is not None:
        value = float(f"{value:.{sig_digits}g}")
    else:
        value = round(value, decimals)
    return 0.0 if np.isclose(value, 0.0, atol=1e-8) else value
