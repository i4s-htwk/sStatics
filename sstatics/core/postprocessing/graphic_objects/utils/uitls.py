import numpy as np


def is_clockwise(x: list[float], z: list[float]):
    x, z = np.asarray(x), np.asarray(z)
    area = np.sum((x[1:] - x[:-1]) * (z[1:] + z[:-1]))
    area += (x[0] - x[-1]) * (z[0] + z[-1])
    return area > 0
