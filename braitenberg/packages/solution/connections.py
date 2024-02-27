from typing import Tuple

import numpy as np

def rescale(a: float, L: float, U: float):
    if np.allclose(L, U):
        return 0.0
    return (a - L) / (U - L)


def genMatrix(shape: Tuple[int, int], modifier: int) -> np.ndarray:
    res = np.zeros(shape=shape, dtype="float32")
    w, h = shape[1], shape[0]
    min_val = 0.2
    offset_vert = 0.35 * h
    slope = 2.5 * h
    for x in range(w):
        sign = modifier * (1 if x < 0.5*w else -1)
        for y in range(h):
            v = ((x-w/2)**2) / slope + offset_vert
            if y < v:
                res[y, x] = 0.0
            else:
                res[y, x] = sign * max(min_val, rescale(y, v, 1.0*h + v))

    return res

def get_motor_left_matrix(shape: Tuple[int, int]) -> np.ndarray:
    res = np.zeros(shape=shape, dtype="float32")

    w, h = shape[1], shape[0]
    res = genMatrix(shape, 1)

    return res


def get_motor_right_matrix(shape: Tuple[int, int]) -> np.ndarray:
    res = np.zeros(shape=shape, dtype="float32")

    w, h = shape[1], shape[0]
    res = genMatrix(shape, -1)
    return res