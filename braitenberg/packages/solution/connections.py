from typing import Tuple

import numpy as np

def rescale(a: float, L: float, U: float):
    if np.allclose(L, U):
        return 0.0
    return (a - L) / (U - L)


def genMatrix(shape: Tuple[int, int], modifier: int) -> np.ndarray:
    res = np.zeros(shape=shape, dtype="float32")
    w, h = shape[1], shape[0]
    for x in range(w):
        sign = modifier * (1 if x < 0.5*w else -1)
        min_val = 0.02
        dir_val = 0.1
        for y in range(h):
            v = ((x-w/2)**2)/(3*h) + 0.32*h
            if y < v and (x < 0.35*w or x > 0.65*w):
                res[y, x] = -sign * (min_val if y < v - 0.15*h else dir_val)
            elif y < v:
                res[y, x] = 0.0
            else:
                res[y, x] = sign * max(dir_val, rescale(y, v, 1.0*h + v))

    return res

def get_motor_left_matrix(shape: Tuple[int, int]) -> np.ndarray:
    res = np.zeros(shape=shape, dtype="float32")

    w, h = shape[1], shape[0]
    """
    #right side far
    res[h-int(1*h):h-int(0.55*h), int(0.87*w):int(1.00*w)] = 0.03
    res[h-int(0.55*h):h-int(0.45*h), int(0.87*w):int(1.00*w)] = -0.33
    res[h-int(0.45*h):h-int(0.3*h), int(0.87*w):int(1.00*w)] = -0.55
    res[h-int(0.30*h):h-int(0.0*h), int(0.87*w):int(1.00*w)] = -0.8

    #right side closes
    res[h-int(1*h):h-int(0.75*h), int(0.65*w):int(0.87*w)] = 0.04
    res[h-int(0.75*h):h-int(0.60*h), int(0.65*w):int(0.87*w)] = 0.05
    res[h-int(0.60*h):h-int(0.48*h), int(0.65*w):int(0.87*w)] = -0.40
    res[h-int(0.48*h):h-int(0.3*h), int(0.65*w):int(0.87*w)] = -0.75
    res[h-int(0.3*h):h-int(0.0*h), int(0.65*w):int(0.87*w)] = -0.85

    #middle column
    res[h-int(0.9*h):h-int(0.70*h), int(0.5*w):int(0.65*w)] = 0.07
    res[h-int(0.65*h):h-int(0.51*h), int(0.5*w):int(0.65*w)] = -0.4
    res[h-int(0.51*h):h-int(0.35*h), int(0.5*w):int(0.65*w)] = -0.8
    res[h-int(0.35*h):h-int(0.0*h), int(0.5*w):int(0.65*w)] = -1.0

    res[h-int(0.9*h):h-int(0.70*h), int(0.35*w):int(0.5*w)] = -0.07
    res[h-int(0.65*h):h-int(0.51*h), int(0.35*w):int(0.5*w)] = 0.4
    res[h-int(0.55*h):h-int(0.35*h), int(0.35*w):int(0.5*w)] = 0.8
    res[h-int(0.35*h):h-int(0.0*h), int(0.35*w):int(0.5*w)] = 1.0


    #left side close
    res[h-int(1*h):h-int(0.75*h), int(0.13*w):int(0.35*w)] = -0.04
    res[h-int(0.75*h):h-int(0.60*h), int(0.13*w):int(0.35*w)] = -0.05
    res[h-int(0.60*h):h-int(0.48*h), int(0.13*w):int(0.35*w)] = 0.40
    res[h-int(0.48*h):h-int(0.3*h), int(0.13*w):int(0.35*w)] = 0.75
    res[h-int(0.3*h):h-int(0.0*h), int(0.13*w):int(0.35*w)] = 0.85

    #left side far
    res[h-int(1*h):h-int(0.55*h), int(0.00*w):int(0.13*w)] = -0.03
    res[h-int(0.55*h):h-int(0.45*h), int(0.00*w):int(0.13*w)] = 0.33
    res[h-int(0.45*h):h-int(0.3*h), int(0.00*w):int(0.13*w)] = 0.55
    res[h-int(0.3*h):h-int(0.0*h), int(0.0*w):int(0.13*w)] = 0.8
    """

    res = genMatrix(shape, 1)

    return res


def get_motor_right_matrix(shape: Tuple[int, int]) -> np.ndarray:
    res = np.zeros(shape=shape, dtype="float32")

    w, h = shape[1], shape[0]
    #res = -get_motor_left_matrix(shape)
    res = genMatrix(shape, -1)
    return res