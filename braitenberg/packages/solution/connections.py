from typing import Tuple

import numpy as np



def get_motor_left_matrix(shape: Tuple[int, int]) -> np.ndarray:
    res = np.zeros(shape=shape, dtype="float32")

    w, h = shape[1], shape[0]
    res[h-int(0.70*h):h, 0:int(0.15*w)] = 1.0
    res[h-int(0.55*h):h-int(0.05*h), int(0.15*w):int(0.35*w)] = 1.0
    res[h-int(0.70*h):h-int(0.55*h), int(0.15*w):int(0.35*w)] = 0.5
    res[h-int(0.70*h):h-int(0.05*h), int(0.35*w):int(0.70*w)] = 1.5

    return res


def get_motor_right_matrix(shape: Tuple[int, int]) -> np.ndarray:
    res = np.zeros(shape=shape, dtype="float32")

    w, h = shape[1], shape[0]
    res[h-int(0.70*h):h, w-int(0.15*w):w] = 1.0
    res[h-int(0.55*h):h-int(0.05*h), w-int(0.35*w):w-int(0.15*w)] = 1.0
    res[h-int(0.70*h):h-int(0.55*h), w-int(0.35*w):w-int(0.15*w)] = 0.5
    
    # ---
    return res