from typing import Dict

import numpy as np


def cos_sim(a, b):
    do = np.dot(a, b)
    do /= (np.linalg.norm(a) * np.linalg.norm(b))
    return do


def max_class(computation_result: Dict) -> str:
    max_idx = np.argmax(computation_result.values())
    return list(computation_result.keys())[max_idx]
