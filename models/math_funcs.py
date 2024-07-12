import numpy as np


def cos_sim(a, b):
    do = np.dot(a, b)
    do /= (np.linalg.norm(a) * np.linalg.norm(b))
    return do