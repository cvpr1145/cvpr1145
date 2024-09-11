import torch
from typing import List
import numpy as np


def AULC(accs, uncertainties):
    # copied from: https://github.com/cvlab-epfl/zigzag/blob/main/exps/notebooks/mnist_classification.ipynb 
    idxs = np.argsort(uncertainties)
    uncs_s = uncertainties[idxs]
    error_s = accs[idxs]

    mean_error = error_s.mean()
    error_csum = np.cumsum(error_s)

    Fs = error_csum / np.arange(1, len(error_s) + 1)
    s = 1 / len(Fs)
    return -1 + s * Fs.sum() / mean_error, Fs

def rAULC(uncertainties, accs):
    perf_aulc, Fsp = AULC(accs, -accs.astype("float"))
    curr_aulc, Fsc = AULC(accs, uncertainties)
    return curr_aulc / perf_aulc, Fsp, Fsc
