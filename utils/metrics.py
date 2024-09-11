import torch
from typing import List
import numpy as np

def uncertainty_estimation(preds:List, uc_type:str="absolute"):
        preds = [x.detach() for x in preds]
        y_stack = torch.stack(preds, dim=0)

        if uc_type == "absolute":
            # Get the number of tensors and the dimension
            n, N, D = y_stack.shape
            
            # Expand tensors to compute pairwise differences
            expanded_tensors_1 = y_stack.unsqueeze(0).expand(n, n, N, D)
            expanded_tensors_2 = y_stack.unsqueeze(1).expand(n, n, N, D)

            # Compute pairwise differences
            pairwise_differences = expanded_tensors_1 - expanded_tensors_2

            # Compute the norms along each dimension
            pairwise_norms = torch.abs(pairwise_differences)
            
            # Exclude self-pairs (diagonal elements) by creating a mask
            mask = torch.ones(n, n) - torch.eye(n)
            
            # Apply the mask to the pairwise norms
            masked_pairwise_norms = pairwise_norms * mask.unsqueeze(-1).unsqueeze(-1)

            # Compute the mean of the non-zero elements along each dimension
            return masked_pairwise_norms.sum(dim=(0, 1)) / mask.sum()
        

def AULC(accs, uncertainties):
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
