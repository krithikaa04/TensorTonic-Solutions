import numpy as np

def kl_divergence(p, q, eps=1e-12):
    """
    Compute KL Divergence D_KL(P || Q).
    """
    p = np.array(p)
    q = np.array(q)
    p = p + eps
    q = q + eps
    summ = np.sum(p*np.log(p/q))
    return summ
