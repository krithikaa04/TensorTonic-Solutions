import numpy as np
def cosine_embedding_loss(x1, x2, label, margin):
    """
    Compute cosine embedding loss for a pair of vectors.
    """
    x1 = np.array(x1)
    x2 = np.array(x2)

    cos_sim = np.dot(x1,x2)/(np.linalg.norm(x1) * np.linalg.norm(x2))

    if label==1:
        return 1-cos_sim
    return max(0,cos_sim-margin)