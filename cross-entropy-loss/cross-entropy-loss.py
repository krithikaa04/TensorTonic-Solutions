import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    N = y_true.shape[0]
    class_prob = y_pred[np.arange(N), y_true]
    loss = -np.mean(np.log(class_prob))
    return loss
