import numpy as np

def mean_squared_error(y_pred, y_true):
    """
    Returns: float MSE
    """
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    sq_diff = (y_pred - y_true)**2

    mse = np.mean(sq_diff)
    return mse

    pass
