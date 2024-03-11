import numpy as np

def compute_gradient(x, y, w, b):
    """
    Computes the gradient for linear regression

    Args:
    x (ndarray (m,)): Data, m examples
    y (ndarray (m,)): target values
    w,b (scalar) : model parameters

    Returns:
    dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
    dj_db (scalar): The gradient of the cost w.r.t. the parameter b
    """

    # Number of examples
    m = len(x)

    # Compute predictions
    y_pred = w * x + b

    # Compute the gradients
    dj_dw = (-2/m) * np.sum(x * (y - y_pred))  # Equation (4)
    dj_db = (-2/m) * np.sum(y - y_pred)         # Equation (5)

    return dj_dw, dj_db
