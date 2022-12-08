import numpy as np
from lifelines.utils import concordance_index as ci


def concordance_index(y_true, y_pred):
    """
    Compute the concordance-index value.

    Parameters
    ----------
    y_true : np.array
        Observed time. Negtive values are considered right censored.
    y_pred : np.array
        Predicted value.

    Returns
    -------
    float
        Concordance index.
    """
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)
    t = np.abs(y_true)
    e = (y_true > 0).astype(np.int32)
    ci_value = ci(t, y_pred, e)
    return ci_value