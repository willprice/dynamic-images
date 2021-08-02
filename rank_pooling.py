"""
Original author: MRzzm
Original author email: zhangzhimeng1@gmail.com
Original github: https://github.com/MRzzm/rank-pooling-python.git

Modified by Will Price to be a bit more pythonic, no material changes.
"""
import numpy as np
import scipy.sparse
from sklearn import svm


def rank_pooling(time_seq, C=1, non_linearity="ssr"):
    """
    This function only calculate the positive direction of rank pooling.
    Args:
        time_eq: :math:`(D, T)` shape ndarray
        C: SVM regularisation hyper parameter
        non_linearity: nonlinear transform (one of ``'ref'``, ``'tanh'``, ``'ssr'``, ``'none'``)
    Returns:
        Rank pooled result, ndarray of shape :math:`(D,)`
    """
    seq_smooth = _smooth_seq(time_seq)
    seq_nonlinear = _get_non_linearity(seq_smooth, non_linearity)
    seq_norm = _normalize(seq_nonlinear)
    seq_len = np.size(seq_norm, 1)
    Labels = np.array(range(1, seq_len + 1))
    seq_svr = scipy.sparse.csr_matrix(np.transpose(seq_norm))
    svr_model = svm.LinearSVR(
        epsilon=0.1,
        tol=0.001,
        C=C,
        loss="squared_epsilon_insensitive",
        fit_intercept=False,
        dual=False,
    )
    svr_model.fit(seq_svr, Labels)
    return svr_model.coef_


def _get_non_linearity(data, non_linearity="ref"):
    # we don't provide the Chi2 kernel in our code
    if non_linearity == "none":
        return data
    if non_linearity == "ref":
        return _root_expand_kernel_map(data)
    elif non_linearity == "tanh":
        return np.tanh(data)
    elif non_linearity == "ssr":
        return np.sign(data) * np.sqrt(np.fabs(data))
    else:
        raise ValueError(
            "We don't provide {} non-linear transformation".format(non_linearity)
        )


def _smooth_seq(seq):
    res = np.cumsum(seq, axis=1)
    seq_len = np.size(res, 1)
    res = res / np.expand_dims(np.linspace(1, seq_len, seq_len), 0)
    return res


def _root_expand_kernel_map(data):
    element_sign = np.sign(data)
    nonlinear_value = np.sqrt(np.fabs(data))
    return np.vstack(
        (nonlinear_value * (element_sign > 0), nonlinear_value * (element_sign < 0))
    )


def _normalize(seq, norm="l2"):
    if norm == "l2":
        seq_norm = np.linalg.norm(seq, ord=2, axis=0)
        seq_norm[seq_norm == 0] = 1
        seq_norm = seq / np.expand_dims(seq_norm, 0)
        return seq_norm
    elif norm == "l1":
        seq_norm = np.linalg.norm(seq, ord=1, axis=0)
        seq_norm[seq_norm == 0] = 1
        seq_norm = seq / np.expand_dims(seq_norm, 0)
        return seq_norm
    else:
        raise ("We only provide l1 and l2 normalization methods")
