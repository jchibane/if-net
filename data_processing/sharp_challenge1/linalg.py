import numpy as np


def squared_norm(x, axis=None, keepdims=False):
    """Squared Euclidean norm of an array."""
    return (x ** 2).sum(axis=axis, keepdims=keepdims)


def norm(x, axis=None, keepdims=False):
    """Euclidean norm of an array."""
    return np.sqrt(squared_norm(x, axis=axis, keepdims=keepdims))


def normed(x, axis=None, keepdims=False):
    """Normalize an array."""
    eps = np.finfo(x.dtype).eps
    return x / (norm(x, axis=axis, keepdims=True) + eps)
