"""
Contains various functions for computing statistics over 3D volumes
"""
import numpy as np


def Dice3d(a, b):
    """
    Dice Similarity coefficient for two 3-dimensional volumes.
    Binary masks: 0 = background, anything else = foreground.
    """
    if len(a.shape) != 3 or len(b.shape) != 3:
        raise Exception(f"Expecting 3 dimensional inputs, got {a.shape} and {b.shape}")
    if a.shape != b.shape:
        raise Exception(f"Expecting inputs of the same shape, got {a.shape} and {b.shape}")

    a = (a > 0).astype(np.float32)
    b = (b > 0).astype(np.float32)
    intersection = np.sum(a * b)
    return (2.0 * intersection) / (np.sum(a) + np.sum(b) + 1e-8)


def Jaccard3d(a, b):
    """
    Jaccard Similarity coefficient for two 3-dimensional volumes.
    """
    if len(a.shape) != 3 or len(b.shape) != 3:
        raise Exception(f"Expecting 3 dimensional inputs, got {a.shape} and {b.shape}")
    if a.shape != b.shape:
        raise Exception(f"Expecting inputs of the same shape, got {a.shape} and {b.shape}")

    a = (a > 0).astype(np.float32)
    b = (b > 0).astype(np.float32)
    intersection = np.sum(a * b)
    union = np.sum(a) + np.sum(b) - intersection
    return intersection / (union + 1e-8)
