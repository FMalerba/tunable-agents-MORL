import numpy as np


def linear_utility(reward: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.dot(reward, weights)


def polinomial_utility(reward: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    poly = np.expand_dims(reward, axis=1).repeat(coefficients.shape[1], axis=1) ** np.arange(coefficients.shape[1])
    return np.sum(poly * coefficients)


def threshold_utility(reward: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return np.where(reward >= thresholds, reward, 0)

