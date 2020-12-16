from typing import Callable
import numpy as np
from functools import partial
from scipy.special import softmax
import gin


def linear_utility(reward: np.ndarray, weights: np.ndarray) -> float:
    return np.dot(reward, weights)


def polinomial_utility(reward: np.ndarray, coefficients: np.ndarray) -> float:
    poly = np.expand_dims(reward, axis=1).repeat(coefficients.shape[1], axis=1) ** np.arange(coefficients.shape[1])
    return np.sum(poly * coefficients)


def threshold_utility(reward: np.ndarray, thresholds: np.ndarray) -> float:
    return np.where(reward >= thresholds, reward, 0)


def generate_random_weights(shape: tuple = (2,)) -> np.ndarray:
    weights = np.random.uniform(size=shape)
    return weights/np.sum(weights)


@gin.configurable
def generate_utility(func: str = 'linear', reward_shape: tuple = (2,), **kwargs) -> Callable[[np.ndarray], float]:
    if func == 'linear':
        if 'weights' in kwargs:
            return partial(linear_utility, weights=kwargs.get('weights'))
        return partial(linear_utility, weights=generate_random_weights(reward_shape))
    
    raise NotImplementedError('Function still to be done')
