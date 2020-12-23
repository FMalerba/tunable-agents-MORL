from typing import Callable
import numpy as np
from functools import partial
import gin


def linear_utility(reward: np.ndarray, weights: np.ndarray) -> float:
    return np.dot(reward, weights)


def polinomial_utility(reward: np.ndarray, coefficients: np.ndarray) -> float:
    poly = np.expand_dims(reward, axis=1).repeat(coefficients.shape[1],
                                                 axis=1)**np.arange(coefficients.shape[1])
    return np.sum(poly * coefficients)


def threshold_utility(reward: np.ndarray, thresholds: np.ndarray) -> float:
    return np.where(reward >= thresholds, reward, 0)


def generate_random_weights(shape: tuple = (2,)) -> np.ndarray:
    weights = np.random.uniform(size=shape)
    return weights / np.sum(weights)


def sample_preference() -> np.ndarray:
    """
    Samples a 6-long vector of preferences for the gathering environment replication study.
    Each preference weight is randomly sampled between -20 and 20 in steps of 5 with the 
    exception of the first two entries which are fixed at -1 and -5. They respectively signal
    the punishment for taking a further time-step and the punishment for hitting a wall
    """
    # The 4 entries of pref are the preference for (respectively): Green, Red, Yellow, Other agent taking red
    pref = np.random.choice(np.arange(-4, 5, dtype=np.float32), size=4) * 5
    w01 = np.array([-1, -5], dtype=np.float32)
    return np.concatenate((w01, pref))


@gin.configurable
def generate_utility(func: str = 'linear',
                     reward_shape: tuple = (2,),
                     **kwargs) -> Callable[[np.ndarray], float]:
    if func == 'linear':
        if 'weights' in kwargs:
            return partial(linear_utility, weights=kwargs.get('weights'))
        return partial(linear_utility, weights=generate_random_weights(reward_shape))

    raise NotImplementedError('Function still to be done')
