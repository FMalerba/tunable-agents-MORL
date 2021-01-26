from typing import Callable, Tuple
import numpy as np
from functools import partial


UtilityFunction = Callable[[np.ndarray], float]


def linear_utility(reward: np.ndarray, weights: np.ndarray) -> float:
    return np.dot(reward, weights)


def polinomial_utility(reward: np.ndarray, coefficients: np.ndarray) -> float:
    poly = np.expand_dims(reward, axis=1).repeat(coefficients.shape[1], axis=1) ** np.arange(coefficients.shape[1])
    return np.sum(poly * coefficients)


def threshold_utility(reward: np.ndarray, thresholds: np.ndarray) -> float:
    return np.where(reward >= thresholds, 1, 0)


def sample_linear_preference() -> np.ndarray:
    """
    Samples a 6-long vector of preferences for the gathering environment replication study.
    Each preference weight is randomly sampled between -20 and 20 in steps of 5 with the 
    exception of the first two entries which are fixed at -1 and -5. They respectively signal
    the punishment for taking a further time-step and the punishment for hitting a wall
    """
    # The 4 entries of pref are the preference for (respectively): Green, Red, Yellow, Other agent taking red
    pref = np.random.choice(np.arange(-4, 5, dtype=np.float32), size=4)*5
    w01 = np.array([-1, -5], dtype=np.float32)
    return np.concatenate((w01, pref))


def sample_thresholds() -> np.ndarray:
    """
    Samples a 6-long vector of thresholds for the gathering environment.
    Each threshold is randomly sampled 
    exception of the first two entries which are fixed at -1 and -5. They respectively signal
    the punishment for taking a further time-step and the punishment for hitting a wall.
    """
    # The 4 entries of thresholds are for (respectively): Green, Red, Yellow, Other agent taking red
    thresholds = np.random.choice(np.arange(0, 5, dtype=np.float32), size=4)
    w01 = np.array([-1, -5], dtype=np.float32)
    return np.concatenate((w01, thresholds))


def sample_utility(utility_type: str = 'linear', utility_repr: np.ndarray = None) -> Tuple[np.ndarray, UtilityFunction]:
    if utility_type == 'linear':
        if utility_repr is not None:
            # Preferences are in range [-20, 20] we normalize them to the range [-0.5, 0.5] for the agent's represantation
            return utility_repr/40, partial(linear_utility, weights=utility_repr)
        weights = sample_linear_preference()
        # Preferences are in range [-20, 20] we normalize them to the range [-0.5, 0.5] for the agent's represantation
        return weights/40, partial(linear_utility, weights=weights)
    elif utility_type == 'threshold':
        if utility_repr is not None:
            return utility_repr, partial(threshold_utility, thresholds=utility_repr)
        thresholds = sample_thresholds()
        return thresholds, partial(threshold_utility, thresholds=thresholds)
    
    raise ValueError('Expected argument "utility_type" to be a string representing a valid implemented utility')
