from typing import Callable, Tuple, Optional
import numpy as np
from functools import partial


UtilityFunction = Callable[[np.ndarray], float]


def linear_utility(reward: np.ndarray, weights: np.ndarray) -> float:
    return np.dot(reward, weights)


def polinomial_utility(reward: np.ndarray, coefficients: np.ndarray) -> float:
    poly = np.expand_dims(reward, axis=1).repeat(coefficients.shape[1], axis=1) ** np.arange(coefficients.shape[1])
    return np.sum(poly * coefficients)


def threshold_utility(reward: np.ndarray, thresholds: np.ndarray, coefficients: np.ndarray) -> float:
    return np.sum(np.where(reward >= thresholds, reward*coefficients, 0))


def sample_linear_preference() -> np.ndarray:
    """
    Samples a 6-long vector of preferences for the gathering environment replication study.
    Each preference weight is randomly sampled between -20 and 20 in steps of 5 with the 
    exception of the first two entries which are fixed at -1 and -5. They respectively signal
    the punishment for taking a further time-step and the punishment for hitting a wall
    """
    # The 4 entries of pref are the preference for (respectively): Green, Red, Yellow, Other agent taking red
    pref = np.random.choice(np.arange(-4, 5, dtype=np.float32), size=4)*5
    # An environment with a negative preference vector will simply stop the episode after the first step.
    # It is therefore pointless to sample such a vector.
    if np.all(pref <= 0):
        return sample_linear_preference()
    
    w01 = np.array([-1, -5], dtype=np.float32)
    return np.concatenate((w01, pref))


def sample_thresholds() -> Tuple[np.ndarray, np.ndarray]:
    """
    Samples a 6-long vector of thresholds for the gathering environment and a 6-long 
    vector of coefficients to be applied once the thresholds are exceeded.
    All thresholds and coefficients are sampled at random with the exception of the first two entries
    which are fixed at a threshold of 0 and coefficients of  -1 and -5. They respectively signal
    the punishment for taking a further time-step and the punishment for hitting a wall.
    """
    # The 4 entries of thresholds and coefficients are for (respectively): Green, Red, Yellow, Other agent taking red
    thresholds = np.random.choice(np.arange(0, 4, dtype=np.float32), size=4)
    coefficients = np.random.choice(np.arange(-4, 5, dtype=np.float32), size=4)*5
    
    # An environment with a negative preference vector will simply stop the episode after the first step.
    # It is therefore pointless to sample such a vector.
    if np.all(coefficients <= 0):
        return sample_thresholds()
    
    w01 = np.array([-1, -5], dtype=np.float32)
    return np.concatenate(([0, 0], thresholds)).astype(np.float32), np.concatenate((w01, coefficients))


def sample_utility(utility_type: str = 'linear', utility_repr: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, UtilityFunction]:
    """
    Samples a utility function of the required utility_type (or constructs one if a utility_repr is provided).
    
    Returns a 3-tuple with the utility's representation for the agent, for the environment and a partial version 
            of the utility function 
    """
    if utility_type == 'linear':
        if utility_repr is not None:
            # Preferences are in range [-20, 20] we normalize them to the range [-0.5, 0.5] for the agent's represantation
            return utility_repr/40, utility_repr, partial(linear_utility, weights=utility_repr)
        weights = sample_linear_preference()
        # Preferences are in range [-20, 20] we normalize them to the range [-0.5, 0.5] for the agent's represantation
        return weights[2:]/40, weights, partial(linear_utility, weights=weights)
    elif utility_type == 'threshold':
        if utility_repr is not None:
            return (utility_repr[:,2:]/[[1],[40]]).flatten(), utility_repr[1], partial(threshold_utility, thresholds=utility_repr[0], coefficients=utility_repr[1])
        thresholds, coefficients = sample_thresholds()
        return (np.array([thresholds, coefficients/40])[:, 2:]).flatten(), coefficients, partial(threshold_utility, thresholds=thresholds, coefficients=coefficients)
    
    raise ValueError('Expected argument "utility_type" to be a string representing a valid implemented utility')
