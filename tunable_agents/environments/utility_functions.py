from abc import ABC, abstractmethod
import numpy as np
from typing import Optional


class UtilityFunction(ABC):

    def __init__(self, utility_repr: np.ndarray, agent_utility_repr: np.ndarray,
                 interests: np.ndarray) -> None:
        """Abstarct class for utility functions to be used in the Gathering environment

        Args:
            utility_repr (np.ndarray): Complete utility representation.
            agent_utility_repr (np.ndarray): Utiilty representation to be fed as input to the agent
                    usually, this will be a flattened version of utility_repr with any unchanging entries
                    removed (e.g. entries for unchanging weights of time and wall penalties).
            interests (np.ndarray): Representation of interests of the agent. This will be used by the environment
                    to determine the end of an episode.
        """
        self._utility_repr = utility_repr
        self._agent_utility_repr = agent_utility_repr
        self._interests = interests

    def __call__(self, rewards: np.ndarray) -> np.ndarray:
        return self.call(rewards)

    @abstractmethod
    def call(self, rewards: np.ndarray) -> np.ndarray:
        """Implements the utility function"""

    @property
    def utility_repr(self) -> np.ndarray:
        return self._utility_repr

    @property
    def agent_utility_repr(self) -> np.ndarray:
        return self._agent_utility_repr

    @property
    def interests(self) -> np.ndarray:
        return self._interests


class LinearUtility(UtilityFunction):

    def __init__(self,
                 weights: Optional[np.ndarray] = None,
                 agent_utility_repr: Optional[np.ndarray] = None) -> None:
        """
        Implements and contains all information and possible different representations of a linear utility function.
        
        At least one of the first two optional arguments must be provided in order to reconstruct the others.
        """
        if weights is not None:
            agent_utility_repr = weights[2:] / 40
        elif agent_utility_repr is not None:
            weights = np.concatenate(([-1, -5], agent_utility_repr * 40)).astype(np.float32)
        else:
            raise ValueError("Expected to receive at least one required argument, but received none.")

        self._weights = weights
        interests = (weights > 0).astype(int)
        super().__init__(utility_repr=weights, agent_utility_repr=agent_utility_repr, interests=interests)

    def call(self, rewards: np.ndarray) -> np.ndarray:
        return np.dot(rewards, self._weights)


class ThresholdUtility(UtilityFunction):

    def __init__(self,
                 thresholds_and_ceofficients: Optional[np.ndarray] = None,
                 agent_utility_repr: Optional[np.ndarray] = None) -> None:
        """
        Implements and contains all information and possible different representations of a threshold utility function.
        
        It is required that either thresholds_and_ceofficients or the agent_utility_repr be provided 
        in order to reconstruct the others.
        """
        if thresholds_and_ceofficients is not None:
            thresholds = thresholds_and_ceofficients[0]
            coefficients = thresholds_and_ceofficients[1]
            agent_utility_repr = (thresholds_and_ceofficients[:, 2:] / [[1], [40]]).flatten()
        elif agent_utility_repr is not None:
            thresholds = np.concatenate(([0, 0], agent_utility_repr[:4])).astype(np.float32)
            coefficients = np.concatenate(([-1, -5], agent_utility_repr[4:] * 40)).astype(np.float32)
        else:
            raise ValueError(
                "Expected to receive at least one utility representation argument, but received none.")

        self._thresholds = thresholds
        self._coefficients = coefficients
        interests = (coefficients > 0).astype(int)
        super().__init__(utility_repr=np.array([thresholds, coefficients]),
                         agent_utility_repr=agent_utility_repr,
                         interests=interests)

    def call(self, rewards: np.ndarray) -> np.ndarray:
        return np.sum(np.where(rewards >= self._thresholds, rewards * self._coefficients, 0))


class TargetUtility(UtilityFunction):

    def __init__(self,
                 target: Optional[np.ndarray] = None,
                 agent_utility_repr: Optional[np.ndarray] = None) -> None:
        """
        Implements and contains all information and possible different representations of a utility function.
        
        It is required that either the target is provided or the agent_utility_repr in order to reconstruct the others.
        """
        if target is not None:
            agent_utility_repr = target
        elif agent_utility_repr is not None:
            target = agent_utility_repr
        else:
            raise ValueError("Expected to receive at least one required argument, but received none.")

        self._target = target/np.linalg.norm(target)
        interests = target
        super().__init__(utility_repr=target, agent_utility_repr=agent_utility_repr, interests=interests)

    def call(self, rewards: np.ndarray) -> np.ndarray:
        # The reward vector is rescaled according to the specs for the Gathering environment.
        # The first two entries are for time and wall penalties, and as such get multiplied by -1
        # and subsequently translated so that they are always >= 0.
        # Episode length is 31 (this is due to unexpected and poorly executed behaviour for the underlying
        # MOGridworld class) and time and wall penalties are 1 per time step, so the translation ought to be
        # of 31.
        return np.min((rewards * [-1, -1, 1, 1, 1, 1] + [31, 31, 0, 0, 0, 0]) / self._target)


def sample_linear_weights() -> np.ndarray:
    """
    Samples a 6-long vector of preferences for the gathering environment replication study.
    Each preference weight is randomly sampled between -20 and 20 in steps of 5 with the 
    exception of the first two entries which are fixed at -1 and -5. They respectively signal
    the punishment for taking a further time-step and the punishment for hitting a wall
    """
    # The 4 entries of pref are the preference for (respectively): Green, Red, Yellow, Other agent taking red
    pref = np.random.choice(np.arange(-20, 21, step=5, dtype=np.float32), size=4)
    # An environment with a negative preference vector will simply stop the episode after the first step.
    # It is therefore pointless to sample such a vector.
    if np.all(pref <= 0):
        return sample_linear_weights()

    w01 = np.array([-1, -5], dtype=np.float32)
    return np.concatenate((w01, pref))


def sample_thresholds_and_coefficients() -> np.ndarray:
    """
    Samples a 6-long vector of thresholds for the gathering environment and a 6-long 
    vector of coefficients to be applied once the thresholds are exceeded.
    All thresholds and coefficients are sampled at random with the exception of the first two entries
    which are fixed at a threshold of 0 and coefficients of  -1 and -5. They respectively signal
    the punishment for taking a further time-step and the punishment for hitting a wall.
    """
    # The 4 entries of thresholds and coefficients are for (respectively): Green, Red, Yellow, Other agent taking red
    thresholds = np.random.choice(np.arange(0, 4, dtype=np.float32), size=4)
    coefficients = np.random.choice(np.arange(-20, 21, step=5, dtype=np.float32), size=4)

    # An environment with a negative preference vector will simply stop the episode after the first step.
    # It is therefore pointless to sample such a vector.
    if np.all(coefficients <= 0):
        return sample_thresholds_and_coefficients()

    w01 = np.array([-1, -5], dtype=np.float32)
    return np.array(
        [np.concatenate(([0, 0], thresholds)).astype(np.float32),
         np.concatenate((w01, coefficients))])


def sample_linear_thresholds() -> np.ndarray:
    """
    Samples a 6-long vector of thresholds for the gathering environment and a 6-long 
    vector of coefficients to be applied once the thresholds are exceeded.
    The sampled vectors are selected so as to be equivalent to a linear utility sampling (i.e. the 
    thresholds are all set to 0).
    The coefficients are sampled at random with the exception of the first two entries
    which are fixed at -1 and -5. They respectively signal the punishment for taking a
    further time-step and the punishment for hitting a wall.
    """
    # The 4 entries of thresholds and coefficients are for (respectively): Green, Red, Yellow, Other agent taking red
    coefficients = np.random.choice(np.arange(-20, 21, step=5, dtype=np.float32), size=4)

    # An environment with a negative preference vector will simply stop the episode after the first step.
    # It is therefore pointless to sample such a vector.
    if np.all(coefficients <= 0):
        return sample_thresholds_and_coefficients()

    w01 = np.array([-1, -5], dtype=np.float32)
    return np.array([[0, 0, 0, 0, 0, 0], np.concatenate((w01, coefficients))], dtype=np.float32)


def sample_target() -> np.ndarray:
    """
    Samples a 6-long vector of targets for the gathering environment.
    Each preference weight is randomly sampled between -20 and 20 in steps of 5 with the 
    exception of the first two entries which are fixed at -1 and -5. They respectively signal
    the punishment for taking a further time-step and the punishment for hitting a wall

    Returns:
        target: The target reward vector for the agent to achieve in the gridworld environment
    """
    targets = np.random.choice(np.arange(3), size=4)
    if np.all(targets == 0):
        return sample_target()
    return np.concatenate([[31, 31], targets])


def sample_utility(utility_type: str = 'linear',
                   utility_repr: Optional[np.ndarray] = None) -> UtilityFunction:
    """
    Samples a UtilityFunction of the required utility_type (or constructs one if a utility_repr is provided).
    
    Returns a 3-tuple with the utility's representation for the agent, for the environment and a partial version 
            of the utility function 
    """
    if utility_type == 'linear':
        if utility_repr is not None:
            return LinearUtility(weights=utility_repr)
        weights = sample_linear_weights()
        return LinearUtility(weights=weights)
    elif utility_type == 'threshold':
        if utility_repr is not None:
            return ThresholdUtility(thresholds_and_ceofficients=utility_repr)
        thresholds_and_coefficients = sample_thresholds_and_coefficients()
        return ThresholdUtility(thresholds_and_ceofficients=thresholds_and_coefficients)
    elif utility_type == "linear_threshold":
        if utility_repr is not None:
            return ThresholdUtility(thresholds_and_ceofficients=utility_repr)
        thresholds_and_coefficients = sample_linear_thresholds()
        return ThresholdUtility(thresholds_and_ceofficients=thresholds_and_coefficients)
    elif utility_type == "target":
        if utility_repr is not None:
            return TargetUtility(target=utility_repr)
        target = sample_target()
        return TargetUtility(target=target)

    raise ValueError(
        'Expected argument "utility_type" to be a string representing a valid implemented utility')
