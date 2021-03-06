import numpy as np
from typing import Callable, Dict, List

import gin

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Concatenate      #pylint: disable=no-name-in-module
from tf_agents.typing.types import ArraySpec
from collections import deque  # Used for replay buffer and reward tracking

Observation = Dict[str, np.ndarray]


class ReplayMemory(deque):
    """
    Inherits from the 'deque' class to add a method called 'sample' for 
    sampling batches from the deque.
    """

    def sample(self, batch_size):
        """
        Sample a minibatch from the replay buffer.
        """
        # Random sample of indices
        indices = np.random.randint(len(self), size=batch_size)
        # Filter the batch from the deque
        batch = [self[index] for index in indices]
        # Unpach and create numpy arrays for each element type in the batch
        states, actions, rewards, next_states, dones = [
            np.array([experience[field_index] for experience in batch]) for field_index in range(5)
        ]
        states = tf.nest.map_structure(stack_wrapper, *states)
        next_states = tf.nest.map_structure(stack_wrapper, *next_states)
        return states, actions, rewards, next_states, dones


def stack_wrapper(*structure):
    return np.stack(structure, axis=0)


@gin.configurable()
class RewardTracker:
    """
    Class for tracking mean rewards and storing all episode rewards for
    analysis.
    """

    def __init__(self, maxlen):
        self.moving_average = deque([-np.inf for _ in range(maxlen)], maxlen=maxlen)
        self.maxlen = maxlen
        self.epsiode_rewards = deque()

    def __repr__(self):
        # For printing
        return self.moving_average.__repr__()

    def append(self, reward):
        self.moving_average.append(reward)
        self.epsiode_rewards.append(reward)

    def mean(self):
        return sum(self.moving_average) / self.maxlen

    def get_reward_data(self):
        episodes = np.array([i for i in range(len(self.epsiode_rewards))]).reshape(-1, 1)

        rewards = np.array(self.epsiode_rewards).reshape(-1, 1)
        return np.concatenate((episodes, rewards), axis=1)


@gin.configurable()
class MovingAverage(deque):

    def mean(self):
        return sum(self) / len(self)


@gin.configurable()
class DQNAgent:

    def __init__(self,
                 epsilon: Callable[[], float],
                 obs_spec: Dict[str, ArraySpec],
                 learning_rate: float = 1e-4,
                 gamma: float = 1,
                 **build_model_kwargs):
        """Creates a DQN Agent with a near-arbitrary underlying model.

        Args:
            epsilon (Callable[[], float]): Epsilon to be used for the epsilon-greedy policy.
                        Note that this should be a callable that takes no argument and returns the epsilon
                        to be used at that point in time. (a callable that receives no argument could still
                        modify it's output via persistent objects like Tensors).
            obs_spec (Dict[str, ArraySpec]): Observation spec used to build the various inputs to the model.
            learning_rate (float, optional): Learning rate for the Adam optimizer. Defaults to 1e-4.
            gamma (float, optional): Gamma value to be used for discounted rewards. Defaults to 1.
        """
        self._epsilon = epsilon
        self._obs_spec = obs_spec
        self._learning_rate = learning_rate  # Learning rate
        self._gamma = gamma  # Discount
        self._output_size = 5

        # Build both models
        self._model = self._build_model(**build_model_kwargs)
        self._target_model = self._build_model(**build_model_kwargs)
        # Make weights the same
        self._target_model.set_weights(self._model.get_weights())

        self._learning_plot_initialised = False
        #self.env_penalty_sign = env.penalty_sign

    def _build_model(self, image_preprocessing_layers: List[keras.layers.Layer],
                     postprocessing_layers: List[keras.layers.Layer]) -> keras.Model:
        """
        Construct the DQN model.
        
        Args:
            image_preprocessing_layers: List of Keras layers to be applied to the image input coming from the gathering environment
            postprocessing_layers: List of Keras layers to be applied on the concatenation of the flattened image encodings and 
                    any other input from the environment.
                    
        The arguments are provided via Gin Config usually.
        """
        inputs = [
            Input(shape=self._obs_spec[key].shape, name=key)
            for key in sorted(self._obs_spec.keys())
        ]
        state_obs_index = sorted(self._obs_spec.keys()).index("state_obs")
        x = inputs[state_obs_index]
        for layer in image_preprocessing_layers:
            x = layer(x)

        x = Concatenate()([x] + inputs[:state_obs_index] + inputs[state_obs_index+1:])

        for layer in postprocessing_layers:
            x = layer(x)

        outputs = x

        model = keras.Model(inputs=inputs, outputs=outputs)

        # Define optimizer and loss function
        self._optimizer = keras.optimizers.Adam(learning_rate=self._learning_rate)
        self._loss_fn = keras.losses.mean_squared_error

        return model

    def update_target_model(self) -> None:
        self._target_model.set_weights(self._model.get_weights())

    def epsilon_greedy_policy(self, observations: Observation, training: bool=False) -> int:
        """
        Select greedy action from model output based on current state with 
        probability epsilon. With probability 1 - epsilon select random action.
        """
        if np.random.rand() < self._epsilon():
            return np.random.choice(self._output_size)
        else:
            return self.greedy_policy(observations=observations, training=training)

    def greedy_policy(self, observations: Observation, training: bool=False) -> int:
        """
        Select greedy action from model output based on current state.
        """
        Q_values = self._model([observations[key][np.newaxis] for key in sorted(observations.keys())],
                               training=training)
        return np.argmax(Q_values)

    def training_step(self, experiences):
        """
        Train the DQN on a batch from the replay buffer.
        Adapted from: 
            https://github.com/ageron/handson-ml2/blob/master/18_reinforcement_learning.ipynb
            [Accessed: 15/06/2020]
        """
        # Sample a batch of S A R S' from replay memory
        states, actions, rewards, next_states, dones = experiences

        # Compute target Q values from 'next_states'
        next_Q_values = self._target_model([next_states[key] for key in sorted(next_states.keys())])

        max_next_Q_values = np.max(next_Q_values, axis=1)
        target_Q_values = (rewards + (1 - dones) * self._gamma * max_next_Q_values)
        target_Q_values = target_Q_values.reshape(-1, 1)  # Make column vector

        # Mask to only consider action taken
        mask = tf.one_hot(actions, self._output_size)  #pylint: disable=no-value-for-parameter
        # Compute loss and gradient for predictions on 'states'
        with tf.GradientTape() as tape:
            all_Q_values = self._model([states[key] for key in sorted(states.keys())],
                                       training=True)
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
            loss = tf.reduce_mean(self._loss_fn(target_Q_values, Q_values))
        grads = tape.gradient(loss, self._model.trainable_variables)
        # Apply gradients
        self._optimizer.apply_gradients(zip(grads, self._model.trainable_variables))

    def save_model(self, path) -> None:
        self._model.save(path)

    def load_model(self, path):
        self._model = keras.models.load_model(path)
        self._target_model = keras.models.clone_model(self._model)
        self._target_model.set_weights(self._model.get_weights())
