import numpy as np
from typing import Callable, Dict, Tuple

import gin

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, Dropout, Dense, Flatten, Concatenate

from collections import deque # Used for replay buffer and reward tracking

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
        indices = np.random.randint(len(self), 
                                    size=batch_size)
        # Filter the batch from the deque
        batch = [self[index] for index in indices]
        # Unpach and create numpy arrays for each element type in the batch
        states, actions, rewards, next_states, dones, weightss = [
                np.array([experience[field_index] for experience in batch])
                for field_index in range(6)]
        return states, actions, rewards, next_states, dones, weightss


@gin.configurable()
class RewardTracker:
    """
    Class for tracking mean rewards and storing all episode rewards for
    analysis.
    """
    def __init__(self, maxlen):
        self.moving_average = deque([-np.inf for _ in range(maxlen)], 
                                    maxlen=maxlen)
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
        episodes = np.array(
            [i for i in range(len(self.epsiode_rewards))]).reshape(-1,1)
        
        rewards = np.array(self.epsiode_rewards).reshape(-1,1)
        return np.concatenate((episodes, rewards), axis=1)


@gin.configurable()
class MovingAverage(deque):
    def mean(self):
        return sum(self) / len(self)


@gin.configurable()
class DQNAgent:
    
    def __init__(self, epsilon: Callable[[], float], utility_repr_shape: Tuple[int], learning_rate: float=1e-4, gamma: float=0.99):
        self._epsilon = epsilon
        self._utility_repr_shape = utility_repr_shape
        self._learning_rate = learning_rate     # Learning rate
        self._gamma = gamma     # Discount
        
        image_size = (8, 8, 3)
        self._state_obs_shape = (8, 8, 9)
        self._output_size = 5
        
        # Build both models
        self._model = self._build_model()
        self._target_model = self._build_model()
        # Make weights the same
        self._target_model.set_weights(self._model.get_weights())
        
        self._learning_plot_initialised = False
        #self.env_penalty_sign = env.penalty_sign
        
    def _build_model(self) -> keras.Model:
        """
        Construct the DQN model.
        """
        # image of size 8x8 with 3 channels (RGB)
        image_input = Input(shape=self._state_obs_shape)
        # preference weights
        utility_repr_input = Input(shape=self._utility_repr_shape) # 4 weights

        # Define Layers
        x = image_input
        x = Conv2D(256, (3, 3), activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Conv2D(256, (3, 3), activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Flatten()(x)
        x = Concatenate()([x, utility_repr_input])
        # x = Dense(128, activation='relu')(x)
        # x = Dropout(0.2)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        
        x = Dense(5)(x)
        outputs = x
        
        # Build full model
        model = keras.Model(inputs=[image_input, utility_repr_input], outputs=outputs)
        
        # Define optimizer and loss function
        self._optimizer = keras.optimizers.Adam(lr=self._learning_rate)
        self._loss_fn = keras.losses.mean_squared_error
        
        return model
    
    def update_target_model(self) -> None:
        self._target_model.set_weights(self._model.get_weights())
    
    def epsilon_greedy_policy(self, observations: Observation) -> int:
        """
        Select greedy action from model output based on current state with 
        probability epsilon. With probability 1 - epsilon select random action.
        """
        if np.random.rand() < self._epsilon():
            return np.random.choice(5)
        else:
            Q_values = self._model.predict([observations["state_obs"][np.newaxis],
                                            observations["utility_representation"][np.newaxis]])
            return np.argmax(Q_values)
    
    def greedy_policy(self, observations: Observation) -> int:
        """
        Select greedy action from model output based on current state.
        """
        Q_values = self._model.predict([observations["state_obs"][np.newaxis],
                                        observations["utility_representation"][np.newaxis]])
        return np.argmax(Q_values)
    
    def training_step(self, experiences):
        """
        Train the DQN on a batch from the replay buffer.
        Adapted from: 
            https://github.com/ageron/handson-ml2/blob/master/18_reinforcement_learning.ipynb
            [Accessed: 15/06/2020]
        """
        # Sample a batch of S A R S' from replay memory
        states, actions, rewards, next_states, dones, weightss = experiences
        
        # Compute target Q values from 'next_states'
        next_Q_values = self._target_model.predict([next_states, weightss])
        
        max_next_Q_values = np.max(next_Q_values, axis=1)
        target_Q_values = (rewards +
                       (1 - dones) * self._gamma * max_next_Q_values)
        target_Q_values = target_Q_values.reshape(-1, 1) # Make column vector
        
        # Mask to only consider action taken
        mask = tf.one_hot(actions, self._output_size) # Number of actions
        # Compute loss and gradient for predictions on 'states'
        with tf.GradientTape() as tape:
            all_Q_values = self._model([states, weightss])
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, 
                                     keepdims=True)
            loss = tf.reduce_mean(self._loss_fn(target_Q_values, Q_values))
        grads = tape.gradient(loss, self._model.trainable_variables)
        # Apply gradients
        self._optimizer.apply_gradients(zip(grads, 
                                           self._model.trainable_variables))
    
    def save_model(self, path) -> None:
        self._model.save(path)
    
    def load_model(self, path):
        self._model = keras.models.load_model(path)
        self._target_model = keras.models.clone_model(self._model)
        self._target_model.set_weights(self._model.get_weights())


