import gin
import numpy as np
from tf_agents.trajectories import time_step as ts
from tf_agents.environments import py_environment
from tf_agents.policies import tf_policy
import matplotlib.pyplot as plt
from tqdm import tqdm


def render_time_step(time_step: ts.TimeStep, ax, action: int = None) -> None:
    image = time_step.observation['state_obs'][:, :, -3:]
    ax.imshow(image)
    ax.set_title('Utility: {}    Action: {}'.format(time_step.reward, action), fontsize=5)
    ax.axis('off')


def policy_play_episode(env: py_environment.PyEnvironment, policy: tf_policy.TFPolicy) -> None:
    time_step = env.reset()
    policy_state = policy.get_initial_state(1)
    
    plt.figure(figsize=(11, 6), dpi=200)
    i = 1
    ax = plt.subplot(4, 8, i)
    render_time_step(time_step, ax)
    while not time_step.is_last():
        action_step = policy.action(time_step, policy_state)
        time_step = env.step(action_step.action)
        policy_state = action_step.state
        i += 1
        ax = plt.subplot(4, 8, i)
        render_time_step(time_step, ax, action_step.action)
    
    plt.tight_layout()
    plt.show()


def model_play_episode(env: py_environment.PyEnvironment, model) -> None:
    time_step = env.reset()
        
    plt.figure(figsize=(11, 6), dpi=200)
    i = 1
    ax = plt.subplot(4, 8, i)
    render_time_step(time_step, ax)
    while not time_step.is_last():
        state = time_step.observation['state_obs']
        weights = time_step.observation['utility_representation']
        action = np.argmax(model.predict([state[np.newaxis], weights[np.newaxis]]))
        time_step = env.step(action)
        i += 1
        ax = plt.subplot(4, 8, i)
        render_time_step(time_step, ax, action)
    
    plt.tight_layout()
    plt.show()


def policy_evaluate_utility(env: py_environment.PyEnvironment, policy: tf_policy.TFPolicy, n_episodes: int) -> float:
    utilities = np.empty(shape=(n_episodes), dtype=np.float32)
    
    for i in tqdm(range(n_episodes)):
        time_step = env.reset()
        policy_state = policy.get_initial_state(1)
        while not time_step.is_last():
            action_step = policy.action(time_step, policy_state)
            time_step = env.step(action_step.action)
            policy_state = action_step.state
        
        utilities[i] = env._prev_step_utility
        
    
    return utilities


def model_evaluate_utility(env: py_environment.PyEnvironment, model, n_episodes: int) -> float:
    utilities = np.empty(shape=(n_episodes), dtype=np.float32)
    
    for i in tqdm(range(n_episodes)):
        time_step = env.reset()
        while not time_step.is_last():
            state = time_step.observation['state_obs']
            weights = time_step.observation['utility_representation']
            action = np.argmax(model.predict([state[np.newaxis], weights[np.newaxis]]))
            time_step = env.step(action)
            
        utilities[i] = env._prev_step_utility
        
    
    return utilities


@gin.configurable
def train_eval(training_id: str, model_id: str, env_id: str):
    """Only defined to avoid configs error for undefined func when loading"""
    pass

