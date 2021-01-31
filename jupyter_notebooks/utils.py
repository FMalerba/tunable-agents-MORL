import numpy as np
from tf_agents.utils import common
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types
from tf_agents.environments import tf_py_environment
from tf_agents.policies import tf_policy
import matplotlib.pyplot as plt
import time


def run_and_render_episode(env: tf_py_environment.TFPyEnvironment, policy: tf_policy.TFPolicy) -> None:
    time_step = env.reset()
    policy_state = policy.get_initial_state(1)
    
    plt.figure(figsize=(11, 6), dpi=200)
    i = 0
    render_time_step(time_step, i=i)
    while not time_step.is_last():
        action_step = policy.action(time_step, policy_state)
        time_step = env.step(action_step.action)
        policy_state = action_step.state
        i += 1
        render_time_step(time_step, action_step.action, i)
    
    plt.tight_layout()
    plt.show()


def render_time_step(time_step: ts.TimeStep, action: int = None, i: int = 0) -> None:
    plt.subplot(4, 8, i+1)
    image = time_step.observation['state_obs'][:, :, -3:]
    plt.imshow(image)
    plt.title('Utility: {}    Action: {}'.format(time_step.reward, action), fontsize=5)
    plt.axis('off')
    #plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    #plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)


def evaluate_average_utility(env: tf_py_environment.TFPyEnvironment, policy: tf_policy.TFPolicy, n_episodes: int) -> float:
    utilities = np.empty(shape=(n_episodes), dtype=np.float32)
    
    for i in range(n_episodes):
        time_step = env.reset()
        policy_state = policy.get_initial_state(1)
        while not time_step.is_last():
            action_step = policy.action(time_step, policy_state)
            time_step = env.step(action_step.action)
            policy_state = action_step.state
        
        utilities[i] = env._prev_step_utility
        
    
    return np.mean(utilities)


