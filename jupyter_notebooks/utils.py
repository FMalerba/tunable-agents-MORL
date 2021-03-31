import gin
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from tf_agents.trajectories import time_step as ts
from tf_agents.environments import py_environment
from tf_agents.policies import tf_policy

from tqdm import tqdm
from typing import Dict, List


# LISTS THAT ARE USED FOR CONSISTENT SORTING OF ENVIRONMENTS AND MODELS
ENVS = ["replication_env", "threshold_env", "threshold_env_linear",
        "cum_rewards_env", "cum_threshold_env", "cum_threshold_env_linear"]
MODELS = ["64_64_model", "less_utils_64_64_model",
          "128_128_64_model", "less_utils_128_128_64_model",
          "256_128_128_64_64_model", "less_utils_256_128_128_64_64_model"]


def load_results(path: str) -> Dict[str, np.ndarray]:
    keys = set()
    for file in os.listdir(path):
        keys.add("-".join(file.split("-")[:2]))
    
    results = dict()
    for key in keys:
        new_key = "-".join(key.split("-")[::-1])
        results[new_key] = [np.load(path + file, allow_pickle=True) for file in os.listdir(path) if key.split("-") == file.split("-")[:2]]
    
    return results


def sorting(key: str) -> int:
    env_sorting = dict([(env, i) for i, env in enumerate(ENVS)])
    model_sorting = dict([(model, i) for i, model in enumerate(MODELS)])
    
    env, model = key.split("-")
    
    return env_sorting[env]*(max(model_sorting.values())+1) + model_sorting[model]


def utilities_table(results: Dict[str, List[np.ndarray]]) -> pd.DataFrame:
    keys = sorted(results.keys(), key=sorting)
    df = pd.DataFrame(columns=["Environment", "Model", "Utility"])
    
    for key in keys:
        if "less_utils" in key:
            continue
        env, model = key.split("-")
        res = [np.mean(result) for result in results[key]]
        mean_util = np.round(np.mean(res), 2)
        std_err = np.round(np.std(res)/np.sqrt(len(res)), 2)
        val = f"{mean_util} (+-{std_err})"
        df = df.append({"Environment": env,
                        "Model": model,
                        "Utility": val
                        },
                       ignore_index=True)
    
    df['Environment'] = pd.Categorical(df["Environment"], ENVS)
    df['Model'] = pd.Categorical(df["Model"], MODELS)
    
    return df


def fixed_env_table(results: Dict[str, List[np.ndarray]]) -> pd.DataFrame:
    keys = sorted(results.keys(), key=sorting)
    df = pd.DataFrame(columns=["Environment", "Model", "Uniques"])
    
    for key in keys:
        if "less_utils" in key:
            continue
        env, model = key.split("-")
        df = df.append({"Environment": env,
                        "Model": model,
                        "Uniques": np.round(np.mean([np.unique(result, axis=0).shape[0] for result in results[key]]), 1)
                        },
                       ignore_index=True)
    
    df['Environment'] = pd.Categorical(df["Environment"], ENVS)
    df['Model'] = pd.Categorical(df["Model"], MODELS)
    
    return df


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

