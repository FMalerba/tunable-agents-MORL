import gin
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import pdist

from tf_agents.trajectories import time_step as ts
from tf_agents.environments import py_environment
from tunable_agents.agent import DQNAgent

from typing import Dict, List, Tuple

# LISTS THAT ARE USED FOR CONSISTENT SORTING OF ENVIRONMENTS AND MODELS
ENVS = [
    "replication_env", "target_env", "threshold_env", "threshold_env_linear", "cum_rewards_env",
    "cum_target_env", "cum_threshold_env", "cum_threshold_env_linear"
]
MODELS = [
    "64_64_model", "less_utils_64_64_model", "128_128_64_model", "less_utils_128_128_64_model",
    "256_128_128_64_64_model", "less_utils_256_128_128_64_64_model", "512_256_256_128_128_64_model"
]


def load_results(path: str) -> Dict[str, np.ndarray]:
    keys = set()
    for file in os.listdir(path):
        keys.add("-".join(file.split("-")[:2]))

    results = dict()
    for key in keys:
        new_key = "-".join(key.split("-")[::-1])
        results[new_key] = [
            np.load(path + file, allow_pickle=True)
            for file in os.listdir(path)
            if key.split("-") == file.split("-")[:2]
        ]

    return results


def sorting(key: str) -> int:
    env_sorting = dict([(env, i) for i, env in enumerate(ENVS)])
    model_sorting = dict([(model, i) for i, model in enumerate(MODELS)])

    env, model = key.split("-")

    return env_sorting[env] * (max(model_sorting.values()) + 1) + model_sorting[model]


def domination_metric(u: np.ndarray, v: np.ndarray) -> float:
    """Used to compute whether the reward vector u dominates v, viceversa, or neither.
    Do note that this is not a metric in the mathematical sense of the word, but this 
    is not a hinderance to its use in the context in which it is used.

    Args:
        u (np.ndarray): Cumulative reward vector at the end of one episode
        v (np.ndarray): Cumulative reward vector at the end of one episode

    Returns:
        float: Float representing the domination relationship between the two vectors.
    """
    if np.all(u >= v):
        return 1
    elif np.all(u <= v):
        return -1
    return 0


def wds_tables(results: Dict[str, List[np.ndarray]]) -> Tuple[pd.DataFrame]:
    keys = sorted(results.keys(), key=sorting)
    intra_wds = pd.DataFrame(columns=["Environment", "Model", "WD"])
    inter_wds = pd.DataFrame(columns=["Environment", "Models", "WD"])

    for key in keys:
        env, model = key.split("-")
        wds = [
            wasserstein_distance(results[key][i], results[key][j])
            for i in range(len(results[key]))
            for j in range(i + 1, len(results[key]))
        ]
        mean_wd = np.round(np.mean(wds), 2)
        std_err = np.round(np.std(wds) / np.sqrt(len(wds)), 2)
        val = f"{mean_wd} (+-{std_err})"
        intra_wds = intra_wds.append({"Environment": env, "Model": model, "WD": val}, ignore_index=True)

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            key_1, key_2 = keys[i], keys[j]
            if key_1.split("-")[0] == key_2.split("-")[0]:
                env, model_1 = key_1.split("-")
                model_2 = key_2.split("-")[1]
                wds = [
                    wasserstein_distance(result_1, result_2)
                    for result_1 in results[key_1]
                    for result_2 in results[key_2]
                ]
                mean_wd = np.round(np.mean(wds), 2)
                std_err = np.round(np.std(wds) / np.sqrt(len(wds)), 2)
                val = f"{mean_wd} (+-{std_err})"
                inter_wds = inter_wds.append(
                    {
                        "Environment": env,
                        "Models": "-".join([model_1, model_2]),
                        "WD": val
                    },
                    ignore_index=True)

    intra_wds['Environment'] = pd.Categorical(intra_wds["Environment"], ENVS)
    intra_wds['Model'] = pd.Categorical(intra_wds["Model"], MODELS)
    inter_wds['Environment'] = pd.Categorical(inter_wds["Environment"], ENVS)

    return intra_wds, inter_wds


def utilities_table(results: Dict[str, List[np.ndarray]]) -> pd.DataFrame:
    keys = sorted(results.keys(), key=sorting)
    df = pd.DataFrame(columns=["Environment", "Model", "Utility"])

    for key in keys:
        if "less_utils" in key:
            continue
        env, model = key.split("-")
        res = [np.mean(result) for result in results[key]]
        mean_util = np.round(np.mean(res), 2)
        std_err = np.round(np.std(res) / np.sqrt(len(res)), 2)
        val = f"{mean_util} (+-{std_err})"
        df = df.append({"Environment": env, "Model": model, "Utility": val}, ignore_index=True)

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
        df = df.append(
            {
                "Environment":
                    env,
                "Model":
                    model,
                "Uniques":
                    np.round(np.mean([np.unique(result, axis=0).shape[0] for result in results[key]]), 1)
            },
            ignore_index=True)

    df['Environment'] = pd.Categorical(df["Environment"], ENVS)
    df['Model'] = pd.Categorical(df["Model"], MODELS)

    return df


def non_dominated_table(results: Dict[str, List[np.ndarray]]) -> pd.DataFrame:
    keys = sorted(results.keys(), key=sorting)
    df = pd.DataFrame(columns=["Environment", "Model", "Non-Dominated"])

    for key in keys:
        if "less_utils" in key:
            continue
        env, model = key.split("-")

        # Cumulative reward vectors are transformed to represent the fact that the first two entries
        # (time and wall penalties) are rewards to be minimized
        uniques = [np.unique(result, axis=0) * [-1, -1, 1, 1, 1, 1] for result in results[key]]
        domination_matrices = [pdist(unique, domination_metric) for unique in uniques]
        non_dominated = []
        for index, domination_matrix in enumerate(domination_matrices):
            m = uniques[index].shape[0]
            is_non_dominated = np.ones(shape=(m,), dtype=bool)
            for i in range(m):
                start_range = m*i + i+1 - ((i + 2) * (i + 1)) // 2
                end_range = m*i + m-1 - ((i + 2) * (i + 1)) // 2
                i_range = domination_matrix[start_range:end_range]
                is_non_dominated[i] = is_non_dominated[i] and np.all(i_range != -1)
                is_non_dominated[i + np.argwhere(i_range == 1)]
            
            non_dominated.append(np.sum(is_non_dominated))

        df = df.append({
            "Environment": env,
            "Model": model,
            "Non-Dominated": np.round(np.mean(non_dominated), 1)
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


def agent_play_episode(env: py_environment.PyEnvironment, agent: DQNAgent) -> None:
    time_step = env.reset()

    plt.figure(figsize=(11, 6), dpi=200)
    i = 1
    ax = plt.subplot(4, 8, i)
    render_time_step(time_step, ax)
    while not time_step.is_last():
        action = agent.greedy_policy(time_step.observation)
        time_step = env.step(action)
        i += 1
        ax = plt.subplot(4, 8, i)
        render_time_step(time_step, ax, action)

    plt.tight_layout()
    plt.show()


@gin.configurable
def train_eval(training_id: str, model_id: str, env_id: str):
    """Only defined to avoid configs error for undefined func when loading"""
    pass
