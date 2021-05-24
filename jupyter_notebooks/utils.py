import gin
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import pdist

from tf_agents.trajectories import time_step as ts
from tf_agents.environments import py_environment
from tunable_agents.agent import DQNAgent
from tunable_agents.environments.utility_functions import DualThresholdUtility, LinearUtility, TargetUtility, ThresholdUtility, UtilityFunction

from tqdm import tqdm
from typing import Dict, List, Tuple

UTILITIES_DICT = {
    "dual_threshold": "Dual Threshold",
    "linear": "Linear",
    "linear_dual_threshold": "Linear Dual Threshold",
    "linear_threshold": "Linear Threshold",
    "target": "Target",
    "threshold": "Threshold"
}
SAMPLINGS_DICT = {"": "", "dense": "Dense ", "continuous": "Continuous "}
# LISTS THAT ARE USED FOR CONSISTENT SORTING OF ENVIRONMENTS AND MODELS
LINEAR_THRESHOLDS = [False, True]
UTILITIES = ["linear", "threshold", "linear_threshold", "dual_threshold", "linear_dual_threshold", "target"]
SAMPLINGS = ["", "dense", "continuous"]
CUM_ENVS = [False, True]
ENVS = [
    "{}{}{}{}".format("Cumulative Rewards " * cum_env, SAMPLINGS_DICT[sampling], "Linear to " * lin_thresh,
                      UTILITIES_DICT[utility])
    for lin_thresh in LINEAR_THRESHOLDS
    for utility in UTILITIES
    for sampling in SAMPLINGS
    for cum_env in CUM_ENVS
    if not (lin_thresh and (not "threshold" in utility))
    if not (sampling and ("target" in utility))
]

MODELS = ["64_64_model", "128_128_64_model", "256_128_128_64_64_model", "512_256_256_128_128_64_model"]

ENV_DICT = dict([("{}{}{}{}env".format("cum_" * cum_env, utility + "_", (sampling + "_") * bool(sampling),
                                       "linear_" * lin_thresh),
                  "{}{}{}{}".format("Cumulative Rewards " * cum_env, SAMPLINGS_DICT[sampling],
                                    "Linear to " * lin_thresh, UTILITIES_DICT[utility]))
                 for lin_thresh in LINEAR_THRESHOLDS
                 for utility in UTILITIES for sampling in SAMPLINGS
                 for cum_env in CUM_ENVS
                 if not (lin_thresh and (not "threshold" in utility))
                 if not (sampling and ("target" in utility))])

FIXED_ENV_UTILITIES = {
    "dual_threshold": [
        DualThresholdUtility(dual_thresholds_and_coefficients=np.array(
            [[31, 31, dual_thresh0, dual_thresh1, dual_thresh2, dual_thresh3], [-1, -5, r0, r1, r2, r3]],
            dtype=np.float32)) for dual_thresh0 in range(3) for dual_thresh1 in range(3)
        for dual_thresh2 in range(3) for dual_thresh3 in range(3) for r0 in np.arange(-20, 21, step=6)
        for r1 in np.arange(-20, 21, step=6) for r2 in np.arange(-20, 21, step=6)
        for r3 in np.arange(-20, 21, step=6)
        if (r0 > 0) or (r1 > 0) or (r2 > 0) or (r3 > 0)
    ],
    "linear_dual_threshold": [
        DualThresholdUtility(dual_thresholds_and_coefficients=np.array(
            [[31, 31, 31, 31, 31, 31], [-1, -5, r0, r1, r2, r3]], dtype=np.float32))
        for r0 in np.arange(-20, 21, step=2) for r1 in np.arange(-20, 21, step=2)
        for r2 in np.arange(-20, 21, step=2)
        for r3 in np.arange(-20, 21, step=2)
        if (r0 > 0) or (r1 > 0) or (r2 > 0) or (r3 > 0)
    ],
    "linear": [
        LinearUtility(weights=np.array([-1, -5, r0, r1, r2, r3], dtype=np.float32))
        for r0 in np.arange(-20, 21, step=2) for r1 in np.arange(-20, 21, step=2)
        for r2 in np.arange(-20, 21, step=2)
        for r3 in np.arange(-20, 21, step=2)
        if (r0 > 0) or (r1 > 0) or (r2 > 0) or (r3 > 0)
    ],
    "threshold": [
        ThresholdUtility(thresholds_and_coefficients=np.array(
            [[0, 0, thresh0, thresh1, thresh2, thresh3], [-1, -5, r0, r1, r2, r3]], dtype=np.float32))
        for thresh0 in range(3) for thresh1 in range(3) for thresh2 in range(3) for thresh3 in range(3)
        for r0 in np.arange(-20, 21, step=6) for r1 in np.arange(-20, 21, step=6)
        for r2 in np.arange(-20, 21, step=6)
        for r3 in np.arange(-20, 21, step=6)
        if (r0 > 0) or (r1 > 0) or (r2 > 0) or (r3 > 0)
    ],
    "linear_threshold": [
        ThresholdUtility(thresholds_and_coefficients=np.array([[0, 0, 0, 0, 0, 0], [-1, -5, r0, r1, r2, r3]],
                                                              dtype=np.float32))
        for r0 in np.arange(-20, 21, step=2) for r1 in np.arange(-20, 21, step=2)
        for r2 in np.arange(-20, 21, step=2)
        for r3 in np.arange(-20, 21, step=2)
        if (r0 > 0) or (r1 > 0) or (r2 > 0) or (r3 > 0)
    ],
    "target": [
        TargetUtility(target=np.array([31, 31, target0, target1, target2, target3], dtype=np.float32))
        for target0 in range(4) for target1 in range(4) for target2 in range(3)
        for target3 in range(4)
        if (target0 > 0) or (target1 > 0) or (target2 > 0) or (target3 > 0)
    ]
}


def load_results(path: Path) -> Dict[str, np.ndarray]:
    keys = set([file.name for file in path.iterdir() if (file.is_dir() and (len(list(file.iterdir())) == 6))])

    results = dict()
    for key in keys:
        key_split = key.split("-")[::-1]
        key_split[0] = ENV_DICT[key_split[0]]
        new_key = "-".join(key_split)
        key_folder = path.joinpath(key)
        if len(list(key_folder.iterdir())) != 6:
            raise RuntimeError(f"Incomplete results for this experiment: {key_folder}")
        results[new_key] = [np.load(file, allow_pickle=True) for file in key_folder.iterdir()]
        sample_sizes = np.array([arr.shape[0] for arr in results[new_key]])
        if np.any(sample_sizes != sample_sizes[0]):
            raise RuntimeError(f"Non-matching results for this experiment: {key_folder}")

    return results


def convert_to_latex(df: pd.DataFrame) -> str:
    latex_output = ""
    for row in df.iterrows():
        latex_output += row[0].replace("Cumulative Rewards", "Cum. Rew.") + " &"
        for cell in row[1].values:
            mean, std = cell.split(" ")
            latex_output += " " + mean + " "
            std = std.replace("+-", "$\pm$")
            latex_output += std
            latex_output += "&"
        latex_output = latex_output[:-1]
        latex_output += "\\\\"

    return latex_output


def load_reward_vector_results(path: Path) -> Dict[str, np.ndarray]:
    keys = set([file.name for file in path.iterdir() if (file.is_dir() and (len(list(file.iterdir())) == 6))])

    results = dict()
    for key in keys:
        key_split = key.split("-")[::-1]
        key_split[0] = ENV_DICT[key_split[0]]
        new_key = "-".join(key_split)
        key_folder = path.joinpath(key)
        results[new_key] = [
            np.array([cum_rew
                      for cum_rew in np.load(file, allow_pickle=True)[:, 1]], dtype=np.float32)
            for file in key_folder.iterdir()
        ]

    return results


def sorting(key: str) -> int:
    env_sorting = dict([(env, i) for i, env in enumerate(ENVS)])
    model_sorting = dict([(model, i) for i, model in enumerate(MODELS)])

    env, model = key.split("-")

    return env_sorting[env] * (max(model_sorting.values()) + 1) + model_sorting[model]


def match_utility_to_fixed_env(env: str) -> List[UtilityFunction]:
    if env in LINEAR_TO_SETTINGS.intersection(LINEAR_DUAL_THRESHOLD_SETTINGS.union(DUAL_THRESHOLD_SETTINGS)):
        return FIXED_ENV_UTILITIES["linear_dual_threshold"]
    elif env in LINEAR_DUAL_THRESHOLD_SETTINGS.union(DUAL_THRESHOLD_SETTINGS):
        return FIXED_ENV_UTILITIES["dual_threshold"]
    elif env in LINEAR_TO_SETTINGS.intersection(LINEAR_THRESHOLD_SETTINGS.union(THRESHOLD_SETTINGS)):
        return FIXED_ENV_UTILITIES["linear_threshold"]
    elif env in LINEAR_THRESHOLD_SETTINGS.union(THRESHOLD_SETTINGS):
        return FIXED_ENV_UTILITIES["threshold"]
    elif env in LINEAR_SETTINGS:
        return FIXED_ENV_UTILITIES["linear"]
    elif env in TARGET_SETTINGS:
        return FIXED_ENV_UTILITIES["target"]


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


def compute_non_dominated(unique: np.ndarray) -> np.ndarray:
    """Given an array of unique reward vectors, determines which ones are non-dominated.

    Args:
        unique (np.ndarray): numpy array of unique reward vectors

    Returns:
        np.ndarray: boolean array where each entry specifies whether the corresponding reward vector is
                    non-dominated within the set or not.
    """
    m = unique.shape[0]
    domination_matrix = pdist(unique, domination_metric)
    is_non_dominated = np.ones(shape=(m,), dtype=bool)
    for i in range(m - 1):
        if is_non_dominated[i]:
            start_range = m * i + i + 1 - ((i + 2) * (i + 1)) // 2
            end_range = m * i + m - 1 - ((i + 2) * (i + 1)) // 2
            i_range = domination_matrix[start_range:end_range + 1]
            is_non_dominated[i] = np.all(i_range != -1)
            is_non_dominated[i + 1 + np.argwhere(i_range == 1)] = 0

    return is_non_dominated


def check_correct_non_dominated(env: str, unique:np.ndarray, inverse: np.ndarray, is_non_dominated: np.ndarray) -> np.ndarray:
    non_dom_indices = np.argwhere(is_non_dominated)
    non_dom_vectors = unique[is_non_dominated]
    indices_to_test = np.isin(inverse, non_dom_indices)
    
    utility_functions = match_utility_to_fixed_env(env=env)
    assert inverse.shape[0] == len(utility_functions), (inverse.shape[0], len(utility_functions))
    utility_functions_to_test = [f for i, f in enumerate(utility_functions) if indices_to_test[i]]
    original_inputs = unique[inverse[indices_to_test]][:, np.newaxis]
    competitors = np.repeat(non_dom_vectors[np.newaxis], np.sum(indices_to_test, dtype=int), axis=0)
    test_inputs = np.concatenate((original_inputs, competitors), axis=1)
    assert len(utility_functions_to_test) == test_inputs.shape[0]
    outputs = np.array([f(x) for f, x in zip(utility_functions_to_test, test_inputs)])
    assert outputs.shape == test_inputs.shape[:-1], (outputs.shape, test_inputs.shape[:-1], env)
    return np.mean(np.argmax(outputs, axis=1) == 0)


def wds_tables(results: Dict[str, List[np.ndarray]]) -> Tuple[pd.DataFrame]:
    keys = sorted(results.keys(), key=sorting)
    intra_wds = pd.DataFrame(columns=["Setting", "Model", "WD"])
    inter_wds = pd.DataFrame(columns=["Setting", "Models", "WD"])

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
        intra_wds = intra_wds.append({"Setting": env, "Model": model, "WD": val}, ignore_index=True)

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
                        "Setting": env,
                        "Models": "-".join([model_1, model_2]),
                        "WD": val
                    }, ignore_index=True)

    intra_wds['Setting'] = pd.Categorical(intra_wds["Setting"], ENVS)
    intra_wds['Model'] = pd.Categorical(intra_wds["Model"], MODELS)
    inter_wds['Setting'] = pd.Categorical(inter_wds["Setting"], ENVS)

    return intra_wds, inter_wds


def utilities_table(results: Dict[str, List[np.ndarray]]) -> pd.DataFrame:
    keys = sorted(results.keys(), key=sorting)
    df = pd.DataFrame(columns=["Setting", "Model", "Utility"])

    for key in keys:
        env, model = key.split("-")
        res = [np.mean(result) for result in results[key]]
        mean_util = np.round(np.mean(res), 2)
        std_err = np.round(np.std(res) / np.sqrt(len(res)), 2)
        val = f"{mean_util} (+-{std_err})"
        df = df.append({"Setting": env, "Model": model, "Utility": val}, ignore_index=True)

    df['Setting'] = pd.Categorical(df["Setting"], ENVS)
    df['Model'] = pd.Categorical(df["Model"], MODELS)

    return df


def uniques_non_dom_table(results: Dict[str, List[np.ndarray]],
                          correct_non_dom_flag: bool = True) -> pd.DataFrame:
    keys = sorted(results.keys(), key=sorting)
    df = pd.DataFrame(
        columns=["Setting", "Model", "Uniques", "Non-Dominated", "Non-Dominated %"] + ["Correct Non-Dominated"]*correct_non_dom_flag)

    for key in tqdm(keys):
        env, model = key.split("-")

        uniques_inverse_counts = [np.unique(result, axis=0, return_inverse=True, return_counts=True)
                                  for result in results[key]]

        non_dominated = []
        non_dominated_perc = []
        correct_non_dom = []
        for unique, inverse, counts in uniques_inverse_counts:
            is_non_dominated = compute_non_dominated(unique=unique * [-1, -1, 1, 1, 1, 1])
            if correct_non_dom_flag:
                correct_non_dom.append(check_correct_non_dominated(env=env, unique=unique, inverse=inverse,
                                                                   is_non_dominated=is_non_dominated))
            non_dominated_perc.append(np.sum(counts[is_non_dominated]) / np.sum(counts))
            non_dominated.append(np.sum(is_non_dominated))

        uniques_shapes = [unique.shape[0] for unique, inverse, counts in uniques_inverse_counts]
        
        mean_val_uniques = np.round(np.mean(uniques_shapes), 1)
        std_err_uniques = np.round(np.std(uniques_shapes) / np.sqrt(len(uniques_shapes)), 2)
        
        mean_val_non_dom = np.round(np.mean(non_dominated), 1)
        std_err_non_dom = np.round(np.std(non_dominated) / np.sqrt(len(non_dominated)), 2)
        
        mean_val_non_dom_perc = np.round(np.mean(non_dominated_perc), 2)
        std_err_non_dom_perc = np.round(np.std(non_dominated_perc) / np.sqrt(len(non_dominated_perc)), 2)
        
        if correct_non_dom_flag:
            mean_val_correct_non_dom = np.round(np.mean(correct_non_dom), 2)
            std_err_correct_non_dom = np.round(np.std(correct_non_dom) / np.sqrt(len(correct_non_dom)), 2)

        row = {
            "Setting": env,
            "Model": model,
            "Uniques": f"{mean_val_uniques} (+-{std_err_uniques})",
            "Non-Dominated": f"{mean_val_non_dom} (+-{std_err_non_dom})",
            "Non-Dominated %": f"{mean_val_non_dom_perc} (+-{std_err_non_dom_perc})"
            }
        if correct_non_dom_flag:
            row["Correct Non-Dominated"] = f"{mean_val_correct_non_dom} (+-{std_err_correct_non_dom})"
        df = df.append(row, ignore_index=True)

    df['Setting'] = pd.Categorical(df["Setting"], ENVS)
    df['Model'] = pd.Categorical(df["Model"], MODELS)

    return df


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['Setting'] = pd.Categorical(df["Setting"], ENVS)
    df['Model'] = pd.Categorical(df["Model"], MODELS)
    return df


def plot_training_history(path: Path, model: str, env: str) -> None:
    exp_dirs = [
        "-".join((model, env, "replication" + train_id)) for train_id in ["", "-1", "-2", "-3", "-4", "-5"]
    ]
    plt.figure(figsize=(25, 10))
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        image_path = path.joinpath(exp_dirs[i], "plots", "reward_plot.png")
        img = mpimg.imread(image_path)
        plt.imshow(img)
        plt.axis('off')
    plt.show()


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


# Some predefined selections of setting to be used in the jupyter notebooks
ALL_SETTINGS = set(ENVS)

CUMULATIVE_SETTINGS = set([env for env in ENVS if "Cumulative Rewards" in env])

DUAL_THRESHOLD_SETTINGS = set([
    "{}{}{}Dual Threshold".format("Cumulative Rewards " * cum_env, SAMPLINGS_DICT[sampling],
                                  "Linear to " * lin_thresh)
    for lin_thresh in LINEAR_THRESHOLDS
    for sampling in SAMPLINGS
    for cum_env in CUM_ENVS
])

LINEAR_DUAL_THRESHOLD_SETTINGS = set([
    "{}{}{}Linear Dual Threshold".format("Cumulative Rewards " * cum_env, SAMPLINGS_DICT[sampling],
                                         "Linear to " * lin_thresh)
    for lin_thresh in LINEAR_THRESHOLDS
    for sampling in SAMPLINGS
    for cum_env in CUM_ENVS
])

LINEAR_SETTINGS = set([
    "{}{}Linear".format("Cumulative Rewards " * cum_env, SAMPLINGS_DICT[sampling])
    for sampling in SAMPLINGS
    for cum_env in CUM_ENVS
])

THRESHOLD_SETTINGS = set([
    "{}{}{}Threshold".format("Cumulative Rewards " * cum_env, SAMPLINGS_DICT[sampling],
                             "Linear to " * lin_thresh)
    for lin_thresh in LINEAR_THRESHOLDS
    for sampling in SAMPLINGS
    for cum_env in CUM_ENVS
])

LINEAR_THRESHOLD_SETTINGS = set([
    "{}{}{}Linear Threshold".format("Cumulative Rewards " * cum_env, SAMPLINGS_DICT[sampling],
                                    "Linear to " * lin_thresh)
    for lin_thresh in LINEAR_THRESHOLDS
    for sampling in SAMPLINGS
    for cum_env in CUM_ENVS
])

TARGET_SETTINGS = {'Cumulative Rewards Target', 'Target'}

LINEAR_TO_SETTINGS = set([
    "{}{}Linear to {}".format("Cumulative Rewards " * cum_env, SAMPLINGS_DICT[sampling],
                              UTILITIES_DICT[utility])
    for utility in ["threshold", "linear_threshold", "dual_threshold", "linear_dual_threshold"]
    for sampling in SAMPLINGS
    for cum_env in CUM_ENVS
])

STANDARD_SAMPLING_SETTINGS = set([
    "{}{}{}".format("Cumulative Rewards " * cum_env, "Linear to " * lin_thresh, UTILITIES_DICT[utility])
    for lin_thresh in LINEAR_THRESHOLDS
    for utility in UTILITIES
    for cum_env in CUM_ENVS
    if not (lin_thresh and (not "threshold" in utility))
])

DENSE_SAMPLING_SETTINGS = set([
    "{}Dense {}{}".format("Cumulative Rewards " * cum_env, "Linear to " * lin_thresh, UTILITIES_DICT[utility])
    for lin_thresh in LINEAR_THRESHOLDS
    for utility in ["linear", "threshold", "linear_threshold", "dual_threshold", "linear_dual_threshold"]
    for cum_env in CUM_ENVS
    if not (lin_thresh and (not "threshold" in utility))
])

CONTINUOUS_SETTINGS = set([
    "{}Continuous {}{}".format("Cumulative Rewards " * cum_env, "Linear to " * lin_thresh,
                               UTILITIES_DICT[utility])
    for lin_thresh in LINEAR_THRESHOLDS
    for utility in ["linear", "threshold", "linear_threshold", "dual_threshold", "linear_dual_threshold"]
    for cum_env in CUM_ENVS
    if not (lin_thresh and (not "threshold" in utility))
])


if __name__ == "__main__":
    results_path = Path("C:/Users/maler/Federico/Università/Master/Tesi/evaluation_results/")
    fixed_env_results_path = results_path.joinpath("fixed_env_results/")
    fixed_env_results = load_results(fixed_env_results_path)
    fixed_env_unique_non_dom_table = uniques_non_dom_table(fixed_env_results)
