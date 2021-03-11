from absl import app
from absl import flags
from absl import logging

import numpy as np
import os
from tqdm import tqdm

import gin

from tunable_agents import utility, agent
from tf_agents.environments import py_environment


@gin.configurable
def train_eval(training_id: str,model_id: str,env_id: str):
    pass


def eval_agent(env: py_environment.PyEnvironment, tf_agent: agent.DQNAgent,
               n_episodes: int, reward_vector: bool=False) -> np.ndarray:
    
    results = []
    for _ in tqdm(range(n_episodes)):
        ts = env.reset()
        observations = ts.observation
        
        episode_reward = 0
        done = False
        while not done:
            action = tf_agent.greedy_policy(observations)
            ts = env.step(action)
            observations, reward, done = ts.observation, ts.reward, ts.is_last()
            
            episode_reward += reward
        
        assert episode_reward == env._prev_step_utility
        if reward_vector:
            results.append([observations['utility_representation'], env._cumulative_rewards])
        else:
            results.append(episode_reward)
    
    if reward_vector:
        results = np.array(results, dtype='object')
    else:
        results = np.array(results)
    
    return results


def main(_):
    logging.set_verbosity(logging.INFO)
    utility.load_gin_configs(FLAGS.gin_files, FLAGS.gin_bindings)
    
    
    experiment_dir = FLAGS.experiment_dir
    model_dir = os.path.join(experiment_dir, 'model')
    model_path = os.path.join(model_dir, 'dqn_model.h5')

    
    env = utility.create_environment()
    tf_agent = agent.DQNAgent(epsilon=0,
                              obs_spec=env.observation_spec())
    
    tf_agent.load_model(model_path)
    
    results = eval_agent(env, tf_agent, FLAGS.n_episodes, FLAGS.reward_vector)
    
    results_dir = os.path.join(FLAGS.results_dir, "reward_vector") if FLAGS.reward_vector else FLAGS.results_dir
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    results_path = results_dir + experiment_dir.split("/")[-1] + ".npy"
    if os.path.exists(results_path):
        results = np.append(np.load(results_path), results, axis=0)
    np.save(results_path, results)


if __name__ == '__main__':
    flags.DEFINE_string('experiment_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                        'Directory where the agent is saved.')
    flags.DEFINE_string('results_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                        'Root directory for writing the results')
    flags.DEFINE_multi_string('gin_files', [], 'List of paths to gin configuration files (e.g.'
                            '"configs/hanabi_rainbow.gin").')
    flags.DEFINE_multi_string(
        'gin_bindings', [], 'Gin bindings to override the values set in the config files '
        '(e.g. "train_eval.num_iterations=100").')
    flags.DEFINE_integer('n_episodes', 10_000, "Number of episodes to evaluate the agent on")
    flags.DEFINE_bool('reward_vector', False, "Whether to compute results for the reward vector"
                    " instead of the utility scalarized result.")
    FLAGS = flags.FLAGS
    flags.mark_flag_as_required('experiment_dir')
    
    #gpus = tf.config.experimental.list_physical_devices('GPU')
    #tf.config.experimental.set_memory_growth(gpus[0], True)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    app.run(main)
