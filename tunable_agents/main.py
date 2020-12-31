import os
import time
from typing import Callable

from absl import app
from absl import flags
from absl import logging

import gin
from tunable_agents import utility
import tensorflow as tf
from tf_agents.agents import tf_agent
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import tf_py_environment
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.utils import common

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_bool('XLA_flag', True, "Whether to use XLA or not. Can't be True when running on Windows")
flags.DEFINE_multi_string('gin_files', [], 'List of paths to gin configuration files (e.g.'
                          '"configs/hanabi_rainbow.gin").')
flags.DEFINE_multi_string(
    'gin_bindings', [], 'Gin bindings to override the values set in the config files '
    '(e.g. "train_eval.num_iterations=100").')

FLAGS = flags.FLAGS


@gin.configurable(allowlist=['collect_episodes'])
def initial_collection(agent: tf_agent.TFAgent, env: tf_py_environment.TFPyEnvironment, rb_observer: list,
                       collect_episodes: int) -> None:
    """
    Samples transitions with the given agent and environment without training
    on the experience collected. This is done to partially fill-up the Replay Buffer.
    """
    if not collect_episodes:
        return
    collect_policy = agent.collect_policy
    print('Sampling {} episodes with no training'.format(collect_episodes))
    start_time = time.time()
    dynamic_episode_driver.DynamicEpisodeDriver(env,
                                                collect_policy,
                                                observers=rb_observer,
                                                num_episodes=collect_episodes).run()
    print('Finished sampling with the Driver, it took {} seconds for {} episodes\n'.format(
        time.time() - start_time, collect_episodes))


@gin.configurable(allowlist=[])
def initial_training(agent_train_function: Callable,
                     replay_buffer: tf_uniform_replay_buffer.TFUniformReplayBuffer, num_steps: int,
                     train_steps: int, batch_size: int) -> None:
    if not train_steps:
        return
    # Dataset generates trajectories with shape [Bx2x...]
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3, sample_batch_size=batch_size,
        num_steps=num_steps).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    print('Starting partial training of Agent from Replay Buffer' '\nCounting Steps:')
    # Commenting out losses (and all their relevant code) to try and see if they are responsible for an observed memory leak.
    # No feedback available yet on whether this is the case or not
    # losses = tf.TensorArray(tf.float32, size=steps)
    c = 0
    start_time = time.time()
    for data in dataset:
        if c % (train_steps / 10) == 0 and c != 0:
            #tf.summary.scalar("loss_agent", tf.math.reduce_mean(losses.stack()), step=train_step)
            print("{}% completed with {} steps done".format(int(c / train_steps * 100), c))
        if c == train_steps:
            break
        experience, data_info = data
        """
        FIXME tensorflow documentation at https://www.tensorflow.org/tensorboard/migrate states that
            default_writers do not cross the tf.function boundary and should instead be called as default
            inside the tf.function. For now our code works because on the first run of the training function
            the code is run in non-graph mode and thus "sees" the writer (and can then use it even in subsequent
            graph-mode executions). This will stop working either if we start to export the compiled functions
            so that we don't have the first "pythonic" run of them or if for some reason we change the file_writer
            during execution. Should the summary writer be passed to the agent training function so that it can be set 
            as default from inside the boundary of tf.function? How does tf-agent solve this issue?
        """

        # losses = losses.write(c, agent_train_function(experience=experience).loss)
        loss_agent_info = agent_train_function(experience=experience)
        c += 1

    # losses = losses.stack()
    print("Ended epoch training for agent, it took {}".format(time.time() - start_time))


@gin.configurable
def train_eval(
        root_dir: str,
        experiment_name: str,
        env_id: str,  # Used to identify a unique environment for RB reuse
        num_iterations: int,
        # Param for collect
        collect_episodes_per_epoch: int,
        # Number of steps for training update
        num_steps: int,
        # Params for train
        train_steps_per_epoch: int,
        batch_size: int,
        # Params for eval
        eval_interval: int,
        num_eval_episodes: int,
        # Param for checkpoints
        checkpoint_interval: int,
        shared_RB: bool = False,
        XLA_flag: bool = True  # When running on Windows machine, should be set to False
):
    """A simple train and eval for DQN."""
    root_dir = os.path.expanduser(root_dir)
    experiment_dir = os.path.join(root_dir, experiment_name)
    train_dir = os.path.join(experiment_dir, 'train')
    eval_dir = os.path.join(experiment_dir, 'eval')

    # The summary writers are set with a 2h flush interval. They are actually meant to never
    # flush by themselves, but only when flush is called on checkpointing. This is to synch
    # tensorboard summaries with the checkpoints so as to not write summaries twice in case of
    # restart of the code. Careful not to checkpoint too seldomly to avoid abusing memory.
    train_summary_writer = tf.summary.create_file_writer(train_dir, flush_millis=3600 * 2 * 1000)
    train_summary_writer.set_as_default()
    eval_summary_writer = tf.summary.create_file_writer(eval_dir, flush_millis=3600 * 2 * 1000)

    tf.profiler.experimental.server.start(6008)
    """
    TODO use ParallelPyEnvironment to run envs in parallel and see how much we can speed up.
        See: https://www.youtube.com/watch?v=U7g7-Jzj9qo&list=TLPQMDkwNDIwMjB-xXfzXt3B5Q&index=2 
        at minute 26:50
        Note: it is more than likely that batching the environment might require also passing a 
        different batch_size parameter to the metrics and the replay buffer. Also note that the
        replay buffer actually stores batch_size*max_length frames, so for example right now to
        have a RB with 50k capacity you would have batch_size=1, max_length=50k. This is probaably
        done for parallelization and memory access issues, where one wants to be sure that the
        parallel runs don't access the same memory slots of the RB... As such if you want to run
        envs in parallel and keep RB capacity fixed you should divide the desired capacity by batch_size
        and use that as max_length parameter. Btw, a frame stored by the RB can be variable;
        if num_steps=2 (as right now) then a frame is  [time_step, action, next_time_step] 
        (where time_step has all info including last reward). If you increase num_steps then 
        it's obvious how a frame would change, and also how this affects the *actual* number of transitions
        that the RB is storing. Also note that if I ever actually manage to do the Prioritized RB,
        it won't support this batch parallelization. The issue lies with the SumTree object 
        (which I imported from the DeepMind framework) and the fact that it doesn't seem to me
        like this object could be parallelized (meaning that all memory access issues are solved) in any way...
    """
    # create the enviroment
    env = utility.create_environment()
    tf_env = tf_py_environment.TFPyEnvironment(env)
    eval_py_env = tf_py_environment.TFPyEnvironment(utility.create_environment())

    train_step = tf.Variable(0, trainable=False, name='global_step', dtype=tf.int64)

    epoch_counter = tf.Variable(0, trainable=False, name='Epoch', dtype=tf.int64)

    # Epsilon implementing decaying behaviour for the two agents
    decaying_epsilon = utility.decaying_epsilon(step=epoch_counter)
    """
    TODO Performance Improvement: "When training on GPUs, make use of the TensorCore. GPU kernels use
        the TensorCore when the precision is fp16 and input/output dimensions are divisible by 8 or 16 (for int8)"
        (from https://www.tensorflow.org/guide/profiler#improve_device_performance). Maybe consider decreasing
        precision to fp16 and possibly compensating with increased model complexity to not lose performance?
        I mean if this allows us to use TensorCore then maybe it is worthwhile (computationally) to increase 
        model size and lower precision. Need to test what the impact on agent performance is.
        See https://www.tensorflow.org/guide/keras/mixed_precision for more info
    """
    # create an agent and a network
    tf_agent = utility.create_agent(
        environment=tf_env,
        # num_steps parameter must differ by 1 between agent and replay_buffer.as_dataset() call
        n_step_update=num_steps - 1,
        decaying_epsilon=decaying_epsilon,
        train_step_counter=train_step)

    # replay buffer
    replay_buffer = utility.create_replay_buffer(data_spec=tf_agent.collect_data_spec,
                                                 batch_size=tf_env.batch_size)

    # metrics
    train_metrics = [
        tf_metrics.NumberOfEpisodes(),
        tf_metrics.EnvironmentSteps(),
        tf_metrics.AverageReturnMetric(buffer_size=collect_episodes_per_epoch),
        tf_metrics.AverageEpisodeLengthMetric(buffer_size=collect_episodes_per_epoch),
    ]

    # replay buffer update for the driver
    replay_observer = [replay_buffer.add_batch]

    eval_metrics = [
        tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
        tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes),
    ]

    # I create the Driver only once instead of recreating it at the start of every epoch
    # This is not a problem only because the agent's collect_policy has the agent's network
    # and its tensors will update together with the agent's on agent.train() calls.
    # If the agent is has a policy that doesn't depend only on Tensors, one must recreate a new
    # policy after every epoch and create a new Driver to feed this to.
    collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(tf_env,
                                                                 tf_agent.collect_policy,
                                                                 observers=replay_observer + train_metrics,
                                                                 num_episodes=collect_episodes_per_epoch)

    # checkpointer:
    train_checkpointer = common.Checkpointer(ckpt_dir=os.path.join(experiment_dir, 'checkpoints', 'train'),
                                             agent=tf_agent,
                                             train_step=train_step,
                                             epoch_counter=epoch_counter,
                                             metrics=metric_utils.MetricsGroup(
                                                 train_metrics, 'train_metrics'))

    policy_checkpointer = common.Checkpointer(ckpt_dir=os.path.join(experiment_dir, 'checkpoints', 'policy'),
                                              policy=tf_agent.policy)

    if shared_RB:
        rb_checkpoint_dir = os.path.join(experiment_dir, 'checkpoints', 'rb')
    else:
        rb_checkpoint_dir = os.path.join(root_dir, 'replay_buffers', env_id)
    rb_checkpointer = common.Checkpointer(ckpt_dir=rb_checkpoint_dir,
                                          max_to_keep=1,
                                          replay_buffer=replay_buffer)
    """
    FIXME Tensorflow documentation of tf.function (https://www.tensorflow.org/api_docs/python/tf/function)
        states that autograph parameter should be set to True for Data-dependent control flow. What does this
        mean? Is our training function not Data-dependent? Currently common.function (which is a wrapper on the 
        tf.function wrapper) passes autograph=False by default.
    TODO Maybe pass experimental_compile=True to common.function? Maybe it's not needed because 
        later in the code we use tf.config.optimizer.set_jit(True) which enables XLA in general?
        Who knows, test and look at performance I would say. Another thing to notice is that
        experimental_compile=True would have the added bonus of telling us if indeed it manages
        to compile or not since "The experimental_compile API has must-compile semantics: either 
        the entire function is compiled with XLA, or an errors.InvalidArgumentError exception is thrown."
        See: https://www.tensorflow.org/xla#explicit_compilation_with_tffunction
    TODO common.function passes the parameter experimental_relax_shapes=True by default. Maybe 
        consider instead passing it as False for efficiency... This is most likely linked to the
        input_signature TODO that follows
    TODO (low priority) add an input_signature parameter so that tf.function knows what to expect
        and won't adapt to the input if something strange happens (which it really shouldn't happen)
    """
    # Compiled version of training functions (much faster)
    agent_train_function = common.function(tf_agent.train)

    # Care that JIT only improves performance if stuff doesn't change shapes often
    tf.config.optimizer.set_jit(XLA_flag)

    initial_collection(tf_agent, tf_env, replay_observer)
    
    initial_training(agent_train_function=agent_train_function, replay_buffer=replay_buffer)

    for _ in range(num_iterations):
        print('EPOCH {}'.format(epoch_counter.numpy()))
        tf.summary.scalar(name='Epsilon', data=decaying_epsilon(), step=epoch_counter)
        # episode driver
        print('\nStarting to run the Driver')
        start_time = time.time()
        collect_driver.run()
        print('Finished running the Driver, it took {} seconds for {} episodes\n'.format(
            time.time() - start_time, collect_episodes_per_epoch))
        """
        TODO Performance Optimization: Try out different batch sizes (TF usually recommends higher batch size) 
            and see how this influences performance, keeping track of possible differences in RAM/VRAM requirements.
            To do this properly the variable train_steps_per_epoch should be changed appropriately (e.g. double
            the batch size --> half the train_steps), but it would be nice to also check that this behaves
            as expected and doesn't impact per-epoch-learning. Per-epoch-learning is an abstract metric I just
            invented that would tell you how much better a model got after an epoch... Essentially one should
            check that the agent manages to reach the same level of performance (measured perhaps in 
            average_return_per_episode == number of fireworks placed) at the same epoch (more or less) even if
            you do this thing of doubling batch_size and halving train_steps_per_epoch.
        """
        # Dataset generates trajectories with shape [Bx2x...]
        dataset = replay_buffer.as_dataset(
            num_parallel_calls=3, sample_batch_size=batch_size,
            num_steps=num_steps).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        print('Starting partial training of Agent from Replay Buffer' '\nCounting Steps:')
        # Commenting out losses_1/_2 (and all their relevant code) to try and see if they are responsible for an observed memory leak.
        # No feedback available yet on whether this is the case or not
        # losses = tf.TensorArray(tf.float32, size=train_steps_per_epoch)
        c = 0
        start_time = time.time()
        for data in dataset:
            if c % (train_steps_per_epoch / 10) == 0 and c != 0:
                #tf.summary.scalar("loss_agent", tf.math.reduce_mean(losses.stack()), step=train_step)
                print("{}% completed with {} steps done".format(int(c / train_steps_per_epoch * 100), c))
            if c == train_steps_per_epoch:
                break
            experience, data_info = data
            """
            FIXME tensorflow documentation at https://www.tensorflow.org/tensorboard/migrate states that
                default_writers do not cross the tf.function boundary and should instead be called as
                default inside the tf.function. For now our code works because on the first run of the
                training function the code is run in non-graph mode and thus "sees" the writer (and can
                then use it even in subsequent graph-mode executions). This will stop working either if
                we start to export the compiled functions so that we don't have the first "pythonic" run
                of them or if for some reason we change the file_writer during execution. Should the 
                summary writer be passed to the agent training function so that it can be set as default
                from inside the boundary of tf.function? How does tf-agent solve this issue?
            """

            # losses = losses.write(c, agent_train_function(experience=experience).loss)
            loss_agent_info = agent_train_function(experience=experience)
            c += 1

        # losses = losses.stack()
        print("Ended epoch training for agent, it took {}".format(time.time() - start_time))

        epoch_counter.assign_add(1)

        for train_metric in train_metrics:
            train_metric.tf_summaries(train_step=epoch_counter, step_metrics=train_metrics[:2])

        train_metrics[2].reset()
        train_metrics[3].reset()

        # Checkpointing and flushing summaries
        if epoch_counter.numpy() % checkpoint_interval == 0:
            train_summary_writer.flush()
            train_checkpointer.save(global_step=epoch_counter.numpy())
            policy_checkpointer.save(global_step=epoch_counter.numpy())
            rb_checkpointer.save(global_step=epoch_counter.numpy())

        # Evaluation Run
        if epoch_counter.numpy() % eval_interval == 0:
            eval_py_policy = tf_agent.policy
            metric_utils.eager_compute(eval_metrics,
                                       eval_py_env,
                                       eval_py_policy,
                                       num_episodes=num_eval_episodes,
                                       train_step=epoch_counter,
                                       summary_writer=eval_summary_writer,
                                       summary_prefix='Metrics')
            eval_summary_writer.flush()
            eval_metrics[0].reset()
            eval_metrics[1].reset()


def main(_):
    logging.set_verbosity(logging.INFO)
    utility.load_gin_configs(FLAGS.gin_files, FLAGS.gin_bindings)
    train_eval(root_dir=FLAGS.root_dir, XLA_flag=FLAGS.XLA_flag)


if __name__ == '__main__':
    flags.mark_flag_as_required('root_dir')
    app.run(main)
