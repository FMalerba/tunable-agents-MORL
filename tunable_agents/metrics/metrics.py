from absl import logging
import gin
import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.metrics import tf_metric
from tf_agents.metrics.tf_metrics import TFDeque
from tf_agents.utils import common
from tf_agents.utils import nest_utils


@gin.configurable()
class ActionDistribution(tf_metric.TFStepMetric):
    
    def __init__(self,
                 name='ActionDistribution',
                 prefix='Metrics',
                 dtype=tf.float32,
                 batch_size=1,
                 buffer_size=10):
        super().__init__(name, prefix=prefix)
        self._buffer = TFDeque(buffer_size, dtype)
        self._dtype = dtype
        self._return_accumulator = common.create_variable(
            initial_value=0, dtype=dtype, shape=(batch_size,), name='Accumulator')
        
    @common.function(autograph=True)
    def call(self, trajectory):
        # Each non-boundary trajectory (first, mid or last) represents a step.
        non_boundary_indices = tf.squeeze(
            tf.where(tf.logical_not(trajectory.is_boundary())), axis=-1)
        self._length_accumulator.scatter_add(
            tf.IndexedSlices(
                tf.ones_like(
                    non_boundary_indices, dtype=self._length_accumulator.dtype),
                non_boundary_indices))

        # Add lengths to buffer when we hit end of episode
        last_indices = tf.squeeze(tf.where(trajectory.is_last()), axis=-1)
        for indx in last_indices:
            self._buffer.add(self._length_accumulator[indx])

        # Clear length accumulator at the end of episodes.
        self._length_accumulator.scatter_update(
            tf.IndexedSlices(
                tf.zeros_like(last_indices, dtype=self._dtype), last_indices))

        return trajectory

    def result(self):
        return self._buffer.mean()

    @common.function
    def reset(self):
        self._buffer.clear()
        self._length_accumulator.assign(tf.zeros_like(self._length_accumulator))



