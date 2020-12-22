from gin import config
import tensorflow as tf

# config.external_configurable(tf.keras.layers.Dense, name='Dense', module='tf.keras.layers')
config.external_configurable(tf.keras.layers.Conv2D, name='Dense')
config.external_configurable(tf.keras.layers.Flatten, name='Flatten')
config.external_configurable(tf.keras.Sequential, name='Sequential')
config.external_configurable(tf.keras.layers.Dropout, name='Dropout')
