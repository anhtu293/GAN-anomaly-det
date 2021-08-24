import numpy as np
import tensorflow as tf


def get_bbox(obj):
    points = np.asarray(obj['points']['exterior'])
    x_axis = points[:,0]
    y_axis = points[:,1]
    return [min(x_axis), min(y_axis), max(x_axis), max(y_axis)]


class Config:
    def __init__(self, dictionary):
        for key in dictionary:
            setattr(self, key, dictionary[key])

    def get(self, key, default):
        return getattr(self, key, default)


class TransformerScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super(TransformerScheduler, self).__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
