import tensorflow as tf
import numpy as np


class Encoder():
    def __init__(self, input_shape, min_log_t=-7.5, max_log_t=-5.0, unconstrained_t=False):
        """
        Constructor
        :param min_log_t: minimum value for the log-scale parameter
        :param max_log_t: maximum value for the log-scale parameter
        :param unconstrained_t: boolean identifying wether scale parameter is restricted
        """
        self.input_shape = input_shape
        self.min_log_t = min_log_t
        self.max_log_t = max_log_t
        self.unconstrained_t = unconstrained_t

        # Network build
        self.inputs = None
        self.hidden = None
        self.outputs = None


    def _build_input(self):
        inputs = tf.keras.layers.Input(shape=self.input_shape, name='encoder_input')
        return inputs

    def _build_hidden(self, inputs):
        hidden = self.hidden_list[0](inputs)
        for layer in self.hidden_list[1:]:
            hidden = layer(hidden)
        return hidden

    def build_encoder_class_tensors(self, latent_dim, scale_dim):
        self.inputs = self._build_input()
        self.hidden_list = self._build_hidden_list()
        self.hidden = self._build_hidden(self.inputs)
        self.outputs = self._build_output(latent_dim, scale_dim, self.hidden)
        #   self.encoder = self._build_encoder()
        print("Encoder built")

    def _build_output(self, latent_dim, scale_dim, hidden):
        # Location parameter
        z_mean = tf.keras.layers.Dense(latent_dim, name='z_mean')(hidden)

        # Scale parameter
        if self.unconstrained_t:
            print("Unconstrained time")
            z_log_t = tf.keras.layers.Dense(scale_dim, name="z_log_var")(hidden)
        else:
            print("Log time between {} and {}".format(self.min_log_t, self.max_log_t))
            z_log_var_pre = tf.keras.layers.Dense(scale_dim, name="z_log_var", activation='tanh')(hidden)
            half_time_interval_length = (self.max_log_t - self.min_log_t) / 2
            time_interval_center = (self.max_log_t + self.min_log_t) / 2
            z_log_t = tf.keras.layers.Lambda(lambda x: tf.math.abs(half_time_interval_length) * x + time_interval_center,
                                          name="z_log_var_restricted")(z_log_var_pre)
        return [z_mean, z_log_t]


