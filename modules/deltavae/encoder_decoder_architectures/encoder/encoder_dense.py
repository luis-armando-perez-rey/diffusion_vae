from modules.deltavae.encoder_decoder_architectures.encoder import encoder_parent
import keras.layers
import tensorflow as tf


class EncoderDense(encoder_parent.Encoder):

    def __init__(self, input_shape,
                 dense_units_list=[64],
                 min_log_t=-7.5,
                 max_log_t=-5.0,
                 unconstrained_t=False,
                 batch_normalization=True):

        self.dense_units_list = dense_units_list

        self.type = "Dense"
        self.batch_normalization = batch_normalization
        super(EncoderDense, self).__init__(input_shape, min_log_t, max_log_t, unconstrained_t)

        self.params_dict = {"type": self.type,
                            "input_shape": self.input_shape,
                            "dense_units_list": self.dense_units_list,
                            "max_log_t": max_log_t,
                            "min_log_t": min_log_t,
                            "unconstrained_t": unconstrained_t,
                            "batch_normalization": batch_normalization
                            }
    def _build_hidden_list(self):
        with tf.name_scope("EncoderDenseHidden") as scope:
            hidden_list = []
            if len(self.input_shape) == 3:
                hidden_list.append(keras.layers.Flatten())
            # Loop through all the dense layers in the network
            for num_layer, num_neurons in enumerate(self.dense_units_list):
                hidden_list.append(keras.layers.Dense(num_neurons, activation=None))
                # Add batch normalization
                if self.batch_normalization:
                    hidden_list.append(keras.layers.BatchNormalization())
                hidden_list.append(keras.layers.Activation('relu'))
        return hidden_list

    # def _build_hidden(self):
    #     """
    #
    #     :return:
    #     """
    #     # Define the input image layer as the first layer in the encoder
    #     with tf.name_scope("EncoderDenseHidden") as scope:
    #         hidden = self.inputs
    #         if len(self.input_shape) == 3:
    #             hidden = keras.layers.Flatten()(hidden)
    #         # Loop through all the dense layers in the network
    #         for num_layer, num_neurons in enumerate(self.dense_units_list):
    #             hidden = keras.layers.Dense(num_neurons, activation=None)(hidden)
    #             # Add batch normalization
    #             if self.batch_normalization:
    #                 hidden = keras.layers.BatchNormalization()(hidden)
    #             hidden = keras.layers.Activation('relu')(hidden)
    #     return hidden


if __name__ == "__main__":
    input_shape = (64, 64, 3)
    filter_list = [64, 64, 64]
    kernel_size_list = [(3, 3), (3, 3), (3, 3)]
    pool_size_list = [None, (2, 2), None]
    dense_units_list = [64]
    min_log_t = -7.5
    max_log_t = -5.0
    unconstrained_t = False
    latent_dim = 2
    scale_dim = 1
    encoder_class = EncoderDense(input_shape, dense_units_list, min_log_t, max_log_t, unconstrained_t)
    encoder_class.build_encoder_class_tensors(latent_dim, scale_dim)
    encoder_class.encoder.summary()
    print("Intermediate shape", encoder_class.intermediate_shape)
