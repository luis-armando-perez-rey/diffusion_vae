from modules.deltavae.encoder_decoder_architectures.decoder import decoder_parent
import keras.layers, keras.models
import numpy as np
import tensorflow as tf


class DecoderVGG(decoder_parent.Decoder):

    def __init__(self, input_shape, intermediate_shape, filter_list=[64, 64, 64],
                 kernel_size_list=[(3, 3), (3, 3), (3, 3)], pool_size_list=[(2, 2), (2, 2), (2, 2)],
                 dense_units_list=[64], decoder_output_activation=None):

        self.filter_list = filter_list
        self.kernel_size_list = kernel_size_list
        self.pool_size_list = pool_size_list
        self.dense_units_list = dense_units_list
        self.activation = decoder_output_activation
        self.intermediate_shape = intermediate_shape

        self.type = "VGG"

        super(DecoderVGG, self).__init__(input_shape)

        self.params_dict = {"type": self.type,
                            "input_shape": input_shape,
                            "intermediate_shape": intermediate_shape,
                            "filter_list": filter_list,
                            "kernel_size_list": kernel_size_list,
                            "pool_size_list": pool_size_list,
                            "dense_units_list": dense_units_list,
                            "decoder_output_activation": decoder_output_activation}
    def _build_hidden_list(self):
        hidden_list = []
        with tf.name_scope("DecoderVGGHidden") as scope:
            # Dense layers
            for num_units, units in reversed(list(enumerate(self.dense_units_list))):
                hidden_list.append(keras.layers.Dense(units, activation=None, name="h_dense_dec_" + str(num_units)))
                hidden_list.append(keras.layers.BatchNormalization())
                hidden_list.append(keras.layers.Activation('relu'))

            hidden_list.append(keras.layers.Dense(np.product(self.intermediate_shape), activation=None))
            hidden_list.append(keras.layers.Activation('relu'))
            hidden_list.append(keras.layers.BatchNormalization())
            hidden_list.append(keras.layers.Reshape(self.intermediate_shape))

            # Convolutional layers
            for num_kernel, kernel_size in reversed(list(enumerate(self.kernel_size_list[1:]))):
                if not (self.pool_size_list[num_kernel] is None):
                    hidden_list.append(keras.layers.UpSampling2D(size=self.pool_size_list[num_kernel]))
                hidden_list.append(keras.layers.Conv2D(filters=self.filter_list[num_kernel - 1], kernel_size=kernel_size,
                                                  strides=(1, 1), padding="same",
                                                  activation=None))
                hidden_list.append(keras.layers.BatchNormalization())
                hidden_list.append(keras.layers.Activation('relu'))

            # if not (self.pool_size_list[0] is None):
            # TODO: FIX THIS!!
            hidden_list.append(keras.layers.UpSampling2D(size=(2, 2)))

        return hidden_list




    def _build_output(self, hidden):
        x_recon = keras.layers.Conv2D(filters=self.input_shape[-1], kernel_size=self.kernel_size_list[0],
                                      strides=(1, 1), padding="same",
                                      activation=self.activation)(hidden)
        return x_recon

    def _build_decoder(self):
        decoder = keras.models.Model(self.inputs, self.outputs, name="DecoderVGGModel")
        return decoder


if __name__ == "__main__":
    latent_dim = 2
    input_shape = (64, 64, 3)
    filter_list = [64, 64, 64]
    kernel_size_list = [(3, 3), (3, 3), (3, 3)]
    pool_size_list = [(2, 2), (2, 2), (2, 2)]
    dense_units_list = [64]
    activation = None

    vertical = input_shape[0]
    horizontal = input_shape[1]
    for num_pool in range(len(pool_size_list)):
        if not (pool_size_list[num_pool] is None):
            vertical_pool = pool_size_list[num_pool][0]
            horizontal_pool = pool_size_list[num_pool][1]
            vertical = vertical // vertical_pool
            horizontal = horizontal // horizontal_pool
    final_filters = filter_list[-1]
    intermediate_shape = (vertical, horizontal, final_filters)

    decoder = DecoderVGG(input_shape, intermediate_shape)
    decoder.build_decoder_class_tensors(latent_dim)
    decoder.decoder.summary()
