from modules.deltavae.encoder_decoder_architectures.decoder import decoder_parent
import tensorflow as tf
import numpy as np
import tensorflow as tf 

class DecoderDense(decoder_parent.Decoder):

    def __init__(self, input_shape,
                 dense_units_list=[64], batch_normalization = True, decoder_output_activation=None):

        self.dense_units_list = dense_units_list
        self.batch_normalization = batch_normalization
        self.activation = decoder_output_activation

        self.type = "Dense"

        super(DecoderDense, self).__init__(input_shape)

        self.params_dict = {"type": self.type,
                            "input_shape": input_shape,
                            "dense_units_list": dense_units_list,
                            "batch_normalization": batch_normalization,
                            "decoder_output_activation": decoder_output_activation}

    def _build_hidden_list(self):
        with tf.compat.v1.name_scope("DecoderDenseHidden") as scope:
            hidden_layer_list = []
            for num_neurons in reversed(self.dense_units_list):
                hidden_layer_list.append(tf.keras.layers.Dense(num_neurons, activation=None))
                if self.batch_normalization:
                    hidden_layer_list.append(tf.keras.layers.BatchNormalization())
                hidden_layer_list.append(tf.keras.layers.Activation('relu'))
        return hidden_layer_list

    def _build_output(self, hidden):
        x_recon = tf.keras.layers.Dense(np.product(self.input_shape), activation = self.activation)(hidden)
        x_recon_reshaped = tf.keras.layers.Reshape(self.input_shape)(x_recon)
        return x_recon_reshaped






if __name__ == "__main__":
    latent_dim = 2
    input_shape = (64,64,3)
    dense_units_list = [64, 32]


    decoder = DecoderDense(input_shape, dense_units_list = dense_units_list)
    decoder.build_decoder_class_tensors(latent_dim)

    decoder.decoder.summary()

