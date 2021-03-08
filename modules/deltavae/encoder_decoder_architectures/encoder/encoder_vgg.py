from modules.deltavae.encoder_decoder_architectures.encoder import encoder_parent
import tensorflow as tf


class EncoderVGG(encoder_parent.Encoder):



    def __init__(self,input_shape, filter_list=[64, 64, 64],
                       kernel_size_list=[(3, 3), (3, 3), (3, 3)],
                       pool_size_list=[(2, 2), (2, 2), (2, 2)],
                     dense_units_list=[64],
                     min_log_t = -7.5,
                     max_log_t= -5.0,
                 unconstrained_t=False):
        self.filter_list = filter_list
        self.kernel_size_list = kernel_size_list
        self.pool_size_list = pool_size_list
        self.dense_units_list = dense_units_list


        self.type = "VGG"

        super(EncoderVGG, self).__init__(input_shape,  min_log_t, max_log_t, unconstrained_t)

        self.intermediate_shape = self._calculate_intermediate_shape()
        self.params_dict = {"type": self.type,
                            "input_shape": input_shape,
                            "filter_list": self.filter_list,
                            "kernel_size_list": self.kernel_size_list,
                            "pool_size_list": self.pool_size_list,
                            "dense_units_list": self.dense_units_list,
                            "intermediate_shape": self.intermediate_shape,
                            "max_log_t": max_log_t,
                            "min_log_t": min_log_t,
                            "unconstrained_t": unconstrained_t
                            }

    def _calculate_intermediate_shape(self):
        vertical = self.input_shape[0]
        horizontal = self.input_shape[1]
        for num_pool in range(len(self.pool_size_list)):
            if not (self.pool_size_list[num_pool] is None):
                vertical_pool = self.pool_size_list[num_pool][0]
                horizontal_pool = self.pool_size_list[num_pool][1]
                vertical = vertical // vertical_pool
                horizontal = horizontal // horizontal_pool
        final_filters = self.filter_list[-1]
        intermediate_shape = (vertical, horizontal, final_filters)
        return intermediate_shape

    def _build_latent_list(self):
        with tf.compat.v1.name_scope("EncoderVGGHidden") as scope:
            hidden_list = []

            # Loop through all the convolutions in the VGG
            for num_filter, filters in enumerate(self.filter_list):
                hidden_list.append(tf.keras.layers.Conv2D(filters=filters,
                                       kernel_size=self.kernel_size_list[num_filter],
                                       strides=(1, 1),
                                       padding="same",
                                       activation = None))
                hidden_list.append(tf.keras.layers.BatchNormalization())
                hidden_list.append(tf.keras.layers.Activation('relu'))

                # Do max pooling in the corresponding case. If an element of pool size list is none then no max pooling is done
                if not (self.pool_size_list[num_filter] is None):
                    hidden_list.append(tf.keras.layers.MaxPooling2D(pool_size= self.pool_size_list[num_filter], padding="same"))

            # Dense layers after the convolutions
            hidden_list.append(tf.keras.layers.Flatten())
            for num_units, units in enumerate(self.dense_units_list[:-1]):
                hidden_list.append(tf.keras.layers.Dense(units, activation=None))
                hidden_list.append(tf.keras.layers.BatchNormalization())
                hidden_list.append(tf.keras.layers.Activation('relu'))
            hidden_list.append(tf.keras.layers.Dense(self.dense_units_list[-1], activation=None))
            hidden_list.append(tf.keras.layers.Activation('relu'))
        return hidden_list

    #
    # def _build_hidden(self):
    #     """
    #     Defines the network that calculates the parameters of the posterior distribution over the labels with respect
    #     given the input images q(z|x)
    #     :param inputs_images: tensorflow input layer that is connected to the hidden_list layers of this encoder
    #     :param label_shape: shape of the labels
    #     :param filter_list: list of filters for each of the convolutional layers in the VGG network
    #     :param kernel_size_list: list of kernel sizes for each fo the convolutional layers in the VGG network (tuples)
    #     :param pool_size_list: list of pooling sizes for each of the pooling layers in the VGG network (tuples)
    #     :param dense_units_list: list of number of neurons of dense layers at the end of the VGG network
    #     :param max_log_t: maximum log_t value for the scale parameter over the circle
    #     :param min_log_t: minimum log_t value for the scale parameter over the circle
    #     :return:
    #     """
    #     # Define the input image layer as the first layer in the encoder
    #     with tf.name_scope("EncoderVGGHidden") as scope:
    #         hidden = self.inputs
    #
    #         # Loop through all the convolutions in the VGG
    #         for num_filter, filters in enumerate(self.filter_list):
    #             hidden = tf.keras.layers.Conv2D(filters=filters,
    #                                    kernel_size=self.kernel_size_list[num_filter],
    #                                    strides=(1, 1),
    #                                    padding="same",
    #                                    activation = None)(hidden)
    #             hidden = tf.keras.layers.BatchNormalization()(hidden)
    #             hidden = tf.keras.layers.Activation('relu')(hidden)
    #
    #             # Do max pooling in the corresponding case. If an element of pool size list is none then no max pooling is done
    #             if not (self.pool_size_list[num_filter] is None):
    #                 hidden = tf.keras.layers.MaxPooling2D(pool_size= self.pool_size_list[num_filter], padding="same")(hidden)
    #
    #         # Dense layers after the convolutions
    #         hidden = tf.keras.layers.Flatten()(hidden)
    #         for num_units, units in enumerate(self.dense_units_list[:-1]):
    #             hidden = tf.keras.layers.Dense(units, activation=None)(hidden)
    #             hidden = tf.keras.layers.BatchNormalization()(hidden)
    #             hidden = tf.keras.layers.Activation('relu')(hidden)
    #         hidden = tf.keras.layers.Dense(self.dense_units_list[-1], activation=None)(hidden)
    #         hidden = tf.keras.layers.Activation('relu')(hidden)
    #     return hidden





if __name__ == "__main__":
    input_shape = (64,64,3)
    filter_list = [64, 64, 64]
    kernel_size_list = [(3, 3), (3, 3), (3, 3)]
    pool_size_list = [None, (2, 2), None]
    dense_units_list = [64]
    min_log_t = -7.5
    max_log_t = -5.0
    unconstrained_t = False
    latent_dim = 2
    scale_dim = 1
    encoder_class = EncoderVGG(input_shape, filter_list, kernel_size_list, pool_size_list,dense_units_list,min_log_t, max_log_t,unconstrained_t)
    encoder_class.build_encoder_class_tensors(latent_dim, scale_dim)
    encoder_class.encoder.summary()
    print("Intermediate shape",encoder_class.intermediate_shape)


