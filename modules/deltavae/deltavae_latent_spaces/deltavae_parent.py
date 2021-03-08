# System imports
import sys
import time

# Standard imports
import tensorflow as tf
import tensorflow.keras.backend as K

# Plotting libraries
import numpy as np
from scipy import special


# Modules
# from modules.callbacks.capacity_callback_true import CapacityCallback

class DiffusionVAE:
    """
    Diffusion variational autoencoder (DeltaVAE) parent class.
    """

    def __init__(self, diffusion_vae_params, encoder_class, decoder_class,symmetric=False):
        """
        DeltaVAE initialization based on general diffusion vae parameters class, encoder class and decoder class

        :param diffusion_vae_params: parameter class
        :param encoder_class: determines the architecture of the encoder
        :param decoder_class: determines the architecture of the decoder
        """

        # Parameters of DeltaVAE
        self.params = diffusion_vae_params
        self.symmetrize = symmetric  # this parameter is necessary for RPN

        # Network classes
        self.encoder_class = encoder_class
        self.decoder_class = decoder_class

        # Delta VAE special parameters
        self.steps = self.params.steps  # how many steps to take in random-walk sampling
        self.truncation_radius = self.params.truncation_radius  # (tanh-implemented) truncation radius for sampling
        self.optimizer = diffusion_vae_params.optimizer  # optimizer for training
        self.r_loss = diffusion_vae_params.r_loss  # type of reconstruction loss

        # Decoder parameter
        self.var_x = diffusion_vae_params.var_x  # decoding distribution variance (normal distribution)

        # Creat the Delta VAE encoder, decoder and autoencoder networks
        self.encoder, self.decoder, self.vae = self._build_network()

    def _build_encoder(self):
        """
        This function defines the encoder model and the corresponding output tensors of the encoder
        :return: encoder: tf.keras model
         z: tensor for sample latent variable
         z_mean_projected: location parameter of the posterior approximate on the manifold
         z_log_t: scale parameter of the posterior approximate
        """
        # Get the input tensor
        self.encoder_class.build_encoder_class_tensors(self.latent_dim, self.scale_dim)
        # Get the output tensors
        z_mean = self.encoder_class.outputs[0]
        z_log_t = self.encoder_class.outputs[1]
        # Project the location parameter
        z_mean_projected = tf.keras.layers.Lambda(self.projection, output_shape=(self.latent_dim,), name="z_projected")(
            z_mean)
        # Tensor for the sample according to the prior
        z = tf.keras.layers.Lambda(self.sampling, output_shape=(self.latent_dim,), name='z')([z_mean_projected, z_log_t])
        # Define the encoder model
        encoder = tf.keras.models.Model(self.encoder_class.inputs, [z_mean_projected, z_log_t, z])
        return encoder, z, z_mean_projected, z_log_t

    def _build_decoder(self, z):
        """
        This function defines the decoder model and the corresponding output tensors of the decoder
        :param z: the tensor corresponding to a sampled latent variable from the posterior approximate
        :return: decoder: model for the decoder
        outputs_vae: tensor corresponding to the location parameter of the decoding distribution
        """
        # Get the inputs tensor
        self.decoder_class.build_decoder_class_tensors(self.latent_dim)

        # Delta-VAE outputs
        if self.symmetrize:
            def average(args):
                z1, z2 = args
                avg = 0.5 * (z1 + z2)
                return avg
            neg_z = tf.keras.layers.Lambda(lambda s: -s)(self.decoder_class.inputs)
            pos_output = self.decoder_class.outputs
            neg_hidden = self.decoder_class._build_hidden(neg_z)
            neg_output = self.decoder_class._build_output(neg_hidden)
            outputs_vae = tf.keras.layers.Lambda(average)([pos_output, neg_output])
            decoder = tf.keras.models.Model(self.decoder_class.inputs, outputs_vae, name = "Decoder"+self.decoder_class.type)
        else:
            decoder = tf.keras.models.Model(self.decoder_class.inputs, self.decoder_class.outputs,
                                         name="Decoder" + self.decoder_class.type)
        outputs_vae = decoder(z)
        return decoder, outputs_vae

    # def build_decoder(self, z):
    #     ################################################################################################################
    #     # DECODING
    #     ################################################################################################################
    #     decoder_layers = []
    #     if self.params.type_layers == "convolutional":
    #         decoder_layers.append(Dense(np.product(self.intermediate_conv_shape[1:])))
    #         decoder_layers.append(Reshape((self.intermediate_conv_shape[1],
    #                                        self.intermediate_conv_shape[2],
    #                                        self.intermediate_conv_shape[3])))
    #     for layer in range(self.params.num_decoding_layers - 1):
    #         if self.params.type_layers == "convolutional":
    #             decoder_layers.append(UpSampling2D(size=(2, 2), data_format=None, name="upsample_" + str(layer)))
    #             decoder_layers.append(Conv2D(filters=self.intermediate_dim * 2 ** (
    #                     self.params.num_decoding_layers - layer - 1),
    #                                          kernel_size=self.params.kernel_size,
    #                                          strides=1,
    #                                          padding='same',
    #                                          decoder_output_activation='relu',
    #                                          name="h_dec_c_" + str(layer)))
    #         elif self.params.type_layers == "dense":
    #             decoder_layers.append(Dense(self.intermediate_dim//(2**(self.params.num_decoding_layers-layer)),
    #                                         decoder_output_activation='relu',
    #                                         name="h_dec_d_" + str(layer)))
    #
    #     if self.r_loss == 'mse':
    #         output_activation = 'linear'
    #     elif self.r_loss == 'binary':
    #         output_activation = 'sigmoid'
    #     else:
    #         print("Loss not appropriately chosen")
    #         output_activation = 'none'
    #
    #     if self.params.type_layers == "convolutional":
    #         decoder_layers.append(UpSampling2D(size=(2, 2), data_format=None, name="upsample_" + str(layer+1)))
    #         outputs_def = Conv2D(filters= self.input_shape[2],
    #                              kernel_size=self.params.kernel_size,
    #                              strides=1,
    #                              padding='same',
    #                              decoder_output_activation='relu',
    #                              name="outputs")
    #     elif self.params.type_layers == "dense":
    #         decoder_layers.append(Dense(self.intermediate_dim,
    #                                     decoder_output_activation = 'relu',
    #                                     name="h_dec_d_"+str(self.params.num_decoding_layers)))
    #         decoder_layers.append(Dense(np.product(self.input_shape),
    #                             decoder_output_activation=output_activation,
    #                             name="outputs"))
    #         outputs_def = Reshape(self.input_shape)
    #
    #     # STANDALONE DECODING
    #     latent_inputs = Input(shape=(self.latent_dim,), name='input_latent')
    #     standalone_decoder_layers = [latent_inputs]
    #     for layer in decoder_layers:
    #         standalone_decoder_layers.append(layer(standalone_decoder_layers[-1]))
    #     _outputs_standalone = outputs_def(standalone_decoder_layers[-1])
    #
    #     def average(args):
    #         z1, z2 = args
    #         avg = 0.5 * (z1 + z2)
    #         return avg
    #
    #     # Symmetrize routine for RPN manifold
    #     if self.symmetrize:
    #         _latent_inputs_reflected = Lambda(lambda s: -s)(latent_inputs)
    #         reversed_decoding_layers = [_latent_inputs_reflected]
    #         for layer in decoder_layers:
    #             reversed_decoding_layers.append(layer(reversed_decoding_layers[-1]))
    #         _outputs_reversed = outputs_def(reversed_decoding_layers[-1])
    #         _outputs = Lambda(average)([_outputs_standalone, _outputs_reversed])
    #         decoder = Model(latent_inputs, _outputs, name='decoder')
    #         outputs_vae = decoder(z)
    #     else:
    #         decoder = Model(latent_inputs, _outputs_standalone, name='decoder')
    #         outputs_vae = decoder(z)
    #
    #     return decoder, outputs_vae

    def _build_loss_functions(self, z_log_scale, z_mean_projected):
        """
        This function creates the metrics and loss functions used during training and evaluation
        :param z_log_scale:
        :param z_mean_projected:
        :return:
        """
        metrics = []

        def kl_loss(inputs, outputs):
            """
            Kullback-Leibler divergence of posterior distribution. Depends directly on the location and scale parameters
            of the posterior approximate
            :param inputs : not used
            :param outputs: not used
            :return:
            """
            loss = self.kl_tensor(z_log_scale, z_mean_projected)
            return loss

        metrics.append(kl_loss)

        def r_loss(inputs, outputs):
            """
            Reconstruction loss part of the variational autoencoder
            :param inputs: input data
            :param outputs: output data from the autoencoder
            :return: r_loss tensor
            """
            if self.r_loss == "mse":
                #print("Reconstruction loss is mean squared error")
                se = (outputs - tf.cast(inputs, tf.float32)) ** 2
                # Sum over the data dimensions
                for dimension in range(len(self.encoder_class.input_shape)):
                    se = K.sum(se, axis=-1)
                loss = 0.5 * (se / self.var_x + tf.cast(np.product(self.encoder_class.input_shape), tf.float32) * np.log(
                    2 * np.pi * self.var_x))

            elif self.r_loss == "binary":
                #print("Reconstruction loss is binary cross entropy")
                epsilon = K.epsilon()
                loss = -tf.cast(inputs, tf.float32) * tf.math.log(epsilon + outputs) \
                       - (1.0 - tf.cast(inputs, tf.float32)) * tf.math.log(epsilon + 1.0 - outputs)
                # Sum over the data dimensions
                for dimension in range(len(self.encoder_class.input_shape)):
                    loss = tf.reduce_sum(input_tensor=loss, axis=-1)
            else:
                print("Error, no reconstruction chosen")
                loss = None
            return loss

        metrics.append(r_loss)

        def mean_squared_error(inputs, outputs):
            """
            Calculates the mean squared error between input data and output data
            :param inputs:
            :param outputs:
            :return: r_loss tensor
            """
            calculated_mse = tf.keras.losses.mse(inputs, outputs)
            return calculated_mse

        metrics.append(mean_squared_error)

        def vae_loss(inputs, outputs):
            loss = K.mean(r_loss(inputs, outputs) + kl_loss(inputs, outputs))
            return loss

        return metrics, vae_loss

    def _build_network(self):
        """
        Creates the Delta-VAE architectures and joins them into the Delta-VAE model together with the corresponding
        loss functions and optimizers.
        :return:
        """
        # Create encoder model
        encoder, z, z_mean_projected, z_log_t = self._build_encoder()
        # Create decoder model
        decoder, outputs = self._build_decoder(z)
        # Create metrics and loss
        metrics, vae_loss = self._build_loss_functions(z_log_t, z_mean_projected)
        # Create Delta-VAE model
        vae = tf.keras.models.Model(self.encoder_class.inputs, outputs, name='vae_mlp')
        vae.compile(optimizer=self.optimizer, loss=[vae_loss], metrics=metrics, experimental_run_tf_function=False)
        # Print summary of networks
        vae.summary()
        encoder.summary()
        decoder.summary()
        return encoder, decoder, vae

    def return_parameters_dict(self):
        """
        This function returns the corresponding dictionaries of the parameters used to create the Delta-VAE. i.e.
        encoder_class parameters, decoder_class parameters and diffusion_vae parameters
        :return:
        """
        return self.encoder_class.params_dict, self.decoder_class.params_dict, self.params.params_dict

    def train_vae(self, train_data, target_data, epochs, batch_size, weights_file, callback_list=None):
        """

        :param callback_list:
        :param train_data:
        :param target_data:
        :param train_data (numpy array): first dimension corresponds to number of datapoints
        while second dimension corresponds to the size of each the datapoint
        :param epochs (int) number of epochs the diffusion vae is trained
        :param batch_size (int) size of the batch used for training each epoch
        :param weights_file (str) complete path for saving the trained weights
        :return:
        """

        self.vae.fit(train_data, target_data,
                     epochs=epochs,
                     batch_size=batch_size,
                     callbacks=callback_list,
                     verbose=2
                     )
        self.vae.save_weights(weights_file)

    def train_vae_checkpoints(self, train_data, epochs, batch_size, weights_file, tensorboard_file, models_filepath):
        """
        Train diffusion variational autoencoder that can
        :param train_data: first dimension corresponds to number of datapoints
        while second dimension corresponds to the size of the datapoint
        :param epochs (int) number of epochs the diffusion vae is trained
        :param batch_size (int) size of the batch used for training each epoch
        :param weights_file (str) complete path for saving the trained weights
        :param tensorboard_file (str) complete path for saving the tensorboard log
        :param models_filepath (str) path to where the diffusion vae models are to be saved
        :return:
        """
        checkpoint = tf.keras.callbacks.ModelCheckpoint(models_filepath, verbose=0, save_best_only=False,
                                                     save_weights_only=True, mode='auto', period=10)

        tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_file)
        self.vae.fit(train_data, train_data,
                     epochs=epochs,
                     batch_size=batch_size,
                     callbacks=[tensorboard_cb, checkpoint],
                     verbose=2
                     )
        self.vae.save_weights(weights_file)

    def train_generator_vae(self, generator, epochs, weights, tensorboard_file):
        """
        Train the diffusion vae whose data is generated from a generator
        :param generator: data generator object that can be used with fit_generator
        :param epochs:
        :param weights:
        :param tensorboard_file:
        :return:
        """
        tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_file)
        self.vae.fit_generator(generator,
                               epochs=epochs,
                               verbose=2,
                               callbacks=[tensorboard_cb],
                               workers=0,
                               use_multiprocessing=False)
        self.vae.save_weights(weights)

    def load_model(self, weight_file):
        """
        Reload the weights of previously trained models
        :return:
        """
        self.vae.load_weights(weight_file)

    def encode_location(self, data, batch_size):
        """
        Encode into the latent space the input data
        :param batch_size:
        :param data (numpy array) first dimension of array corresponds to the number of
        datapoints and the second correspond to the size of each datapoint
        :return:
        """
        encoded = self.encoder.predict(data, batch_size=batch_size)[0]
        return encoded

    def encode_scale(self, data, batch_size):
        time = np.exp(self.encoder.predict(data, batch_size=batch_size)[1])
        return time

    def encode_log_scale(self, data, batch_size):
        log_time = self.encoder.predict(data, batch_size=batch_size)[1]
        return log_time

    def decode(self, latent, batch_size):
        decoded = self.decoder.predict(latent, batch_size=batch_size)
        return decoded

    def autoencode(self, x_test, batch_size):
        autoencoded = self.vae.predict(x_test, batch_size=batch_size)
        return autoencoded

    def sample_latent_posterior(self, data, batch_size=128, num_samples=1):
        """
        Sample a certain number of latent variables from the latent space according to the
        posterior with respect to input data
        :param batch_size:
        :param num_samples:
        :return:
        """
        assert num_samples >= 1, "Samples must be an integer greater equal than one"
        samples = np.zeros((len(data), num_samples, self.latent_dim))
        y, log_t, samples[:, 0, :] = self.encoder.predict(data, batch_size=batch_size)
        t = np.exp(log_t)
        for sample in range(num_samples - 1):
            samples[:, sample + 1, :] = self.encoder.predict(data, batch_size=batch_size)[2]
        return y, t, samples

    def evaluate_metrics(self, x_train, batch_size):
        values = self.vae.evaluate(x=x_train, y=x_train, batch_size=batch_size)
        values = np.array(values)
        return values

    def squared_error(self, x_train, batch_size):
        encoded = self.autoencode(x_train, batch_size)
        squared_error = np.mean(np.sum((encoded - x_train) ** 2, axis=-1), axis=-1)
        return squared_error

    def calculate_log_p_xgz(self, data, decoded_z_samples):
        if self.r_loss == "mse":
            data_size = np.product(data.shape[1:])
            log_exponent = ((data[:, np.newaxis, :] - decoded_z_samples) ** 2) / (2 * self.var_x)
            for dimension in range(len(data.shape[1:])):
                log_exponent = np.sum(log_exponent, axis=-1)

            log_determinant = data_size * np.log(2 * np.pi * self.var_x) / 2
            log_p = -log_determinant - log_exponent
        elif self.r_loss == "binary":
            # log_p = np.zeros(len(data))
            # for num_sample in range(decoded_z_samples.shape[1]):
            #     log_p += np.sum((1-data)*(np.log(1 - decoded_z_samples[:,num_sample,:] + 1e-7)) + np.log(
            #     1e-7 + decoded_z_samples[:,num_sample,:]) * data, axis=-1)

            log_p = (1 - data[:, np.newaxis, :]) * (np.log(1 - decoded_z_samples + 1e-7)) + np.log(
                1e-7 + decoded_z_samples) * data[:, np.newaxis, :]
            for dimension in range(len(data.shape[1:])):
                log_p = np.sum(log_p, axis=-1)
        else:
            log_p = None
        return log_p

    def calculate_log_q_zgx(self, z_samples, t, encoded):
        """
        Estimate the log_posterior for the z_samples obtained with respect to each of the encoded values
        """
        # for num_sample in range(z_samples.shape[1]):
        #    r = np.linalg.norm(z_samples[:,num_sample,:]- encoded, axis = -1)
        #    log_q = -0.5 * self.d * np.log(2*np.pi*(t))
        #    log_q += -r ** 2 / (2 * (t))

        r = np.linalg.norm(z_samples - encoded[:, np.newaxis, :], axis=-1)
        log_term1 = -0.5 * self.d * np.log(2.0 * np.pi * (t))
        log_term2 = -r ** 2.0 / (2.0 * (t))
        coefficient1 = self.S(z_samples) * t / (8.0 * self.d * (r) ** 2)
        coefficient2 = 3 - self.d + (self.d - 1.0) * r ** 2.0 + (self.d - 3.0) * r / np.tan(r)
        log_term3 = np.log(1 + coefficient1 * coefficient2)
        log_q = log_term1 + log_term2 + log_term3
        return log_q

    def estimate_log_likelihood(self, data, batch_size, num_samples):
        """
        Estimates the log-likelihood with weighted importance for a given dataset
        with respect to a certain number of samples from the latent space accoding
        to the approximate posterior
        :param data (numpy array): first dimension corresponds to number of datapoints
        while second dimension corresponds to the size of each the datapoint
        :param batch_size (int): size of the batch for producing the samples
        :param num_samples (int): number of samples taken from the latent space according
        to the approximate posterior distribution
        :return: estimate: corresponds to the estimate log-likelihood value
        """
        datapoint_batch_size = 1000
        estimate = np.zeros(datapoint_batch_size)
        for batch in range(int(len(data) / datapoint_batch_size)):
            print("Evaluating batch {}".format(batch))
            time_start = time.clock()
            encoded, t, z_samples = self.sample_latent_posterior(
                data[batch * datapoint_batch_size:(batch + 1) * datapoint_batch_size], batch_size, num_samples)
            decoded_z_samples = np.zeros((datapoint_batch_size, num_samples, *data.shape[1:]))
            time_elapsed = (time.clock() - time_start)
            print("Shape encoded {}".format(encoded.shape))
            print("Shape z samples {}".format(z_samples.shape))
            print("Time elapsed sampling {} seconds".format(time_elapsed))
            sys.stdout.flush()
            time_start = time.clock()
            for num_sample in range(num_samples):
                sample = z_samples[:, num_sample, :]
                decoded_z_samples[:, num_sample, ...] = self.decode(sample, batch_size=len(data))
            time_elapsed = (time.clock() - time_start)
            print("Shape of decoded z samples {}".format(decoded_z_samples.shape))
            print("Time elapsed decoding {} seconds".format(time_elapsed))
            sys.stdout.flush()
            time_start = time.clock()

            # log_p = np.sum((1 - data[:, np.newaxis, :]) * (np.log(1 - decoded_z_samples + 1e-7)) + np.log(
            #    1e-7 + decoded_z_samples) * data[:, np.newaxis, :], axis=-1)

            logarithm_terms = -self.calculate_log_q_zgx(z_samples, t, encoded)
            logarithm_terms += self.calculate_log_p_xgz(
                data[batch * datapoint_batch_size:(batch + 1) * datapoint_batch_size], decoded_z_samples)
            logarithm_terms += self.log_prior

            logarithm_terms -= np.log(num_samples)
            # log_p_xgz = self.calculate_log_p_xgz(data, decoded_z_samples)
            # log_q_zgx = self.calculate_log_q_zgx(z_samples, t, encoded)
            # log_p_z = self.log_prior
            # weight_estimate = log_p_xgz + log_p_z - log_q_zgx - np.log(num_samples)
            time_elapsed = (time.clock() - time_start)
            print("Time elapsed calculating {} seconds".format(time_elapsed))
            print("Shape of preliminary estimate {}".format(logarithm_terms.shape))
            sys.stdout.flush()
            estimate += special.logsumexp(logarithm_terms, axis=-1)

        estimate = np.sum(estimate, axis=-1) / len(data)
        return estimate

    def estimate_log_likelihood_slow(self, data, batch_size, num_samples):
        """
        Estimates the log-likelihood with weighted importnace for a given dataset
        with respect to a certain number of samples from the latent space accoding
        to the approximate posterior
        :param data (numpy array): first dimension corresponds to number of datapoints
        while second dimension corresponds to the size of each the datapoint
        :param batch_size (int): size of the batch for producing the samples
        :param num_samples (int): number of samples taken from the latent space according
        to the approximate posterior distribution
        :return: estimate: corresponds to the estimate log-likelihood value
        """
        estimate_per_datum = np.zeros(len(data))
        for num_datum, datum in enumerate(data):
            print("Estimating for datum {}".format(num_datum))
            time_start = time.clock()
            encoded, t, z_samples = self.sample_latent_posterior(np.expand_dims(datum, axis=0), batch_size, num_samples)
            time_elapsed = (time.clock() - time_start)
            print(z_samples.shape, time_elapsed)
            sys.stdout.flush()
            # decoded_z_samples = np.zeros((1, num_samples, data.shape[1]))
            time_start = time.clock()
            decoded_z_samples = self.decode(z_samples[0, :, :], batch_size=len(data))
            time_elapsed = (time.clock() - time_start)
            decoded_z_samples = np.expand_dims(decoded_z_samples, axis=0)
            print(decoded_z_samples.shape, time_elapsed)
            sys.stdout.flush()
            # for num_sample in range(num_samples):
            #    sample = z_samples[:, num_sample, :]
            #    decoded_z_samples[:, num_sample, :] = self.decode(sample, batch_size=len(data))
            time_start = time.clock()
            log_p_xgz = self.calculate_log_p_xgz(np.expand_dims(datum, axis=0), decoded_z_samples)
            log_q_zgx = self.calculate_log_q_zgx(z_samples, t, encoded)
            log_p_z = self.log_prior
            weight_estimate = log_p_xgz + log_p_z - log_q_zgx - np.log(num_samples)

            estimate_per_datum[num_datum] = np.mean(special.logsumexp(weight_estimate, axis=-1), axis=-1)
            time_elapsed = (time.clock() - time_start)
            sys.stdout.flush()
        estimate = np.mean(estimate_per_datum)
        return estimate
