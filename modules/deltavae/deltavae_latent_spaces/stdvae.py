'''
Created on Dec 6, 2018
'''

# System imports
import sys
import time
import os

# Standard imports
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

# Plotting libraries
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Project library imports
from modules.deltavae.deltavae_latent_spaces.deltavae_parent import DiffusionVAE
from scipy import stats, special
import itertools


class StandardVAE(DiffusionVAE):
    '''
    classdocs
    '''

    def __init__(self, params, encoder_class, decoder_class):
        '''
        Constructor
        '''
        params.params_dict["manifold"] = "standard"

        # In the vanilla VAE the parameter d is treated as the latent dimension
        self.latent_dim = params.d
        self.scale_dim = self.latent_dim  # dimension of the standard deviation of the posterior

        # Distributions and densities
        self.decoding_distribution = stats.multivariate_normal
        self.prior = stats.multivariate_normal

        # Plotting properties TODO: this might be unstable implementation
        self.x_limits = [-1.0, 1.0]
        self.y_limits = [-1.0, 1.0]
        super(StandardVAE, self).__init__(params, encoder_class, decoder_class)

    def kl_tensor(self, z_log_scale, z_mu):
        loss = 0.5 * tf.reduce_sum(input_tensor=K.exp(z_log_scale) + K.square(z_mu) - z_log_scale - 1, axis=-1)
        if self.params.controlled_capacity:
            self.C = tf.Variable(1.0)
            loss = tf.abs(loss - self.C)
        return loss

    def sampling(self, args):
        """Reparameterization trick by sampling from an isotropic unit Gaussian.

        # Arguments:
            args (tensor): mean and log of variance of Q(z|X)

        # Returns:
            z (tensor): sampled latent vector
        """
        z_mean_projected, z_log_scale = args
        epsilon = K.random_normal(shape=K.shape(z_mean_projected))
        z_sample = z_mean_projected + tf.multiply(epsilon, tf.exp(0.5 * z_log_scale))
        return z_sample

    def projection(self, z):
        """
        This function takes an input latent variable (tensor) in ambient space R^latent_dim and projects it into the
        chosen manifold
        :param z: Input latent variable in R^latent_dim
        :return:
        """
        # For the vanilla VAE no projection is needed
        return z

    # # # # # # # # # #  PLOTTING  FUNCTIONS # # # # # # # # # #

    def save_plot_latent_space(self, x_test, color, batch_size, filename):
        z_mean, _, _ = self.encoder.predict(x_test,
                                            batch_size=batch_size)
        if self.latent_dim == 3:
            fig = plt.figure(figsize=(12, 10))
            ax = Axes3D(fig)
            ax.scatter(z_mean[:, 0], z_mean[:, 1], z_mean[:, 2], c=color)
            min_x = np.amin(z_mean[:, 0])
            max_x = np.amax(z_mean[:, 0])
            min_y = np.amin(z_mean[:, 1])
            max_y = np.amax(z_mean[:, 1])
            min_z = np.amin(z_mean[:, 2])
            max_z = np.amax(z_mean[:, 2])
            ax.set_xlim([min_x, max_x])
            ax.set_ylim([min_y, max_y])
            ax.set_zlim([min_z, max_z])
        elif self.latent_dim == 2:
            fig = plt.figure(figsize=(5, 5))
            ax = plt.gca()
            ax.scatter(z_mean[:, 0], z_mean[:, 1], c= color)
            min_x = np.amin(z_mean[:, 0])
            max_x = np.amax(z_mean[:, 0])
            min_y = np.amin(z_mean[:, 1])
            max_y = np.amax(z_mean[:, 1])
            self.x_limits = [min_x, max_x]
            self.y_limits = [min_y, max_y]
            ax.set_xlim([min_x, max_x])
            ax.set_ylim([min_y, max_y])
        else:
            print("Not implemented for other latent dimensions other than 2 and 3, d = ", self.d)
            return None
        if filename is not None:
            root_dir = os.path.split(filename)[0]
            os.makedirs(root_dir, exist_ok=True)
            root_dir = os.path.split(filename)[0]
            os.makedirs(root_dir, exist_ok=True)
            plt.savefig(filename, bbox_inches='tight')
        return fig, ax



    def save_plot_image_reconstruction(self, batch_size, filename, samples):
        limit = 1.0
        if self.latent_dim == 3:
            sampled_line = np.linspace(-limit, limit, samples)
            combinations = []
            for i in itertools.product(sampled_line, sampled_line):
                combinations.append(i)
            coordinates = np.array(combinations)
            coordinates = np.append(coordinates, np.zeros((len(combinations), 1)), axis=-1)
            images_decoded = self.decode(coordinates, batch_size)
            # Plot the reconstructed ciphers
            fig = plt.figure(figsize=(10, 10))
            for i in range(samples):
                for j in range(samples):
                    ax = fig.add_subplot(samples, samples, j * samples + i + 1)
                    if images_decoded.shape[-1] != 3:
                        ax.imshow(images_decoded[i * samples + j, :, :, 0], cmap="gray")
                    else:
                        ax.imshow(images_decoded[i * samples + j])
                    ax.set_xticks([])
                    ax.set_yticks([])
        if self.latent_dim == 2:
            sampled_line_x = np.linspace(self.x_limits[0], self.x_limits[1], samples)
            sampled_line_y = np.linspace(self.y_limits[0], self.y_limits[1], samples)
            combinations = []
            for i in itertools.product(sampled_line_x, sampled_line_y):
                combinations.append(i)
            coordinates = np.array(combinations)
            images_decoded = self.decode(coordinates, batch_size)
            # Plot the reconstructed ciphers
            fig = plt.figure(figsize=(10, 10))
            for i in range(samples):
                for j in range(samples):
                    ax = fig.add_subplot(samples, samples, j * samples + i + 1)
                    if images_decoded.shape[-1] != 3:
                        ax.imshow(images_decoded[i * samples + j, :, :, 0], cmap="gray")
                    else:
                        ax.imshow(images_decoded[i * samples + j])
                    ax.set_xticks([])
                    ax.set_yticks([])
        else:
            print("Not implemented for other latent dimensions other than 2 and 3, d = ", self.d)
            return None
        if filename is not None:
            root_dir = os.path.split(filename)[0]
            os.makedirs(root_dir, exist_ok=True)
            plt.savefig(filename, bbox_inches='tight')
        return fig, ax

    def plot_prior_reconstruction(self, num_samples, batch_size, filename):
        latent_samples = np.random.normal(0.0, 1.0, (num_samples ** 2, self.latent_dim))
        decoded = self.decode(latent_samples, batch_size=batch_size)
        decoded_reshaped = decoded.reshape((-1, int(np.sqrt(self.image_size)), int(np.sqrt(self.image_size))))
        fig = plt.figure(figsize=(10, 10))
        for i in range(len(decoded_reshaped)):
            plt.subplot(num_samples, num_samples, i + 1)
            plt.imshow(decoded_reshaped[i], cmap="gray")
            plt.xticks([])
            plt.yticks([])
        plt.savefig(filename, bbox_inches='tight')
        if filename is not None:
            root_dir = os.path.split(filename)[0]
            os.makedirs(root_dir, exist_ok=True)
            plt.savefig(filename, bbox_inches='tight')
        return fig

    def posterior_density_approximation(self, t, y, z):
        r = np.linalg.norm(y - z, axis=-1)
        front_coefficient = (1 / (2 * np.pi * t) ** (self.d / 2)) * np.exp(-r ** 2 / (2 * t))
        second_coefficient = self.S * t / (8 * self.d * r ** 2)
        third_coefficient = 3 - self.d + (self.d - 1) * r ** 2 + (self.d - 3) * r * (1 / np.tan(r))
        posterior_density = front_coefficient * (1 + (second_coefficient * third_coefficient))
        return posterior_density

    # def calculate_log_p_xgz(self, data, decoded_z_samples):
    #     if self.r_loss == "mse":
    #         data_size = np.product(data.shape[1:])
    #         log_exponent = ((data[:, np.newaxis, :] - decoded_z_samples) ** 2) / (2 * self.var_x)
    #         for dimension in range(len(data.shape[1:])):
    #             log_exponent = np.sum(log_exponent, axis=-1)
    #
    #         log_determinant = data_size * np.log(2 * np.pi * self.var_x) / 2
    #         log_p = -log_determinant - log_exponent
    #     elif self.r_loss == "binary":
    #         # log_p = np.zeros(len(data))
    #         # for num_sample in range(decoded_z_samples.shape[1]):
    #         #     log_p += np.sum((1-data)*(np.log(1 - decoded_z_samples[:,num_sample,:] + 1e-7)) + np.log(
    #         #     1e-7 + decoded_z_samples[:,num_sample,:]) * data, axis=-1)
    #
    #         log_p = (1 - data[:, np.newaxis, :]) * (np.log(1 - decoded_z_samples + 1e-7)) + np.log(
    #             1e-7 + decoded_z_samples) * data[:, np.newaxis, :]
    #         for dimension in range(len(data.shape[1:])):
    #             log_p = np.sum(log_p, axis=-1)
    #     return log_p

    def calculate_log_q_zgx(self, z_samples, log_var_z, encoded):
        covariance_diag = np.exp(log_var_z)
        inverse_covariance_diag = 1.0 / covariance_diag
        log_exponent = np.sum(
            (inverse_covariance_diag[:, np.newaxis, :] * (z_samples - encoded[:, np.newaxis, :]) ** 2), axis=-1) / 2.0
        log_determinant = np.log(np.prod(2 * np.pi * covariance_diag, axis=-1))[:, np.newaxis] / 2.0
        log_q = -log_determinant - log_exponent
        return log_q

    # def estimate_log_likelihood(self, data, batch_size, num_samples):
    #    encoded, log_var_z, z_samples = self.sample_latent_posterior(data, batch_size, num_samples)
    #    decoded_z_samples = np.zeros((len(data), num_samples, *data.shape[1:]))
    #    prior = stats.multivariate_normal
    #    log_p_z = np.zeros((len(data), num_samples))
    #    for num_sample in range(num_samples):
    #        sample = z_samples[:, num_sample, :]
    #        decoded_z_samples[:, num_sample, ...] = self.decode(sample, batch_size=len(data))
    #        log_p_z[:, num_sample] = prior.logpdf(sample, mean=np.zeros(self.latent_dim), cov=1.0)
    # flat_z_samples = z_samples.reshape((len(encoded) * num_samples, z_samples.shape[2]))
    # flat_decoded_z_samples = self.decode(flat_z_samples, batch_size)
    # decoded_z_samples = flat_decoded_z_samples.reshape((len(encoded), num_samples, data.shape[1]))

    #    log_p_xgz = self.calculate_log_p_xgz(data, decoded_z_samples)
    #    log_q_zgx = self.calculate_log_q_zgx(z_samples, log_var_z, encoded)
    #    # log_p_z = prior.logpdf(flat_z_samples, mean=np.zeros(self.latent_dim), cov=1.0).reshape(
    #    #    (len(data), num_samples))
    #    weight_estimate = log_p_xgz + log_p_z - log_q_zgx - np.log(num_samples)
    #    estimate = np.mean(special.logsumexp(weight_estimate, axis=-1), axis=-1)
    #    return estimate

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
            logarithm_terms += self.prior.logpdf(z_samples, mean=np.zeros(self.latent_dim), cov=1.0)

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
