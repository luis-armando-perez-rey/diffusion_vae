'''
Created on Dec 6, 2018
'''

#System imports
import os

# Standard imports
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import math
from scipy import stats
import itertools

# Plotting libraries
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Project library imports
from modules.deltavae.deltavae_latent_spaces.deltavae_parent import DiffusionVAE

class DiffusionSphereVAE(DiffusionVAE):
    """
    Delta-VAE for the hyperspherical manifold S^d latent space with d degrees of freedom.
    """

    def __init__(self, params, encoder_class, decoder_class):
        '''
        Constructor
        '''
        params.params_dict["manifold"] = "hypersphere"
        self.latent_dim = params.d + 1 # dimension of ambient space
        self.scale_dim = 1 # dimension of time parameter
        self.d = params.d # degrees of freedom in manifold
        self.S = lambda x: self.d * (self.d - 1) # scalar curvature
        self.volume = volume_sphere(self.d) # manifold volume

        # Distributions and densities
        self.decoding_distribution = stats.multivariate_normal
        self.log_prior = np.log(1 / self.volume)
        super(DiffusionSphereVAE, self).__init__(params, encoder_class, decoder_class)

    def kl_tensor(self, log_t, mu_z):
        d = self.params.d
        scalar_curv = d * (d - 1)
        volume = self.volume
        loss = -d * log_t / 2.0 - d * np.log(2.0 * np.pi) / 2.0 - d / 2.0 + np.log(volume) \
               + scalar_curv * K.exp(log_t) / 4.0
        if self.params.controlled_capacity:
            self.C = tf.Variable(1.0, name="c_constant")
            loss = tf.abs(loss - self.C)
            K.set_value(self.C, 2.0)
        return loss

    def sampling(self, args):
        """
        Reparameterization trick by performing random walk over the manifold.
        :param args: [z_mean projected, z_log_t] location parameter and log-scale of posterior approximate
        :return: z_sample, sampled latent variable on the manifold
        """
        z_mean_projected, z_log_t = args
        z_sample = z_mean_projected
        for k in range(self.steps):
            epsilon = K.random_normal(shape=K.shape(z_mean_projected))
            # Define the step taken
            step = K.exp(0.5 * z_log_t) * epsilon / np.sqrt(self.steps)
            # Project back to the manifold
            z_sample = self.projection(z_sample + step)
        return z_sample

    def projection(self, z):
        """
        This function takes an input latent variable (tensor) in ambient space R^latent_dim and projects it into the
        chosen manifold
        :param z: Input latent variable in R^latent_dim
        :return: Projected latent variable in manifold
        """
        z_proj = tf.nn.l2_normalize(z, axis=-1)
        return z_proj

    # # # # # # # # # #  PLOTTING  FUNCTIONS # # # # # # # # # #

    def save_plot_latent_space(self, x_test, color, batch_size, filename):
        """
        Function for plotting and saving the embeddings of the x_test data in the latent space according to chosen coding.

        :param x_test: test data to be encoded in latent space
        :param color: color coding given to each data point embedded in latent space
        :param batch_size: batch size for encoding data
        :param filename: saving filename for image
        :return: boolean that indicates wether the latent space was plotted
        """
        assert len(x_test) == len(color), "Data and color coding are not of the same size"
        z_mean, _, _ = self.encoder.predict(x_test,batch_size=batch_size)
        if self.d == 2:
            fig = plt.figure(figsize=(5, 5))
            ax = Axes3D(fig)
            # Plotting of embeddings
            ax.scatter(z_mean[:, 0], z_mean[:, 1], z_mean[:, 2], c=color)

            # Create grid to show sphere cover
            samples = 20
            theta = 2 * np.pi * np.linspace(0, 1, samples)
            phi = np.pi * np.linspace(0, 1, samples)
            phi_grid, theta_grid = np.meshgrid(phi, theta)
            x = np.sin(phi_grid) * np.cos(theta_grid)
            y = np.sin(phi_grid) * np.sin(theta_grid)
            z = np.cos(phi_grid)

            # Plotting of sphere cover
            ax.plot_surface(x, y, z, rstride=1, cstride=1, color='r', alpha=0.1, linewidth=0, shade=True)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_zlim([-1, 1])

        elif self.d == 1:
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111)
            # Plotting of embeddings
            ax.scatter(z_mean[:, 0], z_mean[:, 1], c=color)
            circle = plt.Circle((0, 0), 1.01, color='k', fill=False)
            ax.add_artist(circle)
            ax.set_ylim([-1.02, 1.02])
            ax.set_xlim([-1.02, 1.02])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")
        else:
            print("Not implemented for d = ", self.d)
            return None
        if filename is not None:
            root_dir = os.path.split(filename)[0]
            os.makedirs(root_dir, exist_ok=True)
            plt.savefig(filename, bbox_inches="tight")
        return fig, ax



    def save_plot_prior_reconstruction(self, num_samples, batch_size, filename):
        """
        Plot and save the reconstructions created from decoding samples of the latent space according to the prior.
        :param num_samples: number of samples from latent space
        :param batch_size: batch size for the decoding of the samples
        :param filename: filename for saving the plotted reconstructed images
        :return:
        """
        latent_samples = np.random.normal(0.0, 1.0, (num_samples ** 2, self.latent_dim))
        latent_samples_normalized = latent_samples / np.linalg.norm(latent_samples, axis=-1)[:, np.newaxis]
        decoded = self.decode(latent_samples_normalized, batch_size=batch_size)
        decoded_reshaped = decoded.reshape((-1, int(np.sqrt(self.image_size)), int(np.sqrt(self.image_size))))
        fig = plt.figure(figsize=(5, 5))
        for i in range(len(decoded_reshaped)):
            plt.subplot(num_samples, num_samples, i + 1)
            plt.imshow(decoded_reshaped[i], cmap="gray")
            plt.xticks([])
            plt.yticks([])
        if filename is not None:
            root_dir = os.path.split(filename)[0]
            os.makedirs(root_dir, exist_ok=True)
            plt.savefig(filename, bbox_inches="tight")
        return fig

    def save_plot_image_reconstruction(self, batch_size, filename, samples):
        """
        Plot and save the images obtained by decoding a regular grid over the latent variable manifold
        :param batch_size: batch size for the decoding
        :param filename: filename where the images are saved
        :param samples: square root of the number of points in the regular grid used
        :return:
        """
        if self.d == 2:
            theta = 2 * np.pi * np.linspace(0, 1, samples)
            phi = np.pi * np.linspace(0, 1, samples)
            combinations = []
            for i in itertools.product(theta, phi):
                combinations.append(i)
            combinations = np.array(combinations)
            coordinates = np.zeros((len(combinations), 3))
            coordinates[:, 0] = np.cos(combinations[:, 0]) * np.sin(combinations[:, 1])
            coordinates[:, 1] = np.sin(combinations[:, 0]) * np.sin(combinations[:, 1])
            coordinates[:, 2] = np.cos(combinations[:, 1])
            images_decoded = self.decode(coordinates, batch_size)
            # Plot the reconstructed ciphers
            fig = plt.figure(figsize=(5, 5))
            for i in range(samples):
                for j in range(samples):
                    ax = fig.add_subplot(samples, samples, j * samples + i + 1)
                    if images_decoded.shape[-1] != 3:
                        ax.imshow(images_decoded[i * samples + j, :, :, 0], cmap="gray")
                    else:
                        ax.imshow(images_decoded[i * samples + j])
                    ax.set_xticks([])
                    ax.set_yticks([])

        elif self.d == 1:
            theta = 2 * np.pi * np.linspace(0, 1, samples)
            coordinates = np.zeros((samples, 2))
            coordinates[:, 0] = np.cos(theta)
            coordinates[:, 1] = np.sin(theta)
            images_decoded = self.decode(coordinates, batch_size)
            # Plot the reconstructed ciphers
            fig = plt.figure(figsize=(50, 5))
            for i in range(samples):
                ax = fig.add_subplot(1, samples, i + 1)
                if images_decoded.shape[-1] != 3:
                    ax.imshow(images_decoded[i, :, :, 0], cmap="gray")
                else:
                    ax.imshow(images_decoded[i])
                ax.set_xticks([])
                ax.set_yticks([])
        else:
            print("Not implemented for other latent dimensions other than 2 and 3, d = ", self.d)
            return None
        if filename is not None:
            root_dir = os.path.split(filename)[0]
            os.makedirs(root_dir, exist_ok=True)
            plt.savefig(filename, bbox_inches='tight')
        return fig



    def posterior_density_approximation(self, t, mu_z, z):
        """
        Estimate the value of the density on the point z for the posterior with location mu_z, scale t
        :param t: scale parameter of the posterior distribution
        :param mu_z: location parameter of the posterior
        :param z: latent variable z where the density is estimated
        :return: posterior_density value for each latent variable z
        """
        r = np.linalg.norm(mu_z - z, axis=-1)
        front_coefficient = (1 / (2 * np.pi * t) ** (self.d / 2)) * np.exp(-r ** 2 / (2 * t))
        second_coefficient = self.S(z) * t / (8 * self.d * r ** 2)
        third_coefficient = 3 - self.d + (self.d - 1) * r ** 2 + (self.d - 3) * r * (1 / np.tan(r))
        posterior_density = front_coefficient * (1 + (second_coefficient * third_coefficient))
        return posterior_density


def volume_sphere(d):
    """ Compute volume of d-sphere
     eps = 0.00001

     np.abs(volume_sphere(1) - 2*math.pi) < eps
        True

     np.abs(volume_sphere(2) - 4*math.pi) < eps
        True

     np.abs(volume_sphere(3) - 2*math.pi**2) < eps
        True
    """

    latent_dim = d + 1

    if latent_dim % 2 == 0:
        k = latent_dim / 2
        volume = latent_dim * math.pi ** k / math.factorial(k)
    else:
        k = (latent_dim - 1) / 2
        volume = latent_dim * 2 * math.factorial(k) * (4 * math.pi) ** k / math.factorial(2 * k + 1)

    return volume


if __name__ == "__main__":
    import doctest

    doctest.testmod()
