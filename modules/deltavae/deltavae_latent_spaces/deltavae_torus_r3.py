'''
Created on Dec 6, 2018
'''
# System imports
import os

# Standard imports
import numpy as np
import tensorflow as tf
import keras.backend as K
import math
import itertools

# Plotting libraries
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Project library imports
from modules.deltavae.deltavae_latent_spaces.deltavae_parent import DiffusionVAE

class DiffusionTorusR3(DiffusionVAE):
    '''
    classdocs
    '''

    def __init__(self, params, encoder_class, decoder_class):
        '''
        Constructor
        '''
        params.params_dict["manifold"] = "torus_r3"
        self.latent_dim = 3 # dimension of ambient space
        self.d = 2.0 # degrees of freedom in manifold
        self.scale_dim = 1 # dimension of time parameter
        self.c = 3.0  # radius to the center of tube
        self.a = 0.6  # radius of tube
        self.S = self.calculate_curvature # scalar curvature
        self.volume = 4 * math.pi * 2 * self.c * self.a # manifold volume
        self.log_prior = np.log(1 / self.volume)

        super(DiffusionTorusR3, self).__init__(params, encoder_class, decoder_class)

    def calculate_curvature(self, z_samples):
        length = np.sqrt(np.sum(z_samples ** 2, axis=-1))
        S = (2 * length) / (self.a * (self.c + self.a * length))
        return S

    def kl_tensor(self, logt, y):
        proj_matrix = K.constant([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
        z_projected = K.dot(y, proj_matrix)
        proj_length = K.sqrt(K.sum(z_projected ** 2, axis=-1))
        scaled_proj_length = (proj_length - self.c) / self.a
        # Note: scalar curvature of 2-torus is twice Gauss curvature
        scalar_curv = 2 * scaled_proj_length / (self.a * (self.c + self.a * scaled_proj_length))
        d = 2  # dimension of manifold
        #loss = - 0.5 * d * logt - 0.5 * d \
        #       + ((d + 4) / 24.) * scalar_curv * K.exp(logt) \
        #       + K.log(4 * math.pi * 2 * self.c * self.a)
        # Rebuttal revision
        loss = - 0.5 * d * logt - 0.5 * d \
               + scalar_curv * K.exp(logt)/4 \
               + K.log(4 * math.pi * 2 * self.c * self.a)
        if self.params.controlled_capacity:
            self.C = tf.Variable(1.0)
            loss = tf.abs(loss-self.C)
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
        c = 3.0  # radius to center of tube
        a = 0.6  # radius of tube
        proj_matrix = K.constant([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
        z_circle = c * tf.nn.l2_normalize(K.dot(z, proj_matrix), dim=-1)
        z_proj = a * tf.nn.l2_normalize(z - z_circle, dim=-1) + z_circle
        return z_proj

    # # # # # # # # # #  PLOTTING  FUNCTIONS # # # # # # # # # #

    def save_plot_latent_space(self, x_test, color, batch_size, filename):
        z_mean, _, _ = self.encoder.predict(x_test,
                                            batch_size=batch_size)
        fig = plt.figure(figsize=(5, 5))
        ax = Axes3D(fig)
        ax.scatter(z_mean[:, 0], z_mean[:, 1], z_mean[:, 2], c=color)
        samples = 20
        a = 0.6
        c = 3.0
        theta = 2 * np.pi * np.linspace(0, 1, samples)
        phi = 2 * np.pi * np.linspace(0, 1, samples)
        phi_grid, theta_grid = np.meshgrid(phi, theta)
        x = (c + a * np.cos(theta_grid)) * np.cos(phi_grid)
        y = (c + a * np.cos(theta_grid)) * np.sin(phi_grid)
        z = a * np.sin(theta_grid)
        ax.set_xlim([-c, c])
        ax.set_ylim([-c, c])
        ax.set_zlim([-c, c])
        ax.plot_surface(x, y, z, rstride=1, cstride=1, color='r', alpha=0.1, linewidth=0, shade=True)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_aspect("equal")
        if filename is not None:
            root_dir = os.path.split(filename)[0]
            os.makedirs(root_dir, exist_ok=True)
            plt.savefig(filename, bbox_inches="tight")
        return fig, ax

    def save_plot_image_reconstruction(self, batch_size, filename, samples):
        theta = 2 * np.pi * np.linspace(0, 1, samples)
        phi = 2 * np.pi * np.linspace(0, 1, samples)
        combinations = []
        for i in itertools.product(theta, phi):
            combinations.append(i)
        combinations = np.array(combinations)
        coordinates = np.zeros((len(combinations), 3))
        c = 3
        a = 0.6
        coordinates[:, 0] = (c + a * np.cos(combinations[:, 0])) * np.cos(combinations[:, 1])
        coordinates[:, 1] = (c + a * np.cos(combinations[:, 0])) * np.sin(combinations[:, 1])
        coordinates[:, 2] = np.sin(combinations[:, 0])
        images_decoded = self.decode(coordinates, batch_size)
        # Plot the reconstructed ciphers
        fig = plt.figure(figsize=(5, 5))
        for i in range(samples):
            for j in range(samples):
                ax = fig.add_subplot(samples, samples, j * samples + i + 1)
                if images_decoded.shape[-1] != 3:
                    ax.imshow(images_decoded[i, :, :, 0], cmap="gray")
                else:
                    ax.imshow(images_decoded[i])
                ax.set_xticks([])
                ax.set_yticks([])
        if filename is not None:
            root_dir = os.path.split(filename)[0]
            os.makedirs(root_dir, exist_ok=True)
            plt.savefig(filename, bbox_inches='tight')
        return fig

