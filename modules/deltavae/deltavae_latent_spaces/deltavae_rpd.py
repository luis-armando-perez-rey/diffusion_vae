'''
Created on Dec 6, 2018
'''

# System imports
import os

# Standard imports
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from scipy import stats

# Plotting libraries
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Project library imports
from modules.deltavae.deltavae_latent_spaces.deltavae_parent import DiffusionVAE
from modules.deltavae.deltavae_latent_spaces.deltavae_sphere import volume_sphere

class DiffusionRPNVAE(DiffusionVAE):
    '''
    classdocs
    '''

    def __init__(self, params, encoder_class, decoder_class):
        '''
        Constructor
        '''
        params.params_dict["manifold"] = "rpd"
        self.latent_dim = params.d + 1 # dimension of ambient space
        self.scale_dim = 1 # dimension of time parameter
        self.d = params.d # degrees of freedom in manifold
        self.S = lambda x: self.d * (self.d - 1) # scalar curvature
        self.volume = volume_sphere(self.d) / 2 # manifold volume

        # Distribution and densities
        self.decoding_distribution = stats.multivariate_normal
        self.log_prior = np.log(1 / self.volume)
        super(DiffusionRPNVAE, self).__init__(params,encoder_class, decoder_class, symmetric=True)

    def kl_tensor(self, logt, mu_z):
        scalar_curv = self.d * (self.d - 1)
        volume = self.volume
        #loss = -d * logt / 2.0 - d * np.log(2.0 * np.pi) / 2.0 - d / 2.0 + np.log(volume) \
        #       + (d + 4) * scalar_curv * K.exp(logt) / 24
        # Rebuttal revision
        loss = -self.d * logt / 2.0 - self.d * np.log(2.0 * np.pi) / 2.0 - self.d / 2.0 + np.log(volume) \
               + scalar_curv * K.exp(logt) / 4
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
        z_proj = tf.nn.l2_normalize(z, axis=-1)
        return z_proj

    def save_plot_latent_space(self, x_test, color, batch_size, filename):
        z_mean, _, _ = self.encoder.predict(x_test,batch_size=batch_size)
        if self.d == 3:
            fig = plt.figure(figsize=(12, 10))
            ax = Axes3D(fig)
            # Stereographic projection
            stereo_proj = stereographic_projection(z_mean)
            ax.scatter(stereo_proj[:, 0], stereo_proj[:, 1], stereo_proj[:, 2], c=color)
            samples = 20

            theta = 2 * np.pi * np.linspace(0, 1, samples)
            phi = np.pi * np.linspace(0, 1, samples)
            phi_grid, theta_grid = np.meshgrid(phi, theta)
            x = np.sin(phi_grid) * np.cos(theta_grid)
            y = np.sin(phi_grid) * np.sin(theta_grid)
            z = np.cos(phi_grid)

            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_zlim([-1, 1])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.plot_surface(x, y, z, rstride=1, cstride=1, color='r', alpha=0.1, linewidth=0, shade=True)
            ax.set_aspect("equal")

        if self.d == 2:
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111)
            stereo_proj = stereographic_projection(z_mean)
            ax.scatter(stereo_proj[:, 0], stereo_proj[:, 1], c=color, s=5)
            ax.set_ylim([-1.02, 1.02])
            ax.set_xlim([-1.02, 1.02])
            ax.spines['top'].set_color('none')
            ax.spines['bottom'].set_color('none')
            ax.spines['left'].set_color('none')
            ax.spines['right'].set_color('none')
            circle = plt.Circle((0, 0), 1.01, color='k', fill=False)
            ax.add_artist(circle)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")
        if filename is not None:
            root_dir = os.path.split(filename)[0]
            os.makedirs(root_dir, exist_ok=True)
            plt.savefig(filename, bbox_inches="tight")
        return fig, ax



    def save_plot_image_reconstruction(self, batch_size, filename, samples):
        print("Not implemented")



def stereographic_projection(z_values):
    z_upper = np.reshape(-np.sign(z_values[:, -1]), (-1, 1)) * z_values
    z_0 = z_upper[:, -1]
    stereo_projection = np.copy(z_upper[:, 0:-1]) / (1 - z_0[:, np.newaxis])
    return stereo_projection


if __name__ == "__main__":
    batch_size = 5
    z = np.random.normal(size=(batch_size, 4))
    z = z / np.reshape(np.linalg.norm(z, axis=-1), (-1, 1))
    print(stereographic_projection(z))
