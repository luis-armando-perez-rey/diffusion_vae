'''
Created on Dec 6, 2018
'''

# System imports
import os

# Standard imports
import numpy as np
import tensorflow as tf
import keras.backend as K
from scipy import stats

# Plotting libraries
import matplotlib.pyplot as plt

# Project library imports
from modules.deltavae.deltavae_latent_spaces.deltavae_parent import DiffusionVAE
from modules.deltavae.deltavae_latent_spaces.deltavae_sphere import volume_sphere


class DiffusionO3VAE(DiffusionVAE):
    '''
    classdocs
    '''

    def __init__(self, params, encoder_class, decoder_class):
        '''
        Constructor
        '''
        params.params_dict["manifold"] = "o3"
        self.latent_dim = 9 # dimension of ambient space
        self.scale_dim = 1 # dimension of time parameter
        # The volume of O(3) is twice the volume of SO(3)
        self.volume = np.sqrt(2) ** 3 * volume_sphere(3) # manifold volume
        self.S = lambda x:self.params.d *(self.params.d-1) # scalar curvature

        # Distributions and densities
        self.decoding_distribution = stats.multivariate_normal
        self.log_prior = np.log(1 / self.volume)
        super(DiffusionO3VAE, self).__init__( params, encoder_class, decoder_class)

    def kl_tensor(self, logt, y):
        d = 3
        scalar_curv = d * (d - 1) / 2
        volume = self.volume
        loss = -d * logt / 2.0 - d * np.log(2.0 * np.pi) / 2.0 - d / 2.0 + np.log(volume) \
               + scalar_curv * K.exp(logt) / 4
        if self.params.controlled_capacity:
            self.C = tf.Variable(1.0)
            loss = tf.abs(loss-self.C)
        return loss

    def sampling(self, args):
        """Reparameterization trick by sampling from an isotropic unit Gaussian.

        # Arguments:
            args (tensor): mean and log of variance of Q(z|X)

        # Returns:
            z (tensor): sampled latent vector
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
        z_reshaped = tf.reshape(z, [-1, 3, 3])
        s, u, v = tf.linalg.svd(z_reshaped, full_matrices=True)
        z_proj = tf.reshape(tf.matmul(u, v, transpose_b=True), [-1, 9])
        return z_proj

    def encode_matrix(self, data, batch_size):
        encoded = self.encode_location(data, batch_size)
        encoded = encoded.reshape((-1, 3, 3))
        return encoded

    # # # # # # # # # #  PLOTTING  FUNCTIONS # # # # # # # # # #

    def save_plot_latent_space(self, x_test, color, batch_size, filename):
        z_mean = self.encode_matrix(x_test, batch_size=batch_size)
        angles_positive = []
        positive_y = []
        angles_negative = []
        negative_y = []
        for num_z, z in enumerate(z_mean):
            if np.linalg.det(z) >= 0:
                angles_positive.append(self.rotationMatrixToEulerAngles(z))
                positive_y.append(color[num_z])
            else:
                angles_negative.append(self.rotationMatrixToEulerAngles(-z))
                negative_y.append(color[num_z])

        angles_positive = np.array(angles_positive)
        angles_negative = np.array(angles_negative)
        positive_y = np.array(positive_y)
        negative_y = np.array(negative_y)

        fig = plt.figure(figsize=(24, 10))
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.set_title("Positive")
        ax.scatter(angles_positive[:, 0], angles_positive[:, 1], angles_positive[:, 2], c=positive_y)
        ax = fig.add_subplot(1, 2, 2, projection='3d')
        ax.scatter(angles_negative[:, 0], angles_negative[:, 1], angles_negative[:, 2], c=negative_y)
        ax.set_title("Negative")
        if filename is not None:
            root_dir = os.path.split(filename)[0]
            os.makedirs(root_dir, exist_ok=True)
            plt.savefig(filename, bbox_inches="tight")
        return fig, ax

    def save_plot_image_reconstruction(self, batch_size, filename, samples):
        print("Not implemented")
        return None

    # # Checks if a matrix is a valid rotation matrix.
    def isRotationMatrix(self, R):
        Rt = np.transpose(R)
        shouldBeIdentity = np.dot(Rt, R)
        I = np.identity(3, dtype=R.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < 1e-5

    # Calculates rotation matrix to euler angles
    # The result is the same as MATLAB except the order
    # of the euler angles ( x and z are swapped ).
    def rotationMatrixToEulerAngles(self, R):

        assert (self.isRotationMatrix(R)), "Not a rotation matrix"

        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0

        return np.array([x, y, z])
