'''
Created on Dec 6, 2018
'''

# System imports
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
from modules.deltavae.deltavae_latent_spaces.deltavae_sphere import volume_sphere

@tf.custom_gradient
def projection_O3(x):
    """
    Implementation of the projection into O(3) manifold
    :param x: matrix of 3x3
    :return:
    """
    S, U, V = tf.linalg.svd(x)
    projection = tf.matmul(U, V, transpose_b=True)

    def grad(dy):
        S_reshaped = tf.reshape(S, [-1, 3, 1])
        S_tiled = tf.tile(S_reshaped, [1, 1, 3])
        S_tiled_t = tf.transpose(a=S_tiled, perm=[0, 2, 1])
        Q = 2 * tf.math.reciprocal(S_tiled + S_tiled_t)
        first_parenthesis = tf.matmul(tf.matmul(U, dy, transpose_a=True), V) - tf.matmul(
            tf.matmul(V, dy, transpose_a=True, transpose_b=True), U)
        second_parenthesis = tf.math.multiply(Q, first_parenthesis)
        gradient = tf.matmul(tf.matmul(U, second_parenthesis), V, transpose_b=True) / 2
        return gradient

    return projection, grad



class DiffusionSO3VAE(DiffusionVAE):
    '''
    Delta-VAE for the SO(3) manifold latent space. Projection is given in terms of the Singular Value Decomposition
    '''

    def __init__(self, params, encoder_class, decoder_class):
        '''
        Constructor
        '''
        params.params_dict["manifold"] = "so3"
        self.latent_dim = 9 # dimension of ambient space
        self.scale_dim = 1 # dimension of time parameter
        self.d = 3 # degrees of freedom in manifold
        self.S = lambda x: self.d * (self.d - 1) / 2 #scalar curvature of manifold
        # The SO(3) is isometric to 'standard' RP3 times sqrt(2)
        self.volume = np.sqrt(2) ** 3 * volume_sphere(3) / 2 # manifold volume
        super(DiffusionSO3VAE, self).__init__(params, encoder_class, decoder_class)

    def kl_tensor(self, logt, mu_z):
        scalar_curv = self.d * (self.d - 1) / 2  # The SO(3) is isometric to sqrt(2) times 'standard' RP^3
        volume = self.volume
        loss = -self.d * logt / 2.0 - self.d * np.log(2.0 * np.pi) / 2.0 - self.d / 2.0 + np.log(volume) \
               + scalar_curv * K.exp(logt) / 4
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
        z_reshaped = tf.reshape(z, [-1, 3, 3])
        z_O3 = projection_O3(z_reshaped)
        z_O3_reshaped = tf.reshape(z_O3, [-1, 9])
        z_proj = tf.reshape(tf.linalg.det(z_O3), [-1, 1]) * z_O3_reshaped
        return z_proj

    def encode_matrix(self, data, batch_size):
        encoded = self.encode_location(data, batch_size)
        encoded = encoded.reshape((-1, 3, 3))
        return encoded

    # # # # # # # # # #  PLOTTING  FUNCTIONS # # # # # # # # # #
    def get_angles(self, matrices):
        angles = np.array(
            [np.arccos((np.clip(np.trace(matrix), a_min=-1.0, a_max=3.0) - 1) / 2.0) for matrix in matrices])
        return angles

    def get_rp3(self, matrices):
        angles = self.get_angles(matrices)
        axes = np.zeros((len(matrices), 3))
        quotient_sine = np.zeros((len(matrices), 1))
        # non_orthogonal = 0
        for num_matrix, matrix in enumerate(matrices):
            # if not(isorthogonal(matrix, tolerance = 1e-10)):
            #    non_orthogonal += 1
            difference = matrix - matrix.transpose()
            if np.isclose(angles[num_matrix], 0.0):
                quotient_sine[num_matrix] = 0.0
                axes[num_matrix, :] = np.zeros(3)
            elif np.abs(angles[num_matrix] - np.pi) <= 0.0133:
                quotient_sine[num_matrix] = np.pi
                B = (matrix + np.eye(3)) / 2.0
                axes[num_matrix, 0] = np.sqrt(B[0, 0])
                # axes[num_matrix, 0] = np.sign(-B[0,2])
                axes[num_matrix, 1] = np.sqrt(B[1, 1])
                axes[num_matrix, 1] *= np.sign(B[0, 1]) * np.sign(axes[num_matrix, 0])
                axes[num_matrix, 2] = np.sqrt(B[2, 2])
                axes[num_matrix, 2] *= np.sign(axes[num_matrix, 1]) * np.sign(B[1, 2])
            else:
                quotient_sine[num_matrix] = angles[num_matrix] / (2 * np.sin(angles[num_matrix]))
                axes[num_matrix, 0] = difference[2, 1]
                axes[num_matrix, 1] = difference[0, 2]
                axes[num_matrix, 2] = difference[1, 0]
        axes = quotient_sine * axes / np.pi
        # print("Number of non orthogonal matrices {}".format(non_orthogonal))
        return axes

    def get_rp3(self, matrices):
        angles = self.get_angles(matrices)
        axes = np.zeros((len(matrices), 3))
        quotient_sine = np.zeros((len(matrices), 1))
        # non_orthogonal = 0
        for num_matrix, matrix in enumerate(matrices):
            # if not(isorthogonal(matrix, tolerance = 1e-10)):
            #    non_orthogonal += 1
            difference = matrix - matrix.transpose()
            if np.isclose(angles[num_matrix], 0.0):
                quotient_sine[num_matrix] = 0.0
                axes[num_matrix, :] = np.zeros(3)
            elif np.abs(angles[num_matrix] - np.pi) <= 0.0133:
                quotient_sine[num_matrix] = np.pi
                B = (matrix + np.eye(3)) / 2.0
                axes[num_matrix, 0] = np.sqrt(B[0, 0])
                # axes[num_matrix, 0] = np.sign(-B[0,2])
                axes[num_matrix, 1] = np.sqrt(B[1, 1])
                axes[num_matrix, 1] *= np.sign(B[0, 1]) * np.sign(axes[num_matrix, 0])
                axes[num_matrix, 2] = np.sqrt(B[2, 2])
                axes[num_matrix, 2] *= np.sign(axes[num_matrix, 1]) * np.sign(B[1, 2])
            else:
                quotient_sine[num_matrix] = angles[num_matrix] / (2 * np.sin(angles[num_matrix]))
                axes[num_matrix, 0] = difference[2, 1]
                axes[num_matrix, 1] = difference[0, 2]
                axes[num_matrix, 2] = difference[1, 0]
        axes = quotient_sine * axes / np.pi
        # print("Number of non orthogonal matrices {}".format(non_orthogonal))
        return axes

    def save_plot_latent_space(self, x_test, color, batch_size, filename):
        z_mean = self.encode_matrix(x_test, batch_size=batch_size)
        axes = self.get_rp3(z_mean)
        fig = plt.figure(figsize=(5, 5))
        ax = Axes3D(fig)
        ax.scatter(axes[:, 0], axes[:, 1], axes[:, 2], c=color)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        if filename is not None:
            root_dir = os.path.split(filename)[0]
            os.makedirs(root_dir, exist_ok=True)
            plt.savefig(filename, bbox_inches="tight")
        return fig, ax


    def save_plot_image_reconstruction(self, batch_size, filename, samples):
        # angles_2pi = np.linspace(-1, -0.5, samples) * np.pi
        # angles_pi = np.linspace(-1, -0.5, samples) * np.pi / 2.0
        # mesh = np.meshgrid(angles_2pi, angles_pi, angles_2pi)
        # flat_mesh = np.array([mesh[0].flatten(), mesh[1].flatten(), mesh[2].flatten()]).transpose()
        # regular_matrices = np.zeros((len(flat_mesh), 3, 3))
        # for num_angle, angle in enumerate(flat_mesh):
        #     regular_matrices[num_angle, :, :] = self.eulerAnglesToRotationMatrix(angle)
        # decoded_images = self.decode(regular_matrices.reshape([len(regular_matrices), 9]), batch_size=batch_size)
        # decoded_images_reshaped = decoded_images.reshape((-1, int(np.sqrt(self.image_size)), int(np.sqrt(self.image_size))))
        # fig = plt.figure(figsize=(10, 24))
        # gs0 = gridspec.GridSpec(samples // 2, 2)
        # for gs in gs0:
        #     gs_sub = gridspec.GridSpecFromSubplotSpec(samples, samples, subplot_spec=gs)
        #     for i in range(samples):
        #         for j in range(samples):
        #             ax00 = fig.add_subplot(gs_sub[i, j])
        #             ax00.imshow(decoded_images_reshaped[i + j * samples * samples])
        #             ax00.set_xticks([])
        #             ax00.set_yticks([])
        # plt.savefig(filename)
        print("Not implemented")

    # # Checks if a matrix is a valid rotation matrix.
    def isRotationMatrix(self, R):
        """
        Asserts whether matrix R is a rotation matrix
        :param R: rotation matrix
        :return: returns a boolean; True if input is a rotation matrix
        """
        Rt = np.transpose(R)
        shouldBeIdentity = np.dot(Rt, R)
        I = np.identity(3, dtype=R.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < 1e-5

    def rotationMatrixToEulerAngles(self, R):
        """
        Transoforms rotation matrix to Euler angles
        :param R: rotation matrix
        :return: [x, y, z] rotation angles
        """

        # assert (self.isRotationMatrix(R)), "Not a rotation matrix"

        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

        singular = sy < 1e-5

        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0

        return np.array([x, y, z])

    # Calculates Rotation Matrix given euler angles.
    def eulerAnglesToRotationMatrix(self, theta):

        R_x = np.array([[1, 0, 0],
                        [0, np.cos(theta[0]), -np.sin(theta[0])],
                        [0, np.sin(theta[0]), np.cos(theta[0])]
                        ])

        R_y = np.array([[np.cos(theta[1]), 0, np.sin(theta[1])],
                        [0, 1, 0],
                        [-np.sin(theta[1]), 0, np.cos(theta[1])]
                        ])

        R_z = np.array([[np.cos(theta[2]), -np.sin(theta[2]), 0],
                        [np.sin(theta[2]), np.cos(theta[2]), 0],
                        [0, 0, 1]
                        ])

        R = np.dot(R_z, np.dot(R_y, R_x))

        return R
