'''
Created on Dec 6, 2018


'''
# System imports
import os

# Standard imports
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import itertools
from scipy import stats

# Plotting libraries
import matplotlib.pyplot as plt

# Project library imports
from modules.deltavae.deltavae_latent_spaces.deltavae_parent import DiffusionVAE

class DiffusionFlatTorusVAE(DiffusionVAE):
    """
    Delta-VAE for Clifford Torus (flat torus) manifold with 2 degrees of freedom.
    """


    def __init__(self, params, encoder_class, decoder_class):
        '''
        Constructor
        '''
        params.params_dict["manifold"] = "clifford_torus"
        self.latent_dim = 4
        self.scale_dim = 1
        self.d = 2
        self.S = lambda x: 0.0
        self.volume = 4 * np.pi ** 2
        self.log_prior = np.log(1 / self.volume)
        self.decoding_distribution = stats.multivariate_normal
        super(DiffusionFlatTorusVAE, self).__init__(params, encoder_class, decoder_class)

    def kl_tensor(self, logt, mu_z):
        loss = -0.5 * logt * self.d - 0.5 * self.d + np.log(4 * np.pi ** 2)
        if self.params.controlled_capacity:
            self.C = tf.Variable(1.0, name = "c_constant")
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
        proj_matrix1 = K.constant([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        proj_matrix2 = K.constant([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        z_proj1 = tf.nn.l2_normalize(K.dot(z, proj_matrix1), axis=-1)
        z_proj2 = tf.nn.l2_normalize(K.dot(z, proj_matrix2), axis=-1)
        z_proj = z_proj1 + z_proj2
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
        z_mean, _, _ = self.encoder.predict(x_test,
                                            batch_size=batch_size)
        fig = plt.figure(figsize=(5, 5))
        ax = plt.gca()
        angle1 = np.arctan2(z_mean[:, 1], z_mean[:, 0])
        angle2 = np.arctan2(z_mean[:, 3], z_mean[:, 2])
        ax.scatter(angle1, angle2, c=color)
        ax.set_aspect("equal")
        ax.set_xlim([-np.pi, np.pi])
        ax.set_xticks(np.linspace(-np.pi, np.pi, 5, endpoint=True))
        ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
        ax.set_ylim([-np.pi, np.pi])
        ax.set_yticks(np.linspace(-np.pi, np.pi, 5, endpoint=True))
        ax.set_yticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
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
        angles = 2 * np.pi * np.random.uniform(0.0, 1.0, (num_samples ** 2, 2))
        projected = np.zeros((len(angles), 4))
        projected[:, 0] = np.cos(angles[:, 0])
        projected[:, 1] = np.sin(angles[:, 0])
        projected[:, 2] = np.cos(angles[:, 1])
        projected[:, 3] = np.sin(angles[:, 1])
        decoded = self.decode(projected, batch_size=batch_size)
        fig = plt.figure(figsize=(5, 5))
        for i in range(len(decoded)):
            plt.subplot(num_samples, num_samples, i + 1)
            plt.imshow(decoded[i], cmap="gray")
            plt.xticks([])
            plt.yticks([])
        if filename is not None:
            root_dir = os.path.split(filename)[0]
            os.makedirs(root_dir, exist_ok=True)
            plt.savefig(filename, bbox_inches="tight")

    def save_plot_image_reconstruction(self, batch_size, filename, samples):
        """
       Plot and save the images obtained by decoding a regular grid over the latent variable manifold
       :param batch_size: batch size for the decoding
       :param filename: filename where the images are saved
       :param samples: square root of the number of points in the regular grid used
       :return:
       """
        # Fix this
        theta = np.pi * np.linspace(-1, 1, samples)
        phi = np.pi * np.linspace(-1, 1, samples)
        combinations = []
        for i in itertools.product(theta, phi):
            combinations.append(i)
        combinations = np.array(combinations)
        coordinates = np.zeros((len(combinations), 4))

        coordinates[:, 0] = np.cos(combinations[:, 0])
        coordinates[:, 1] = np.sin(combinations[:, 0])
        coordinates[:, 2] = np.cos(combinations[:, 1])
        coordinates[:, 3] = np.sin(combinations[:, 1])
        decoded = self.decode(coordinates, batch_size)
        # Plot the reconstructed ciphers
        fig = plt.figure(figsize=(5, 5))

        ax0 = fig.add_subplot(111)
        ax0.spines['top'].set_color('none')
        ax0.spines['bottom'].set_color('none')
        ax0.spines['left'].set_color('none')
        ax0.spines['right'].set_color('none')
        ax0.set_xticks([])
        ax0.set_yticks([])

        for i in range(samples):
            for j in range(samples):
                ax = fig.add_subplot(samples, samples, j * samples + i + 1)
                if decoded.shape[-1]==1:
                    ax.imshow(decoded[i * samples + j,:,:,0], cmap="gray")
                else:
                    ax.imshow(decoded[i * samples + j], cmap="gray")
                ax.set_xticks([])
                ax.set_yticks([])

        if filename is not None:
            root_dir = os.path.split(filename)[0]
            os.makedirs(root_dir, exist_ok=True)
            plt.savefig(filename, bbox_inches='tight')
        return fig, ax



    def interactive_reconstruction(self, data, batch_size):
        fig = plt.figure(figsize = (5,5))
        ax = fig.add_subplot(1,1,1)
        ax = self.plot_latent_space_ax(data, batch_size, ax)

        ax.set_xlim([-np.pi, np.pi])
        ax.set_xticks(np.linspace(-np.pi, np.pi, 5, endpoint=True))
        ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
        ax.set_ylim([-np.pi, np.pi])
        ax.set_yticks(np.linspace(-np.pi, np.pi, 5, endpoint=True))
        ax.set_yticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])

        latent_variable = np.zeros((1, 4))
        sub_ax = plt.axes([0.65, 0.65, 0.2, 0.2])

        sub_ax.set_visible(False)

        def click(event):
            # Get positions of the event of clicking
            latent_variable[0, 0] = np.cos(event.xdata)
            latent_variable[0, 1] = np.sin(event.xdata)
            latent_variable[0, 2] = np.cos(event.ydata)
            latent_variable[0, 3] = np.sin(event.ydata)
            print(latent_variable)
            decoded = self.decode(latent_variable, batch_size)
            # Make important assertions for plotting
            assert decoded.ndim == 4, "The output array does not correspond to the format (sample, height, width, channel)"
            if decoded.shape[-1] == 1:
                decoded = decoded[:, :, :, 0]

            # Show the data reconstruction from input space
            sub_ax.clear()
            figure_coord = fig.transFigure.inverted().transform((event.x, event.y))
            sub_ax.set_position([figure_coord[0], figure_coord[1], 0.2, 0.2])
            sub_ax.imshow(decoded[0], cmap="gray")
            sub_ax.set_xticks([])
            sub_ax.set_yticks([])
            sub_ax.set_visible(True)
            fig.canvas.draw_idle()

        # Add the callback for clicking
        fig.canvas.mpl_connect('button_press_event', click)
        plt.show()
