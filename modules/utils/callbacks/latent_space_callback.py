import tensorflow as tf
import time
import numpy as np
import imageio
import glob
import os

class LatentSpaceCallback(tf.keras.callbacks.Callback):
    def __init__(self, train_data, vae,save_folder, experiment_name, periodicity, batch_size):
        self.train_data = train_data
        self.vae = vae
        self.save_folder = save_folder
        self.experiment_name = experiment_name
        self.batch_size = batch_size
        self.periodicity = periodicity
        self.time_stamp = time.strftime("%Y-%m-%d-%H-%M_")
        self.filename = os.path.join(save_folder, experiment_name + '_' + self.time_stamp)

    def on_train_begin(self, logs={}):
        '''Initialize the learning rate to the maximum value at the start of training.'''
        #matrix_utils.plot_angles_from_matrices(self.train_data, 1, 1)
        #plt.savefig(self.filename + '_angle_epoch_-1' + '.png')
        self.vae.save_plot_latent_space(self.train_data, self.batch_size, self.filename + '_epoch_-1' + '.png')

    def on_epoch_end(self, epoch, logs={}):
        '''Check for end of current cycle, apply restarts when necessary.'''
        if epoch % self.periodicity == 0:
            self.vae.save_plot_latent_space(self.train_data, self.batch_size, self.filename + '_epoch_' + str(epoch) + '.png')


    def on_train_end(self, logs={}):
        # RP3 plot
        path_list = np.array(list(glob.glob(self.filename + '*')))
        print(path_list)
        epochs = np.array([int(x.split('_epoch_')[-1].split('.')[0]) for x in path_list])
        ordered_indexes = np.argsort(epochs)
        images = np.array([imageio.imread(x) for x in path_list[ordered_indexes]])
        imageio.mimsave(self.filename+ '.gif', images, fps=1000)



