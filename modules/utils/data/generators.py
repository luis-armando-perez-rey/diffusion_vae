'''
Module with a class to create a generator for low-frequency images.
'''


import numpy as np
import scipy.ndimage
from modules.utils import dataset_creation
import tensorflow as tf
import random


class RotationGenerator(tf.keras.utils.Sequence):
    '''
    A class to create a generator of low-frequency images.
    '''

    def __init__(self, images, batch_size=128, max_rotation = 360, rotations_per_image = 20, labels = False, flat = False):
        '''
        Constructor
        '''
        self.images = images
        self.batch_size = batch_size
        self.indexes = np.arange(len(images))
        self.max_rotation = max_rotation
        self.rotations_per_image = rotations_per_image
        self.labels = labels
        self.num_batches = int(np.floor(self.rotations_per_image * len(self.images) / self.batch_size))
        self.flat = flat
        print("Total number of images per epoch {}".format(len(self.images)*self.rotations_per_image))
        print("Number of batches per epoch {}".format(self.num_batches))

    def __len__(self):
        'Denotes the number of batches per epoch'
        number_images = self.rotations_per_image * len(self.images) / self.batch_size
        return int(np.floor(number_images))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        X, Y = self.__data_generation(indexes)
        if self.labels:
            return X, Y
        else:
            return X, X

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.repeat(np.arange(len(self.images)), self.rotations_per_image)
        np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        if self.flat:
            X = np.empty((self.batch_size, np.product(self.images.shape[1:])), dtype=np.float32)
        else:
            X = np.empty((self.batch_size, *self.images.shape[1:]), dtype=np.float32)
        Y = np.empty(self.batch_size, dtype=int)
        for num_index, index in enumerate(indexes):
            rotation = random.randint(0, self.max_rotation)
            image = scipy.ndimage.rotate(self.images[index], rotation, reshape = False,mode = "constant", cval = 0.0)
            if self.flat:
                X[num_index] = image.flatten()
            else:
                X[num_index,] = image
            Y[num_index] = rotation
        return X, Y



