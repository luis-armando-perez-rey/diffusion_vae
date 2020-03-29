from modules.utils.data_disentanglement.ground_truth.cars3d import Cars3D
from modules.utils.data_disentanglement.ground_truth.norb import SmallNORB
import numpy as np

def load_cars(path, num_channels):
    cars_class = Cars3D(path)
    x_train = cars_class.images[:183*24, :, :, :num_channels]
    render_number = np.zeros(4392)
    for i in range(24):
        render_number[183*i:(183)*(i+1)] = i
    return x_train, render_number

def load_norb_category(path, category):
    small_norb = SmallNORB(path)
    images = np.expand_dims(small_norb.images, axis = -1)
    features = small_norb.features
    images = images[features[:,0] ==category]
    features = features[features[:,0] == category]
    return images, features