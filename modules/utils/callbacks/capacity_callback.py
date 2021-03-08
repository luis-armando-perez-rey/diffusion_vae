from tensorflow.keras.callbacks import Callback
import tensorflow as tf


class CapacityCallback(Callback):
    def __init__(self, min_capacity, max_capacity, total_epochs):
        super(CapacityCallback, self).__init__()

        self.min_capacity = min_capacity
        self.max_capacity = max_capacity
        self.C = min_capacity
        # K.set_value(self.C, self.min_capacity)
        self.total_epochs = total_epochs

    def on_epoch_end(self, epoch, logs=None):
        graph = tf.compat.v1.get_default_graph()
        # new_c = graph.get_tensor_by_name('loss/DecoderDenseModel_loss/vae_loss/c_constant:0')
        # assign_new_c = tf.assign(new_c, self.min_capacity + epoch * (
        #            (self.max_capacity - self.min_capacity) / self.total_epochs))
        self.C = self.min_capacity + epoch * (
                (self.max_capacity - self.min_capacity) / self.total_epochs)
        # with tf.Session() as sess:
        # sess.run(assign_new_c)
        #    print(sess.run(new_c))
        # tf.assign(self.C, self.min_capacity+epoch*((self.max_capacity-self.min_capacity)/self.total_epochs))
        # K.set_value(self.C, self.min_capacity+epoch*((self.max_capacity-self.min_capacity)/self.total_epochs))
        # print(K.get_value(self.C))
        print(self.C)
