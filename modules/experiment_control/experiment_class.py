'''
Created on Dec 7, 2018


'''
import os
import time
import pandas as pd
import numpy as np
import json
import tensorflow as tf
from modules.utils.visualization import plot_utils
from modules.utils.callbacks.capacity_callback import CapacityCallback



class Experiment(object):
    '''
    classdocs
    '''

    # initial path ?
    # manage csv file / database

    def __init__(self, diffusionVAE, experiment_params, train_data, path, target_data=None):
        '''
        Constructor
        '''
        self.diffusionVAE = diffusionVAE
        self.experiment_params = experiment_params
        self.train_data = train_data
        if target_data is None:
            self.target_data = train_data
        else:
            self.target_data = target_data
        self.path = path
        self.csv_record = os.path.join(self.path, "diffusion_vae_experiments.csv")
        self.time_stamp = None

    def run(self, scheduler=None, tensorboard=True):
        '''
        Runs experiment
        '''

        self.time_stamp = time.strftime("%Y-%m-%d-%H-%M_")
        # Weight file
        weights_dir = os.path.join(self.path, "weights_folder")
        os.makedirs(weights_dir, exist_ok=True)
        weights_file = os.path.join(weights_dir, self.time_stamp + ".h5")

        # Tensorboard_file
        tensorboard_dir = os.path.join(self.path, "tensorboard")
        os.makedirs(tensorboard_dir, exist_ok=True)
        tensorboard_file = os.path.join(tensorboard_dir, self.time_stamp)
        print(tensorboard_file)

        self.diffusionVAE.train_vae(self.train_data,
                                    self.target_data,
                                    self.experiment_params.epochs,
                                    self.experiment_params.batch_size,
                                    weights_file=weights_file,
                                    tensorboard_file=tensorboard_file,
                                    scheduler=scheduler,
                                    tensorboard=tensorboard
                                    )
        # If we want to see the learning rate in tensorboard
        # self.diffusionVAE.train_vae(self.train_data,
        #                            self.experiment_params.epochs,
        #                            self.experiment_params.batch_size,
        #                            weights_file,
        #                            tensorboard_file, scheduler, tensorboard_cb_func=lr_tensorboard.LRTensorBoard)

        # Append record of experiments to csv file
        experiment_params_df = self.experiment_params.params_to_df()
        experiment_params_df.insert(0, "timestamp", self.time_stamp)
        diffusion_vae_params_df = self.diffusionVAE.params.params_to_df()
        diffusion_vae_params_df.insert(0, "timestamp", self.time_stamp)
        diffusion_vae_params_df["manifold"] = self.diffusionVAE.manifold
        merged_df = pd.merge(diffusion_vae_params_df, experiment_params_df, on="timestamp")
        if os.path.isfile(self.csv_record):
            with open(self.csv_record, 'a') as f:
                merged_df.to_csv(f, header=False, index=False)
        else:
            with open(self.csv_record, 'a') as f:
                merged_df.to_csv(f, header=True, index=False)

    def run_cvae(self, tensorboard=True):
        '''
        Runs experiment
        '''

        self.time_stamp = time.strftime("%Y-%m-%d-%H-%M_")
        # Weight file
        weights_dir = os.path.join(self.path, "weights_folder")
        os.makedirs(weights_dir, exist_ok=True)
        weights_file = os.path.join(weights_dir, self.time_stamp + ".h5")

        # Tensorboard_file
        tensorboard_dir = os.path.join(self.path, "tensorboard")
        os.makedirs(tensorboard_dir, exist_ok=True)
        tensorboard_file = os.path.join(tensorboard_dir, self.time_stamp)
        print(tensorboard_file)

        # Parameters
        parameters_dir = os.path.join(self.path, "parameters")
        os.makedirs(parameters_dir, exist_ok=True)
        experiment_path = os.path.join(parameters_dir, self.time_stamp)
        os.makedirs(experiment_path, exist_ok=True)

        self.diffusionVAE.train_cvae(self.train_data[0],
                                     self.train_data[1],
                                     self.train_data[0],
                                     self.experiment_params.epochs,
                                     self.experiment_params.batch_size,
                                     weights_file=weights_file,
                                     tensorboard_file=tensorboard_file,
                                     experiment_path=experiment_path,
                                     tensorboard=tensorboard
                                     )

    def run_cvae_fixed(self, parameters, tensorboard=True):
        '''
        Runs experiment with CVAE for angle estimation with a fixed code I made 17 sept 2019
        '''

        self.time_stamp = time.strftime("%Y-%m-%d-%H-%M_")
        # Weight file
        weights_dir = os.path.join(self.path, "weights_folder")
        os.makedirs(weights_dir, exist_ok=True)
        weights_file = os.path.join(weights_dir, self.time_stamp + ".h5")

        # Tensorboard_file
        tensorboard_dir = os.path.join(self.path, "tensorboard")
        os.makedirs(tensorboard_dir, exist_ok=True)
        tensorboard_file = os.path.join(tensorboard_dir, self.time_stamp)
        print(tensorboard_file)


        terminate_nan = tf.keras.callbacks.TerminateOnNaN()
        callbacks_list = [terminate_nan]
        if tensorboard:
            tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_file)
            callbacks_list.append(tensorboard_cb)

        self.diffusionVAE.fit([self.train_data[0], self.train_data[1]],
                              [self.train_data[0], self.train_data[1]],
                              epochs=self.experiment_params.epochs,
                              batch_size=self.experiment_params.batch_size,
                              callbacks=callbacks_list,
                              verbose=2
                              )
        # Saving
        self.diffusionVAE.save_weights(weights_file)

        # Parameters
        parameters_dir = os.path.join(self.path, "parameters")
        os.makedirs(parameters_dir, exist_ok=True)
        experiment_path = os.path.join(parameters_dir, self.time_stamp)
        os.makedirs(experiment_path, exist_ok=True)
        parameters_file = os.path.join(experiment_path, self.time_stamp + '.json')


        with open(parameters_file, 'w') as f:
            json.dump(parameters, f)

    def run_cvae_fixed_early_stop_split(self, parameters, validation_split, tensorboard=True, monitor = "val_loss", patience = 15):
        '''
        Runs experiment with CVAE for angle estimation with a fixed code I made 17 sept 2019
        '''

        self.time_stamp = time.strftime("%Y-%m-%d-%H-%M_")
        # Weight file
        weights_dir = os.path.join(self.path, "weights_folder")
        os.makedirs(weights_dir, exist_ok=True)
        weights_file = os.path.join(weights_dir, self.time_stamp + ".h5")

        # Tensorboard_file
        tensorboard_dir = os.path.join(self.path, "tensorboard")
        os.makedirs(tensorboard_dir, exist_ok=True)
        tensorboard_file = os.path.join(tensorboard_dir, self.time_stamp)
        print(tensorboard_file)


        terminate_nan = tf.keras.callbacks.TerminateOnNaN()
        early_stop_cb = tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=patience, restore_best_weights=True, mode= 'min')
        callbacks_list = [terminate_nan, early_stop_cb]
        if tensorboard:
            tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_file)
            callbacks_list.append(tensorboard_cb)

        self.diffusionVAE.fit([self.train_data[0], self.train_data[1]],
                              [self.train_data[0], self.train_data[1]],
                              epochs=self.experiment_params.epochs,
                              batch_size=self.experiment_params.batch_size,
                              callbacks=callbacks_list,
                              verbose=2,
                              validation_split=validation_split
                              )
        # Saving
        self.diffusionVAE.save_weights(weights_file)

        # Parameters
        parameters_dir = os.path.join(self.path, "parameters")
        os.makedirs(parameters_dir, exist_ok=True)
        experiment_path = os.path.join(parameters_dir, self.time_stamp)
        os.makedirs(experiment_path, exist_ok=True)
        parameters_file = os.path.join(experiment_path, self.time_stamp + '.json')
        with open(parameters_file, 'w') as f:
            json.dump(parameters, f)





    def run_cvae_fixed_semi_early_stop_split(self, parameters, validation_split, tensorboard=True, monitor = "val_loss", patience = 15):
        '''
        Runs experiment with CVAE for angle estimation with a fixed code I made 17 sept 2019
        '''

        self.time_stamp = time.strftime("%Y-%m-%d-%H-%M_")
        # Weight file
        weights_dir = os.path.join(self.path, "weights_folder")
        os.makedirs(weights_dir, exist_ok=True)
        weights_file = os.path.join(weights_dir, self.time_stamp + ".h5")

        # Tensorboard_file
        tensorboard_dir = os.path.join(self.path, "tensorboard")
        os.makedirs(tensorboard_dir, exist_ok=True)
        tensorboard_file = os.path.join(tensorboard_dir, self.time_stamp)
        print(tensorboard_file)


        terminate_nan = tf.keras.callbacks.TerminateOnNaN()
        early_stop_cb = tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=patience, restore_best_weights=True, mode= 'min')
        callbacks_list = [terminate_nan, early_stop_cb]
        if tensorboard:
            tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_file)
            callbacks_list.append(tensorboard_cb)

        self.diffusionVAE.fit([self.train_data[0], self.train_data[1], self.train_data[2]],
                              [self.train_data[0], self.train_data[1]],
                              epochs=self.experiment_params.epochs,
                              batch_size=self.experiment_params.batch_size,
                              callbacks=callbacks_list,
                              verbose=2,
                              validation_split=validation_split
                              )
        # Saving
        self.diffusionVAE.save_weights(weights_file)

        # Parameters
        parameters_dir = os.path.join(self.path, "parameters")
        os.makedirs(parameters_dir, exist_ok=True)
        experiment_path = os.path.join(parameters_dir, self.time_stamp)
        os.makedirs(experiment_path, exist_ok=True)
        parameters_file = os.path.join(experiment_path, self.time_stamp + '.json')

        with open(parameters_file, 'w') as f:
            json.dump(parameters, f)


    def run_cvae_fixed_early_stop_custom_split(self, parameters, validation_data, tensorboard=True, monitor = "val_loss", patience = 15) :
        '''
        Runs experiment with CVAE for angle estimation with a fixed code I made 17 sept 2019
        '''

        self.time_stamp = time.strftime("%Y-%m-%d-%H-%M_")
        # Weight file
        weights_dir = os.path.join(self.path, "weights_folder")
        os.makedirs(weights_dir, exist_ok=True)
        weights_file = os.path.join(weights_dir, self.time_stamp + ".h5")

        # Tensorboard_file
        tensorboard_dir = os.path.join(self.path, "tensorboard")
        os.makedirs(tensorboard_dir, exist_ok=True)
        tensorboard_file = os.path.join(tensorboard_dir, self.time_stamp)
        print(tensorboard_file)


        terminate_nan = tf.keras.callbacks.TerminateOnNaN()
        early_stop_cb = tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=patience, restore_best_weights=True, mode= 'min')
        callbacks_list = [terminate_nan, early_stop_cb]
        if tensorboard:
            tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_file)
            callbacks_list.append(tensorboard_cb)

        self.diffusionVAE.fit([self.train_data[0], self.train_data[1]],
                              [self.train_data[0], self.train_data[1]],
                              epochs=self.experiment_params.epochs,
                              batch_size=self.experiment_params.batch_size,
                              callbacks=callbacks_list,
                              verbose=2,
                              validation_data=validation_data
                              )
        # Saving
        self.diffusionVAE.save_weights(weights_file)

        # Parameters
        parameters_dir = os.path.join(self.path, "parameters")
        os.makedirs(parameters_dir, exist_ok=True)
        experiment_path = os.path.join(parameters_dir, self.time_stamp)
        os.makedirs(experiment_path, exist_ok=True)
        parameters_file = os.path.join(experiment_path, self.time_stamp + '.json')

        with open(parameters_file, 'w') as f:
            json.dump(parameters, f)

    def run_cvae_fixed_semi(self, parameters, tensorboard=True):
        '''
        Runs experiment with CVAE for angle estimation with a fixed code I made 17 sept 2019
        '''

        self.time_stamp = time.strftime("%Y-%m-%d-%H-%M_")
        # Weight file
        weights_dir = os.path.join(self.path, "weights_folder")
        os.makedirs(weights_dir, exist_ok=True)
        weights_file = os.path.join(weights_dir, self.time_stamp + ".h5")

        # Tensorboard_file
        tensorboard_dir = os.path.join(self.path, "tensorboard")
        os.makedirs(tensorboard_dir, exist_ok=True)
        tensorboard_file = os.path.join(tensorboard_dir, self.time_stamp)
        print(tensorboard_file)


        terminate_nan = tf.keras.callbacks.TerminateOnNaN()
        callbacks_list = [terminate_nan]
        if tensorboard:
            tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_file)
            callbacks_list.append(tensorboard_cb)

        self.diffusionVAE.fit([self.train_data[0], self.train_data[1], self.train_data[2]],
                              [self.train_data[0], self.train_data[1]],
                              epochs=self.experiment_params.epochs,
                              batch_size=self.experiment_params.batch_size,
                              callbacks=callbacks_list,
                              verbose=2
                              )
        # Saving
        self.diffusionVAE.save_weights(weights_file)

        # Parameters
        parameters_dir = os.path.join(self.path, "parameters")
        os.makedirs(parameters_dir, exist_ok=True)
        experiment_path = os.path.join(parameters_dir, self.time_stamp)
        os.makedirs(experiment_path, exist_ok=True)
        parameters_file = os.path.join(experiment_path, self.time_stamp + '.json')

        with open(parameters_file, 'w') as f:
            json.dump(parameters, f)

    def run_cvae_fixed_semi_early_stop_custom(self, parameters, validation_data, tensorboard=True, monitor = "val_loss", patience = 15):
        '''
        Runs experiment with CVAE for angle estimation with a fixed code I made 17 sept 2019
        '''

        self.time_stamp = time.strftime("%Y-%m-%d-%H-%M_")
        # Weight file
        weights_dir = os.path.join(self.path, "weights_folder")
        os.makedirs(weights_dir, exist_ok=True)
        weights_file = os.path.join(weights_dir, self.time_stamp + ".h5")

        # Tensorboard_file
        tensorboard_dir = os.path.join(self.path, "tensorboard")
        os.makedirs(tensorboard_dir, exist_ok=True)
        tensorboard_file = os.path.join(tensorboard_dir, self.time_stamp)
        print(tensorboard_file)



        terminate_nan = tf.keras.callbacks.TerminateOnNaN()
        early_stop_cb = tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=patience, restore_best_weights=True, mode= 'min')
        callbacks_list = [terminate_nan, early_stop_cb]
        if tensorboard:
            tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_file)
            callbacks_list.append(tensorboard_cb)

        self.diffusionVAE.fit([self.train_data[0], self.train_data[1], self.train_data[2]],
                              [self.train_data[0], self.train_data[1]],
                              epochs=self.experiment_params.epochs,
                              batch_size=self.experiment_params.batch_size,
                              callbacks=callbacks_list,
                              verbose=2,
                              validation_data=validation_data
                              )


        # Saving
        self.diffusionVAE.save_weights(weights_file)

        # Parameters
        parameters_dir = os.path.join(self.path, "parameters")
        os.makedirs(parameters_dir, exist_ok=True)
        experiment_path = os.path.join(parameters_dir, self.time_stamp)
        os.makedirs(experiment_path, exist_ok=True)
        parameters_file = os.path.join(experiment_path, self.time_stamp + '.json')

        with open(parameters_file, 'w') as f:
            json.dump(parameters, f)

    def run_regression_angles_early_stop_custom(self, parameters, validation_data, tensorboard=True, monitor = "val_loss", patience = 15):
        '''
        Runs experiment with CVAE for angle estimation with a fixed code I made 17 sept 2019
        '''

        self.time_stamp = time.strftime("%Y-%m-%d-%H-%M_")
        # Weight file
        weights_dir = os.path.join(self.path, "weights_folder")
        os.makedirs(weights_dir, exist_ok=True)
        weights_file = os.path.join(weights_dir, self.time_stamp + ".h5")

        # Tensorboard_file
        tensorboard_dir = os.path.join(self.path, "tensorboard")
        os.makedirs(tensorboard_dir, exist_ok=True)
        tensorboard_file = os.path.join(tensorboard_dir, self.time_stamp)
        print(tensorboard_file)



        terminate_nan = tf.keras.callbacks.TerminateOnNaN()
        early_stop_cb = tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=patience, restore_best_weights=True, mode= 'min')
        callbacks_list = [terminate_nan, early_stop_cb]
        if tensorboard:
            tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_file)
            callbacks_list.append(tensorboard_cb)

        self.diffusionVAE.fit([self.train_data[0]],
                              [self.train_data[1]],
                              epochs=self.experiment_params.epochs,
                              batch_size=self.experiment_params.batch_size,
                              callbacks=callbacks_list,
                              verbose=2,
                              validation_data=validation_data
                              )


        # Saving
        self.diffusionVAE.save_weights(weights_file)

        # Parameters
        parameters_dir = os.path.join(self.path, "parameters")
        os.makedirs(parameters_dir, exist_ok=True)
        experiment_path = os.path.join(parameters_dir, self.time_stamp)
        os.makedirs(experiment_path, exist_ok=True)
        parameters_file = os.path.join(experiment_path, self.time_stamp + '.json')

        with open(parameters_file, 'w') as f:
            json.dump(parameters, f)




    def run_generator(self, generator):
        '''
        Runs experiment
        '''
        self.time_stamp = time.strftime("%Y-%m-%d-%H-%M_")
        # Weight file
        weights_dir = os.path.join(self.path, "weights_folder")
        os.makedirs(weights_dir, exist_ok=True)
        weights_file = os.path.join(weights_dir, self.time_stamp + ".h5")

        # Tensorboard_file
        tensorboard_dir = os.path.join(self.path, "tensorboard")
        os.makedirs(tensorboard_dir, exist_ok=True)
        tensorboard_file = os.path.join(tensorboard_dir, self.time_stamp)

        self.diffusionVAE.train_generator_vae(generator,
                                              self.experiment_params.epochs,
                                              weights_file,
                                              tensorboard_file)

        # Append record of experiments to csv file
        experiment_params_df = self.experiment_params.params_to_df()
        experiment_params_df.insert(0, "timestamp", self.time_stamp)
        diffusion_vae_params_df = self.diffusionVAE.params.params_to_df()
        diffusion_vae_params_df.insert(0, "timestamp", self.time_stamp)
        diffusion_vae_params_df["manifold"] = self.diffusionVAE.manifold
        merged_df = pd.merge(diffusion_vae_params_df, experiment_params_df, on="timestamp")
        if os.path.isfile(self.csv_record):
            with open(self.csv_record, 'a') as f:
                merged_df.to_csv(f, header=False, index=False)
        else:
            with open(self.csv_record, 'a') as f:
                merged_df.to_csv(f, header=True, index=False)

    def run_generator_fourier(self, low_freq_generator, steps_per_epoch):
        '''
        Runs experiment
        '''
        generator = low_freq_generator.generate()
        self.time_stamp = time.strftime("%Y-%m-%d-%H-%M_")
        # Weight file
        weights_dir = os.path.join(self.path, "weights_folder")
        os.makedirs(weights_dir, exist_ok=True)
        weights_file = os.path.join(weights_dir, self.time_stamp + ".h5")

        # Tensorboard_file
        tensorboard_dir = os.path.join(self.path, "tensorboard")
        os.makedirs(tensorboard_dir, exist_ok=True)
        tensorboard_file = os.path.join(tensorboard_dir, self.time_stamp)

        # Fourier_components_file
        components_dir = os.path.join(self.path, "fourier_components")
        os.makedirs(components_dir, exist_ok=True)
        components_file = os.path.join(components_dir, self.time_stamp + '.npy')
        np.save(components_file, low_freq_generator._fourier_components)

        self.diffusionVAE.train_generator_vae(generator, steps_per_epoch,
                                              self.experiment_params.epochs,
                                              weights_file,
                                              tensorboard_file)

        # Append record of experiments to csv file
        experiment_params_df = self.experiment_params.params_to_df()
        experiment_params_df.insert(0, "timestamp", self.time_stamp)
        diffusion_vae_params_df = self.diffusionVAE.params.params_to_df()
        diffusion_vae_params_df.insert(0, "timestamp", self.time_stamp)
        diffusion_vae_params_df["manifold"] = self.diffusionVAE.manifold
        merged_df = pd.merge(diffusion_vae_params_df, experiment_params_df, on="timestamp")
        if os.path.isfile(self.csv_record):
            with open(self.csv_record, 'a') as f:
                merged_df.to_csv(f, header=False, index=False)
        else:
            with open(self.csv_record, 'a') as f:
                merged_df.to_csv(f, header=True, index=False)

    def run_checkpoints(self):
        '''
        Runs experiment
        '''

        self.time_stamp = time.strftime("%Y-%m-%d-%H-%M_")
        # Weight file
        weights_dir = os.path.join(self.path, "weights_folder")
        os.makedirs(weights_dir, exist_ok=True)
        weights_file = os.path.join(weights_dir, self.time_stamp + ".h5")

        # Checkpoints folder
        weights_dir_checkpoint = os.path.join(weights_dir, self.time_stamp)
        os.makedirs(weights_dir_checkpoint, exist_ok=True)

        # Tensorboard_file
        tensorboard_dir = os.path.join(self.path, "tensorboard")
        os.makedirs(tensorboard_dir, exist_ok=True)
        tensorboard_file = os.path.join(tensorboard_dir, self.time_stamp)

        self.diffusionVAE.train_vae_checkpoints(self.train_data,
                                                self.experiment_params.epochs,
                                                self.experiment_params.batch_size,
                                                weights_file,
                                                tensorboard_file, weights_dir_checkpoint)

        # Append record of experiments to csv file
        experiment_params_df = self.experiment_params.params_to_df()
        experiment_params_df.insert(0, "timestamp", self.time_stamp)
        diffusion_vae_params_df = self.diffusionVAE.params.params_to_df()
        diffusion_vae_params_df.insert(0, "timestamp", self.time_stamp)
        diffusion_vae_params_df["manifold"] = self.diffusionVAE.manifold
        merged_df = pd.merge(diffusion_vae_params_df, experiment_params_df, on="timestamp")
        if os.path.isfile(self.csv_record):
            with open(self.csv_record, 'a') as f:
                merged_df.to_csv(f, header=False, index=False)
        else:
            with open(self.csv_record, 'a') as f:
                merged_df.to_csv(f, header=True, index=False)


    def run_vae(self, callbacks_list = [], tensorboard = True):
        '''
        Runs experiment with CVAE for angle estimation with a fixed code I made 17 sept 2019
        '''

        self.time_stamp = time.strftime("%Y-%m-%d-%H-%M_")
        # Weight file
        weights_dir = os.path.join(self.path, "weights_folder")
        os.makedirs(weights_dir, exist_ok=True)
        weights_file = os.path.join(weights_dir, self.time_stamp + ".h5")

        # Tensorboard_file
        tensorboard_dir = os.path.join(self.path, "tensorboard")
        os.makedirs(tensorboard_dir, exist_ok=True)
        tensorboard_file = os.path.join(tensorboard_dir, self.time_stamp)
        print(tensorboard_file)


        if tensorboard:
            tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_file)
            callbacks_list.append(tensorboard_cb)
        if self.diffusionVAE.params.controlled_capacity:
            callbacks_list.append(CapacityCallback(self.diffusionVAE.params.min_capacity, self.diffusionVAE.params.max_capacity, self.experiment_params.epochs))


        self.diffusionVAE.vae.fit(self.train_data, self.target_data,
                              epochs=self.experiment_params.epochs,
                              batch_size=self.experiment_params.batch_size,
                              callbacks=callbacks_list,
                              verbose=2
                              )
        ########## SAVING ##########
        self.diffusionVAE.vae.save_weights(weights_file)

        # Saving folder
        parameters_dir = os.path.join(self.path, "parameters")
        os.makedirs(parameters_dir, exist_ok=True)
        experiment_path = os.path.join(parameters_dir, self.time_stamp)
        os.makedirs(experiment_path, exist_ok=True)

        # Get parameters

        encoder_params_dict, decoder_params_dict, vae_params_dict = self.diffusionVAE.return_parameters_dict()

        # Encoder parameters
        parameters_file = os.path.join(experiment_path, "encoder_"+self.time_stamp + '.json')
        with open(parameters_file, 'w') as f:
            json.dump(encoder_params_dict, f)

        # Decoder parameters
        parameters_file = os.path.join(experiment_path, "decoder_" + self.time_stamp + '.json')
        with open(parameters_file, 'w') as f:
            json.dump(decoder_params_dict, f)

        # VAE parameters
        parameters_file = os.path.join(experiment_path, "vae_" + self.time_stamp + '.json')
        with open(parameters_file, 'w') as f:
            json.dump(vae_params_dict, f)

        # Experiment parameters
        parameters_file = os.path.join(experiment_path, "exp_" + self.time_stamp + '.json')
        with open(parameters_file, 'w') as f:
            json.dump(self.experiment_params.params_dict, f)




    def plot_outcomes(self, x_train, y_train, extra_label=""):
        image_dir = os.path.join(self.path, "images")
        os.makedirs(image_dir, exist_ok=True)
        latent_space_image_filename = os.path.join(image_dir, self.time_stamp + "_latent" + extra_label + ".png")
        reconstruction_image_filename = os.path.join(image_dir, self.time_stamp + "reconstruction.png")
        self.diffusionVAE.save_plot_latent_space((x_train, y_train), 128, latent_space_image_filename)
        self.diffusionVAE.save_plot_image_reconstruction(128, reconstruction_image_filename, 20)

    def plot_latent(self, x_train, y_train, extra_label=""):
        image_dir = os.path.join(self.path, "images")
        os.makedirs(image_dir, exist_ok=True)
        latent_space_image_filename = os.path.join(image_dir, self.time_stamp + "_latent" + extra_label + ".png")
        self.diffusionVAE.save_plot_latent_space((x_train, y_train), 128, latent_space_image_filename)

    def plot_outcomes_generator_basic(self, generator, extra_label=""):
        image_dir = os.path.join(self.path, "images")
        os.makedirs(image_dir, exist_ok=True)
        latent_space_image_filename = os.path.join(image_dir, self.time_stamp + "_latent" + extra_label + ".png")
        reconstruction_image_filename = os.path.join(image_dir, self.time_stamp + "reconstruction.png")
        num_batches = len(generator)
        # Initialize data
        x_train, y_train = generator.__getitem__(0)
        for batch in range(num_batches - 1):
            data = generator.__getitem__(batch + 1)
            x_train = np.append(x_train, data[0], axis=0)
            y_train = np.append(y_train, data[1], axis=0)
        self.diffusionVAE.save_plot_latent_space((x_train, y_train), 128, latent_space_image_filename)
        self.diffusionVAE.save_plot_image_reconstruction(128, reconstruction_image_filename, 20)

    def plot_outcomes_generator(self, low_freq_gen, batches):
        image_dir = os.path.join(self.path, "images")
        os.makedirs(image_dir, exist_ok=True)
        latent_space_image_filename0 = os.path.join(image_dir, self.time_stamp + "_latent1.png")
        latent_space_image_filename1 = os.path.join(image_dir, self.time_stamp + "_latent2.png")
        reconstruction_image_filename = os.path.join(image_dir, self.time_stamp + "reconstruction.png")
        generator = low_freq_gen.generate_shifts()
        data = next(generator)
        x_train = data[0]
        y_train = data[1]
        for batch in range(batches):
            data = next(generator)
            x_train = np.append(x_train, data[0], axis=0)
            y_train = np.append(y_train, data[1], axis=0)
        colors0 = plot_utils.labels_to_circular_colors(y_train[:, 0])
        colors1 = plot_utils.labels_to_circular_colors(y_train[:, 1])
        self.diffusionVAE.save_plot_latent_space((x_train, colors0), 128, latent_space_image_filename0)
        self.diffusionVAE.save_plot_latent_space((x_train, colors1), 128, latent_space_image_filename1)
        self.diffusionVAE.save_plot_image_reconstruction(128, reconstruction_image_filename, 20)

    def plot_example_datapoint_generator(self, low_freq_gen):
        image_dir = os.path.join(self.path, "images")
        os.makedirs(image_dir, exist_ok=True)
        image_filename = os.path.join(image_dir, self.time_stamp + "_datapoint.png")
        generator = low_freq_gen.generate_shifts()
        data = next(generator)
        x_train = data[0]
        plot_utils.plot_datapoint(x_train[0].reshape((low_freq_gen.im_size, low_freq_gen.im_size)), image_filename)
