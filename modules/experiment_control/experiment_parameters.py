import pandas as pd


class ExperimentParams(object):
    """
    Class to define the training parameters used for a training experiment
    """

    def __init__(self, epochs, batch_size):
        """
        Constructor for experiment parameters class
        :param epochs: number of epochs to be run the training
        :param batch_size: number of samples in each batch
        """
        self.epochs = epochs
        self.batch_size = batch_size
        self.params_dict = self.params_to_dict()

    def params_to_dict(self):
        dictionary = {"epochs": [self.epochs],
                "batch_size": [self.batch_size]}
        return dictionary

    def params_to_df(self):
        df = pd.DataFrame(self.params_dict)
        return df
