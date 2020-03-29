from keras.optimizers import Adam
class DiffusionVAEParams(object):
    '''
    classdocs
    '''

    def __init__(self,
                 steps=10,
                 truncation_radius=0.5,
                 var_x=1.0,
                 r_loss="mse",
                 d=2,
                 optimizer = Adam(),
                 controlled_capacity = False,
                 min_capacity = 0.0,
                 max_capacity = 0.0
                ):
        '''
        Constructor
        '''

        # Data parameters

        # Architecture parameters
        self.r_loss = r_loss
        self.optimizer = optimizer

        # Sampling parameters
        self.steps = steps# Calculate the relevant quantities
        self.truncation_radius = truncation_radius

        # Decoder parameters
        self.var_x = var_x

        # Manifold parameters
        self.d = d

        # Controlled capacity (Understanding disentangling in beta-VAE, Burgess et al)
        self.controlled_capacity = controlled_capacity
        self.min_capacity = min_capacity
        self.max_capacity = max_capacity

        # Summary parameters
        self.params_dict = self.params_to_dictionary()


    def params_to_dictionary(self):
        dictionary = {"r_loss": self.r_loss,
                "var_x": self.var_x,
                "steps": self.steps,
                "truncation_radius": self.truncation_radius,
                "d": self.d,
                "controlled_capacity": self.controlled_capacity,
                "min_capacity": self.min_capacity,
                "max_capacity": self.max_capacity}
        return dictionary

    #def params_to_df(self):
    #    df = pd.DataFrame(self.params_dict)
    #    return df
