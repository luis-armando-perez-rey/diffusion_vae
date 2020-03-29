import keras.layers


class Decoder:
    def __init__(self, input_shape):
        """
        Constructor
        :param min_log_t: minimum value for the log-scale parameter
        :param max_log_t: maximum value for the log-scale parameter
        :param unconstrained_t: boolean identifying wether scale parameter is restricted
        """
        self.inputs = None
        self.hidden_layer_list = None
        self.hidden_list = None
        self.outputs = None
        self.decoder = None
        self.input_shape = input_shape # input shape of the encoder network


    def _build_input(self, latent_dim):
        self.inputs = keras.layers.Input(shape=(latent_dim,), name='decoder_input')
        return self.inputs

    def _build_hidden(self, inputs):
        hidden = self.hidden_layer_list[0](inputs)
        for layer in self.hidden_layer_list[1:]:
            hidden = layer(hidden)
        return hidden

    def build_decoder_class_tensors(self, latent_dim):
        self.inputs = self._build_input(latent_dim)
        self.hidden_layer_list = self._build_hidden_list()
        self.hidden_list = self._build_hidden(self.inputs)
        self.outputs = self._build_output(self.hidden_list)
        print("Decoder built")



