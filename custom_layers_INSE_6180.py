import tensorflow as tf
import numpy as np
import tensorflow.keras as keras

class Encoder_Block(keras.layers.Layer):
    # encoder layers of compression network
    def __init__(self, hidden_layer_sizes, activation=keras.activations.tanh, **kwargs):
        super().__init__(**kwargs)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.hidden= [keras.layers.Dense(size, activation=self.activation, kernel_initializer="he_normal",
                                         kernel_regularizer=keras.regularizers.l2(0.01)) for size in self.hidden_layer_sizes[:-1]]
        self.out = keras.layers.Dense(self.hidden_layer_sizes[-1], activation=self.activation,
                                      kernel_initializer="he_normal", kernel_regularizer=keras.regularizers.l2(0.01))

    def call(self, inputs):
        # self.input_size = input_size = inputs.shape[1]
        z = inputs
        for layer in self.hidden:
            z = layer(z)
        z = self.out(z)
        return z

    def get_config(self):  # not shown
        base_config = super().get_config()  # not shown
        return {**base_config}  # not shown

class Decoder_Block(keras.layers.Layer):
    # decoder layers of compresion network
    def __init__(self, hidden_layer_sizes, activation=keras.activations.tanh, input_size = None, **kwargs):
        super().__init__(**kwargs)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.input_size = input_size
        self.hidden = [keras.layers.Dense(size, activation=self.activation, kernel_initializer="he_normal",
                                          kernel_regularizer=keras.regularizers.l2(0.01)) for size in self.hidden_layer_sizes[:-1][::-1]]
        self.out = keras.layers.Dense(self.input_size, activation=keras.activations.linear,
                                      kernel_initializer="he_normal", kernel_regularizer=keras.regularizers.l2(0.01))

    def call(self, inputs):
        z = inputs
        for layer in self.hidden:
            z = layer(z)
        x_res = self.out(z)
        return x_res

    def get_config(self):  # not shown
        base_config = super().get_config()  # not shown
        return {**base_config}  # not shown


class Feature_Extraction_Block(keras.layers.Layer):
    # Defining the Custom Error Construction Layer between the Input Layer and Output Layer of the Autoencoder
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        def euclid_norm(x):
            return tf.sqrt(tf.reduce_sum(tf.square(x), axis=1))
        x_ori, x_res, z_c = inputs
        # Calculate Euclid norm, distance
        norm_x = euclid_norm(x_ori)
        norm_x_dash = euclid_norm(x_res)
        dist_x = euclid_norm(x_ori - x_res)
        dot_x = tf.reduce_sum(x_ori - x_res, axis=1)

        #  1. loss_E : relative Euclidean distance
        #  2. loss_C : cosine similarity
        min_val = 1e-3
        loss_E = dist_x / (norm_x + min_val)
        loss_C = 0.5 * (1.0 - dot_x / (norm_x * norm_x_dash + min_val))
        return tf.concat([loss_E[:,np.newaxis], loss_C[:,np.newaxis], z_c], axis=1)

    def get_config(self):  # not shown
        base_config = super().get_config()  # not shown
        return {**base_config}  # not shown

class Estimation_Block(keras.layers.Layer):
    # estimation network
    def __init__(self, hidden_layer_sizes, activation=keras.activations.tanh, dropout_rate = None, **kwargs):
        super().__init__(**kwargs)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.dropout_layer = keras.layers.Dropout(rate=dropout_rate)
        self.hidden = [keras.layers.Dense(size, activation=self.activation, kernel_initializer="he_normal",
                                          kernel_regularizer=keras.regularizers.l2(0.01))
                               for size in self.hidden_layer_sizes[:-1]]
        self.out = keras.layers.Dense(self.hidden_layer_sizes[-1], activation=keras.activations.softmax,
                                      kernel_initializer="he_normal", kernel_regularizer=keras.regularizers.l2(0.01))

    def call(self, inputs):
        z = inputs
        for layer in self.hidden:
            z = layer(z)
            if self.dropout_rate is not None:
                z = self.dropout_layer(z)
        output = self.out(z)
        return output

    def get_config(self):  # not shown
        base_config = super().get_config()  # not shown
        return {**base_config}  # not shown