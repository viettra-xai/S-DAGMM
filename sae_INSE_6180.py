
import tensorflow.keras as keras
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler




class SAE:
    """ Stack Autoencoder.
    """

    MODEL_FILENAME = "SAE_model"
    SCALER_FILENAME = "SAE_scaler"


    def __init__(self, sae_hiddens, out_directory, dropout_rate=0.1, n_epochs=100, normalize=True, random_seed=42):
        
        self.sae_hiddens = sae_hiddens
        self.out_directory = out_directory
        self.rate = dropout_rate
        self.n_epochs = n_epochs
        self.normalize = normalize
        self.scaler = None
        self.seed = random_seed

    def build_model(self, inputs):
        input_shape = inputs.shape[-1]
        self.enc_model = keras.models.Sequential()
        self.enc_model.add(keras.layers.InputLayer(input_shape=input_shape))
        self.enc_model.add(keras.layers.Dropout(rate=self.rate))
        for size in self.sae_hiddens[:-1]:
            self.enc_model.add(keras.layers.Dense(size, activation="elu", kernel_initializer="he_normal",
                                         kernel_regularizer=keras.regularizers.l2(0.01)))
            self.enc_model.add(keras.layers.Dropout(rate=self.rate))
        self.enc_model.add(keras.layers.Dense(self.sae_hiddens[-1], activation="elu", kernel_initializer="he_normal",
                                         kernel_regularizer=keras.regularizers.l2(0.01)))
        
        self.dec_model = keras.models.Sequential()
        self.dec_model.add(keras.layers.InputLayer(input_shape=self.sae_hiddens[-1]))
        self.dec_model.add(keras.layers.Dropout(rate=self.rate))
        for size in self.sae_hiddens[:-1][::-1]:
            self.dec_model.add(keras.layers.Dense(size, activation="elu", kernel_initializer="he_normal",
                                         kernel_regularizer=keras.regularizers.l2(0.01)))
            self.dec_model.add(keras.layers.Dropout(rate=self.rate))

        self.dec_model.add(keras.layers.Dense(input_shape, activation="linear", kernel_initializer="he_normal"))
        
        self.model = keras.models.Sequential([self.enc_model, self.dec_model])

        return self.model
    
#     def build_model(self, inputs):
#         input_shape = inputs.shape[-1]
#         self.model = keras.models.Sequential()
#         self.model.add(keras.layers.InputLayer(input_shape=input_shape))
#         self.model.add(keras.layers.Dropout(rate=self.rate))
#         for size in self.sae_hiddens:
#             self.model.add(keras.layers.Dense(size, activation="elu", kernel_initializer="he_normal",
#                                          kernel_regularizer=keras.regularizers.l2(0.01)))
#             self.model.add(keras.layers.Dropout(rate=self.rate))
#         for size in self.sae_hiddens[:-1][::-1]:
#             self.model.add(keras.layers.Dense(size, activation="elu", kernel_initializer="he_normal",
#                                          kernel_regularizer=keras.regularizers.l2(0.01)))
#             self.model.add(keras.layers.Dropout(rate=self.rate))

#         self.model.add(keras.layers.Dense(input_shape, activation="linear", kernel_initializer="glorot_uniform"))

#         return self.model
    
    def fit(self, inputs):
        X = inputs
        if self.normalize:
            self.scaler = StandardScaler().fit(X)
            X = self.scaler.transform(X)
        X_train, X_valid = train_test_split(X, test_size=0.1, random_state=42)

        self.model.compile(loss="mse",
                           optimizer="nadam")
        checkpoint_cb = keras.callbacks.ModelCheckpoint(self.out_directory + "model_SAE.h5",
                                                        save_best_only=True)
        early_stopping_cb = keras.callbacks.EarlyStopping(patience=5,
                                                          restore_best_weights=True)

        history = self.model.fit(X_train, X_train, epochs=self.n_epochs,
                              validation_data=[X_valid, X_valid],
                              callbacks=[checkpoint_cb, early_stopping_cb])

        self.model = keras.models.load_model(self.out_directory + "model_SAE.h5")

    def restore(self):
        model = self.model
        return model
