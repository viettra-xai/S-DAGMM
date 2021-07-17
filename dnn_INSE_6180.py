
import tensorflow.keras as keras
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler




class DNN:
    """ Deep Neural Network.
    """

    MODEL_FILENAME = "DNN_model"
    SCALER_FILENAME = "DNN_scaler"

    def __init__(self, dnn_hiddens, output_size, out_directory, pretrained_model, rate=0.1,
                 n_epochs=100, pretrain_sae=True, pretrain_dagmm=True, monte_carlo=True, normalize=True, random_seed=42):

        self.dnn_hiddens = dnn_hiddens
        self.output_size = output_size
        self.out_directory = out_directory
        self.pretrained_model = pretrained_model
        self.rate = rate
        self.n_epochs = n_epochs
        self.pretrain_sae = pretrain_sae
        self.pretrain_dagmm = pretrain_dagmm
        self.monte_carlo = monte_carlo
        self.normalize = normalize
        self.scaler = None
        self.seed = random_seed

    def build_model(self, inputs):
        input_shape = inputs.shape[-1]
        self.model = keras.models.Sequential()
        self.model.add(keras.layers.InputLayer(input_shape=input_shape))
        self.model.add(keras.layers.Dropout(rate=self.rate))
        for size in self.dnn_hiddens:
            self.model.add(keras.layers.Dense(size, activation="elu", kernel_initializer="he_normal",
                                         kernel_regularizer=keras.regularizers.l2(1e-4)))
            self.model.add(keras.layers.Dropout(rate=self.rate))
        
        if self.pretrain_sae is True:
            self.model.set_weights(self.pretrained_model.layers[0].get_weights())

        elif self.pretrain_dagmm is True:
            self.model.set_weights(self.pretrained_model.layers[1].get_weights())

        self.model.add(keras.layers.Dense(self.output_size, activation="softmax", kernel_initializer="glorot_uniform"))

        return self.model
    def fit(self, inputs):
        X, y = inputs
        if self.normalize:
            self.scaler = StandardScaler().fit(X)
            X = self.scaler.transform(X)
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
        if self.pretrain_sae is True or self.pretrain_dagmm is True:
            for layer in self.model.layers[:-1]:
                layer.trainable = False
            self.model.compile(loss="sparse_categorical_crossentropy",
                               optimizer="nadam",
                               metrics=["accuracy"])
            history = self.model.fit(X_train, y_train, epochs=10,
                                  validation_data=(X_valid, y_valid))
            for layer in self.model.layers[:-1]:
                layer.trainable = True
        self.model.compile(loss="sparse_categorical_crossentropy",
                           optimizer="nadam",
                           metrics=["accuracy"])
        checkpoint_cb = keras.callbacks.ModelCheckpoint(self.out_directory + "model_DNN.h5",
                                                        save_best_only=True)
        early_stopping_cb = keras.callbacks.EarlyStopping(patience=5,
                                                          restore_best_weights=True)

        history = self.model.fit(X_train, y_train, epochs=self.n_epochs,
                              validation_data=(X_valid, y_valid),
                              callbacks=[checkpoint_cb, early_stopping_cb])

        self.model = keras.models.load_model(self.out_directory + "model_DNN.h5")

    def predict(self, inputs):
        X_test = inputs
        if self.normalize:
            X_test = self.scaler.transform(X_test)
        if self.monte_carlo is True:
            y_probas = np.stack([self.model(X_test, training=True)
                             for sample in range(100)])  # make 100 predictions over the test set
            y_proba = y_probas.mean(axis=0)  # average to get a single prediction
            y_pred = np.argmax(y_proba, axis=1)
        else:
            y_pred = np.argmax(self.model.predict(X_test), axis=1)
        return y_pred
    
    def score(self, inputs):
        X_test, y_test = inputs
        if self.normalize:
            X_test = self.scaler.transform(X_test)
        if self.monte_carlo is True:
            y_probas = np.stack([self.model(X_test, training=True)
                             for sample in range(100)])  # make 100 predictions over the test set
            y_proba = y_probas.mean(axis=0)  # average to get a single prediction
            y_pred = np.argmax(y_proba, axis=1)
        else:
            y_pred = np.argmax(self.model.predict(X_test), axis=1)
        accuracy = np.sum(y_pred == y_test) / len(y_test)
        print("DNN accuracy: %.2f" % accuracy)
        return accuracy
