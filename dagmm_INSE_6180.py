import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras



from custom_layers_INSE_6180 import Encoder_Block, Decoder_Block, Feature_Extraction_Block, Estimation_Block
from gmm_INSE_6180 import GMM



class DAGMM:
    """ Deep Autoencoding Gaussian Mixture Model.
    """

    MODEL_FILENAME = "DAGMM_model"
    SCALER_FILENAME = "DAGMM_scaler"

    def __init__(self, comp_hiddens, comp_activation,
            est_hiddens, est_activation, est_dropout_ratio=0.5,
            n_epochs=1000, batch_size = 128,
            lambda1=0.1, lambda2=0.0001, normalize=True, random_seed=42):

        self.comp_hiddens = comp_hiddens
        self.comp_activation = comp_activation
        self.est_hiddens = est_hiddens
        self.est_activation = est_activation
        self.est_dropout_ratio = est_dropout_ratio
        n_comp = est_hiddens[-1]
        self.gmm = GMM(n_comp)
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.normalize = normalize
        self.scaler = None
        self.seed = random_seed

    def build(self, inputs):
        # Building Model Using the Functional API
        self.input = input = keras.layers.Input(shape=inputs.shape[-1], name="input")  # x_ori
        self.encoder_ouput = encoder_output = Encoder_Block(self.comp_hiddens, self.comp_activation,
                                                            name="encoder")(input)  # z_c
        self.decoder_ouput = decoder_output = Decoder_Block(self.comp_hiddens, self.comp_activation,
                                                            input_size=inputs.shape[-1], name="decoder")(encoder_output)  # x_res
        self.estimation_input = estimation_input = Feature_Extraction_Block(name="feature_extraction")(
            (input, decoder_output, encoder_output))  # z
        self.estimation_ouput = estimation_output = Estimation_Block(self.est_hiddens, self.est_activation,
                                                                     self.est_dropout_ratio, name="estimation_net")(estimation_input)  # gamma

        self.model = model = keras.models.Model(inputs=[input], outputs=[estimation_output])


    def dagmm_loss(self, input):
        x_ori = self.model.layers[0](input)  # input layer
        z_c = self.model.layers[1](x_ori)  # encoder block
        x_res = self.model.layers[2](z_c)  # decoder block
        z = self.model.layers[3]((x_ori, x_res, z_c))  # feature_extraction block
        gamma = self.model.layers[4](z)
        reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.square(x_ori - x_res), axis=1), axis=0)
#         self.gmm.fix_op()
        self.gmm.fit(z, gamma)
        energy = self.gmm.energy(z)
        diag_loss = self.gmm.cov_diag_loss()
        energy_loss = self.lambda1 * tf.reduce_mean(energy)
        cov_loss = self.lambda2 * diag_loss
        loss = reconstruction_loss + energy_loss + cov_loss
        return loss

    def random_batch(self, X, y=None, batch_size=32):
        idx = np.random.randint(len(X), size=batch_size)
        return X[idx]

    def progress_bar(self, iteration, total, size=30):
        running = iteration < total
        c = ">" if running else "="
        p = (size - 1) * iteration // total
        fmt = "{{:-{}d}}/{{}} [{{}}]".format(len(str(total)))
        params = [iteration, total, "=" * p + c + "." * (size - p - 1)]
        return fmt.format(*params)

    def print_status_bar(self,iteration, total, loss, metrics=None, size=30):
        metrics = " - ".join(["{}: {:.4f}".format(n, m.result())
                              for m, n in zip([loss] + (metrics or []), ["mean_loss", "val_loss"])])
        # metrics = " - ".join(["{}: {:.4f}".format(n, m.result())
        #                       for m, n in [loss] + (metrics or []) for n in ["mean_loss", "val_loss"]])
        end = "" if iteration < total else "\n"
        print("\r{} - {}".format(self.progress_bar(iteration, total), metrics), end=end)

    def fit(self, inputs):
        tf.random.set_seed(self.seed)
        np.random.seed(seed=self.seed)
        if self.normalize:
            self.scaler = StandardScaler().fit(inputs)
            inputs = self.scaler.transform(inputs)
        X_train, X_valid = train_test_split(inputs, test_size=0.1, random_state=42)

        n_steps = len(X_train) // self.batch_size
        optimizer = keras.optimizers.Nadam(lr=0.01)
        # loss_fn = keras.losses.mean_squared_error
        mean_loss = keras.metrics.Mean()
        metrics = keras.metrics.Mean()
        minimum_val_loss = float("inf")
        best_epoch = None
        best_model = None

        for epoch in range(1, self.n_epochs + 1):
            print("Epoch {}/{}".format(epoch, self.n_epochs))
            for step in range(1, n_steps + 1):
                X_batch = self.random_batch(X_train, batch_size=self.batch_size)
                with tf.GradientTape() as tape:
                    main_loss = self.dagmm_loss(X_batch)
                    loss = tf.add_n([main_loss] + self.model.losses)
                gradients = tape.gradient(loss, self.model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                for variable in self.model.variables:
                    if variable.constraint is not None:
                        variable.assign(variable.constraint(variable))
                mean_loss(loss)
                # metrics = mean_loss
                self.print_status_bar(step * self.batch_size, len(inputs), mean_loss)
            val_loss = self.dagmm_loss(X_valid)
            if val_loss < minimum_val_loss:
                minimum_val_loss = val_loss
                self.best_epoch = best_epoch = epoch
                self.model.save_weights("my_keras_weights.ckpt")
                # self.best_model = best_model = clone_model(self.model)
            metrics(val_loss)
            self.print_status_bar(len(inputs), len(inputs), mean_loss, [metrics])
            for metric in [mean_loss] + [metrics]:
                metric.reset_states()
            print("Best Epoch: %d" % (best_epoch))
        self.model.load_weights("my_keras_weights.ckpt")


    def predict(self, inputs):
        if self.normalize:
            inputs = self.scaler.transform(inputs)
        hiddens = [layer for layer in self.model.layers]
        x_ori = hiddens[0](inputs)
        z_c = hiddens[1](x_ori)  # encoder block
        x_res = hiddens[2](z_c)  # decoder block
        z = hiddens[3]((x_ori, x_res, z_c))  # feature_extraction block
        energies = self.gmm.energy(z)

        return energies.numpy()

    def restore(self):
        model = self.model
        return model


