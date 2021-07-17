import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras



class GMM:
    """ Gaussian Mixture Model (GMM) """
    def __init__(self, n_comp):
        self.n_comp = n_comp
        self.phi = self.mu = self.sigma = None
        self.training = False

    def fit(self, z, gamma):
        """ fit data to GMM model

        Parameters
        ----------
        z : tf.Tensor, shape (n_samples, n_features)
            data fitted to GMM.
        gamma : tf.Tensor, shape (n_samples, n_comp)
            probability. each row is correspond to row of z.
        """

        # Calculate mu, sigma
        # i   : index of samples
        # k   : index of components
        # l,m : index of features
        gamma_sum = tf.reduce_sum(gamma, axis=0)
        self.phi = tf.reduce_mean(gamma, axis=0)
        self.mu =  tf.einsum('ik,il->kl', gamma, z) / gamma_sum[:, np.newaxis]
        z_centered = tf.sqrt(gamma[:, :, np.newaxis]) * (z[:, np.newaxis, :] - self.mu[np.newaxis, :, :])
        self.sigma =  tf.einsum('ikl,ikm->klm', z_centered, z_centered) / gamma_sum[:, np.newaxis,
                                                                                     np.newaxis]
        # Calculate a cholesky decomposition of covariance in advance
        n_features = z.shape[1]
        min_vals = tf.linalg.diag(tf.ones(n_features, dtype=tf.float32)) * 1e-3
        self.L = tf.linalg.cholesky(self.sigma + min_vals[np.newaxis, :, :])

        self.training = False

    def fix_op(self):
        """ return operator to fix paramters of GMM
        """

        self.phi = self.mu = self.sigma = self.L = None


    def energy(self, z):
        """ calculate an energy of each row of z

        Parameters
        ----------
        z : tf.Tensor, shape (n_samples, n_features)
            data each row of which is calculated its energy.

        Returns
        -------
        energy : tf.Tensor, shape (n_samples)
            calculated energies
        """

        z_centered = z[:, np.newaxis, :] - self.mu[np.newaxis, :, :]  # ikl
        v = tf.linalg.triangular_solve(self.L, tf.transpose(z_centered, [1, 2, 0]))  # kli

        # log(det(Sigma)) = 2 * sum[log(diag(L))]
        log_det_sigma = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(self.L)), axis=1)

        # To calculate energies, use "log-sum-exp"
        d = z.shape[1]
        logits = tf.math.log(self.phi[:, np.newaxis]) - 0.5 * (tf.reduce_sum(tf.square(v), axis=1)
                                                          + d * tf.math.log(
                    2.0 * tf.constant(np.pi, dtype="float32")) + log_det_sigma[:, np.newaxis])
        energies = - tf.reduce_logsumexp(logits, axis=0)

        return energies

    def cov_diag_loss(self):
        diag_loss = tf.reduce_sum(tf.divide(1, tf.linalg.diag_part(self.sigma)))

        return diag_loss

