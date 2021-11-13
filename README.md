# SDAGMM Tensorflow implementation
Deep Autoencoding Gaussian Mixture Model.

This implementation is based on the paper
**Deep Autoencoding Gaussian Mixture Model for Unsupervised Anomaly Detection**
[[Bo Zong et al (2018)]]

this is UNOFFICIAL implementation.

# Requirements
- python (3.5-3.6)
- Tensorflow >= 2.3
- Numpy
- sklearn

# Usage instructions
To use DAGMM model, you need to create "DAGMM" object.
At initialize, you have to specify next 4 variables at least.

- ``comp_hiddens`` : list of int
  - sizes of hidden layers of compression network
  - For example, if the sizes are ``[n1, n2]``,
  structure of compression network is:
  ``input_size -> n1 -> n2 -> n1 -> input_sizes``
- ``comp_activation`` : function
  - activation function of compression network
- ``est_hiddens`` : list of int
  - sizes of hidden layers of estimation network.
  - The last element of this list is assigned as n_comp.
  - For example, if the sizes are ``[n1, n2]``,
    structure of estimation network is:
    ``input_size -> n1 -> n2 (= n_comp)``
- ``est_activation`` : function
  - activation function of estimation network

Then you fit the training data, and predict to get energies
(anomaly score). It looks like the model interface of scikit-learn.

# File codes
- dagmm_INSE_6180: define the DAGMM class
- custom_layers_INSE_6180: define custom layers in the DAGMM model
- gmm_INSE_6180: calculate parameters of GMM, calculate the energy of samples
- dnn_INSE_6180: define the DNN class with the option of pretraining
- S-DAGMM (Outlier Detection): Section 2.2 in the manuscript
- DAGMM (Pretraining Technique): Section 2.3 in the manuscript



# Example
## Small Example
``` python
import tensorflow as tf
from dagmm_INSE_6180 import DAGMM

# Initialize
model_dagmm = DAGMM(comp_hiddens=[16,8,1], comp_activation="elu",
                  est_hiddens=[15, 3], est_activation="elu", est_dropout_ratio=0.2,
                  n_epochs=20, batch_size=128, normalize=True)
                  
# Fit the training data to model
model_dagmm.build(data)
model_dagmm.fit(data)

# Evaluate energies
# (the more the energy is, the more it is anomary)
energy = model_dagmm.predict(data)

```

## Jupyter Notebook Example
You can use next jupyter notebook examples using DAGMM model.
- [Simple DAGMM Example notebook](DAGMM - Example.ipynb) :
This example uses random samples of mixture of gaussian.



