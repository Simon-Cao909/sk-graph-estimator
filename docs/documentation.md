# SKGraphEstimator

## Attributes

The required and optional attributes during initialization
- model_structure (list or tuple): Specifies the model architecture
                                   See architecture.md on how to properly format this
- build_setting (str, default="normal"): Decides the format of model_structure
                                         Must be either 'normal' or 'quick'
                                         See architecture.md for more information
- input_shape (tuple, default=None): The input shape
                                     If None, it will be guessed from the feature shape
- epochs (int, default=100): The number of epochs to train the model for
- batch_size (int, default=32): The batch size for training
- early_stopping (bool, default=True): Whether the model should stop training early if validation loss
                                       doesn't drop after n_iter_no_change iterations
- n_iter_no_change (int, default=10): The amount of iterations without validation loss change until
                                      the model stops training (only matters if early_stopping is True)
- validation_split (float): Should be between 0 and 1
                            This will determine how the training and validation data are split
                            with validation_split being the fraction of validation data
- verbose (int): If 0, nothing is printed. If 1, the process of training is printed
- loss (str or callable or list, default='mse'): The loss function used. See Keras for custom ones
                                                 If your model has a multi-output layer, you can use
                                                 a list where the ith loss corresponds to the ith output
- metrics (list, tuple, dict, or None, default=None): The metrics tracked during training
- optimizer (str, default='adam'): The optimizer used in training. See Keras for possibilities
- learning_rate (float, default=1e-4): The learning rate for training
- random_state (int or None, default=None): The random state. Used for reproducible results
- shuffle (bool, default=True): Whether to shuffle the data before training

## Public Methods

.build_model()
- This builds the model from model_structure declared during initialization
- Parameters: None
- :return (keras.Model): The fully built and compiled model

.fit()
- This trains the model on the given features and labels
- For vector valued outputs, this is identical to an sklearn model's .fit()
- Parameters:
  - X (array-like) - The features of shape (n_samples, ...)
  - y (array-like) - The labels of shape (n_samples, ...) or (n_samples,)
  - **fit_params - Any additional parameters used in Keras when calling keras.Model.fit(...)
- :return (self): The trained estimator

.predict()
- This predicts the labels given the features
- For vector valued outputs, this is identical to an sklearn model's .predict()
- Parameters:
  - X (array-like) - The features of shape (n_samples, ...)
- :return (numpy.ndarray): The labels of shape (n_samples, ...) or (n_samples,) (for one output case)

# SKGraphRegressor

## Attributes

Same as SKGraphEstimator

## Public Methods

Same as SKGraphEstimator

.score()
- Returns the R^2 score of the model
- Parameters
  - X (array-like) - The features of shape (n_samples, ...)
  - y (array-like) - The labels of shape (n_samples, ...) or (n_samples,)
- :return (float or ndarray of floats or None): The score or ndarray of scores

# SKGraphClassifier

## Attributes

Same as SKGraphEstimator

## Public Methods

Same as SKGraphEstimator

.score()
- Returns the accuracy score of the model
- Parameters
  - X (array-like) - The features of shape (n_samples, ...)
  - y (array-like) - The labels of shape (n_samples, ...) or (n_samples,)
- :return (float or ndarray of floats or None): The score or ndarray of scores

# SKGraphAutoencoder

## Attributes

The required and optional attributes during initialization
- encoder_structure (list or tuple): model structure for the encoder
                                        See architecture.md for how to format this
- decoder_structure (list or tuple): model structure for the decoder
                                        See architecture.md for how to format this
- model_type (str, default='standard'): Specifies the autoencoder type
                                        Must be either 'standard' or 'variational'
- build_setting (str, default='normal'): Decides the format of model_structure
                                            Must be either 'normal' or 'quick'
                                            See architecture.md for more information
- input_shape (tuple, default=None): The input shape
                                        If None, it will be guessed from the feature shape
- epochs (int, default=100): The number of epochs to train the model for
- batch_size (int, default=32): The batch size for training
- early_stopping (bool, default=True): Whether the model should stop training early if validation
                                        loss doesn't drop after n_iter_no_change iterations
- n_iter_no_change (int, default=10): The amount of iterations without validation 
                                        loss change until the model stops training
                                        (only matters if early_stopping is True)
- validation_split (float): Should be between 0 and 1
                            This will determine how the training and validation data are split
                            with validation_split being the fraction of validation data
- verbose (int): If 0, nothing is printed. If 1, the process of training is printed
- optimizer (str, default='adam'): The optimizer used in training. See Keras for possibilities
- learning_rate (float, default=1e-4): The learning rate for training
- random_state (int or None, default=None): The random state. Used for reproducible results
- shuffle (bool, default=True): Whether to shuffle the data before training

## Public Methods

.build_model()
- Builds and compiles the autoencoder using the given
  encoder_structure and decoder_structure
- :return (keras.Model): The autoencoder model

.fit()
- Trains the model on the given features
- Parameters:
    - X (array-like) - The features of shape (n_samples, ...)
    - y (None) - Leave this as None
    - fit_params - Any additional fit parameters used in Keras
- :return (self): The trained autoencoder

.predict()
- Same as SKGraphEstimator

.encode()
- Encodes the given input
- Parameters:
    - X (array-like) - The input array of shape (n_samples, *input_shape_)
                       or input_shape_
- :return (np.ndarray): The latent representation of X
                        of shape (n_samples, *latent_shape_)
                        or latent_shape_

.decode()
- Decodes the given latent representation
- Parameters:
    - latent (array-like) - The latent array of shape (n_samples, *latent_shape_)
                            or latent_shape_
- :return (np.ndarray): The output of the decoder
                        of shape (n_samples, *output_shape_)
                        or output_shape_

.score()
- Returns the negative MSE score
- Parameters
  - X (array-like) - The features of shape (n_samples, ...)
  - y (None) - Leave this as None
- :return (float): The score