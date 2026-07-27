# SKGraphEstimator

## Attributes

The required and optional attributes during initialization

model_structure : list or tuple
    Specifies the model architecture.
    See architecture.md for how to format this.

build_setting : str, default="normal"
    Decides the format of model_structure.
    Must be either 'normal' or 'quick'.
    See architecture.md for more information.

input_shape : tuple, default=None
    The input shape.
    If None, it will be guessed from the feature shape.

epochs : int, default=100
    The number of epochs to train the model for.

batch_size : int, default=32
    The batch size for training.

early_stopping : bool, default=True
    Whether the model should stop training early if validation
    loss doesn't drop after n_iter_no_change iterations.

n_iter_no_change : int, default=10
    The amount of iterations without validation loss change until
    the model stops training.
    (Only matters if early_stopping is True.)

validation_split : float
    Should be between 0 and 1.
    This will determine how the training and validation data are split,
    with validation_split being the fraction of validation data.

verbose : int
    If 0, nothing is printed.
    If 1, the process of training is printed.

loss : str or callable or list, default="mse"
    The loss function used. See Keras for custom ones.
    If your model has a multi-output layer, you can use a list
    where the ith loss corresponds to the ith output.

metrics : list, tuple, dict, or None, default=None
    The metrics tracked during training.

optimizer : str, default="adam"
    The optimizer used in training.
    See Keras for possibilities.

learning_rate : float, default=1e-4
    The learning rate for training.

random_state : int or None, default=None
    The random state.
    Used for reproducible results.

shuffle : bool, default=True
    Whether to shuffle the data before training.

scoring_weights : list or tuple or None, default=None
    For multi-headed output only.
    Determines how the average score is weighted.
    The ith element of this denotes the weighting of the score
    corresponding to the ith output.

## Public Methods

### .build_model()

Builds the keras model from the given model structure.

Returns
-------
model : keras.Model
    The fully built and compiled model.

Raises
------
ValueError
    If the multi-output block did not come last.

### .fit()

Trains the model on the given features and labels.

Parameters
----------
X : array-like
    The features of shape ``(n_samples, *input_shape_)``.

y : array-like or list
    The labels of shape ``(n_samples, *output_shape_)`` or
    ``(n_samples,)`` for single output, or a list of labels for
    multi-output.

**fit_params
    Any additional fit parameters used in Keras.

Returns
-------
self
    The trained estimator.

Raises
------
ValueError
    If X is sparse.

### .predict()

Predicts the labels given the features.

Parameters
----------
X : array-like
    The features of shape ``(n_samples, *input_shape_)``.

Returns
-------
y or [y1,...] : numpy.ndarray or list
    The labels of shape ``(n_samples, *output_shape_)`` or
    ``(n_samples,)`` for single output.

    For multi-output, it is a list of ndarrays with shape
    ``(n_samples, *output_shape_)`` or ``(n_samples,)``.

### .score()
Scores the model based on how it performs on given data.

- For SKGraphEstimator, this returns the neg mse score.
- For SKGraphRegressor, this returns the r2 score.
- For SKGraphClassifier, this returns the accuracy score.

Parameters
----------
X : array-like
    The features of shape ``(n_samples, *input_shape_)``.

y : array-like or list
    The labels of shape ``(n_samples, *output_shape_)`` or
    ``(n_samples,)`` for single output, or a list of labels for
    multi-output.

Returns
-------
score : float or None
    The score or weighted mean of scores (for multi-output).

# SKGraphRegressor

## Attributes

Same as SKGraphEstimator

## Public Methods

Same as SKGraphEstimator

# SKGraphClassifier

## Attributes

Same as SKGraphEstimator

## Public Methods

Same as SKGraphEstimator

# SKGraphAutoencoder

## Attributes

The required and optional attributes during initialization

encoder_structure : list or tuple
    Model structure for the encoder.

    See architecture.md for how to format this.

decoder_structure : list or tuple
    Model structure for the decoder.

    See architecture.md for how to format this.

model_type : str, default='standard'
    Specifies the autoencoder type.

    Must be either ``'standard'`` or ``'variational'``.

build_setting : str, default='normal'
    Decides the format of model_structure.

    Must be either ``'normal'`` or ``'quick'``.

    See architecture.md for more information.

input_shape : tuple, default=None
    The input shape.

    If None, it will be guessed from the feature shape.

epochs : int, default=100
    The number of epochs to train the model for.

batch_size : int, default=32
    The batch size for training.

early_stopping : bool, default=True
    Whether the model should stop training early if validation
    loss doesn't drop after n_iter_no_change iterations.

n_iter_no_change : int, default=10
    The amount of iterations without validation loss change until
    the model stops training.

    (Only matters if early_stopping is True.)

validation_split : float
    Should be between 0 and 1.

    This will determine how the training and validation data are split,
    with validation_split being the fraction of validation data.

verbose : int
    If 0, nothing is printed.

    If 1, the process of training is printed.

optimizer : str, default='adam'
    The optimizer used in training.

    See Keras for possibilities.

learning_rate : float, default=1e-4
    The learning rate for training.

random_state : int or None, default=None
    The random state.

    Used for reproducible results.

shuffle : bool, default=True
    Whether to shuffle the data before training.

## Public Methods

### .build_model()
Builds and compiles the autoencoder using the given
encoder_structure and decoder_structure.

Returns
-------
model : keras.Model
    The autoencoder model.

Raises
------
ValueError
    If the model type is not 'standard' 
    or 'variational'.

NotImplementedError
    If the multi-output layer is used.

### .fit()
Trains the model on the given features.

Parameters
----------
X : array-like
    The features of shape ``(n_samples, *input_shape_)``.

y : None, default=None
    Leave this as None.

**fit_params
    Any additional fit parameters used in Keras.

Returns
-------
self
    The trained autoencoder.

Raises
------
ValueError
    If the input and output shape are not the same.

### .predict()

Same as SKGraphEstimator

### .encode()
Encodes the given input.

Parameters
----------
X : array-like
    The input array of shape ``(n_samples, *input_shape_)``
    or ``input_shape_``.

Returns
-------
latent : np.ndarray
    The latent representation of X of shape
    ``(n_samples, *latent_shape_)`` or ``latent_shape_``.

Raises
------
ValueError
    If the input shape is not equal 
    to ``(n_samples, *input_shape_)`` 
    or ``input_shape_``

### .decode()
Decodes the given latent representation.

Parameters
----------
latent : array-like
    The latent array of shape ``(n_samples, *latent_shape_)``
    or ``latent_shape_``.

Returns
-------
decoded : np.ndarray
    The output of the decoder of shape
    ``(n_samples, *output_shape_)`` or ``output_shape_``.

Raises
------
ValueError
    If the latent shape is not equal 
    to ``(n_samples, *latent_shape_)`` 
    or ``latent_shape_``

### .score()
Scores the model based on how it performs on given data.

Returns the negative MSE score.

Parameters
----------
X : array-like
    The features of shape ``(n_samples, *input_shape_)``.

y : None, default=None
    Leave this as None.

Returns
-------
score : float or None
    The negative MSE score.