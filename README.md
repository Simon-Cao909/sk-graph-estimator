# sk-graph-estimator
SKGraphEstimator is a neural network builder that combines scikit-learn's user-friendly features with Keras's versatility. You can create very complex networks easily, allowing you to test vastly different architectures quickly and efficiently. Further, as it is a subclass of scikit-learn's BaseEstimator, it can be used in things like GridSearchCV.

# Example code
```python
from sk_graph_estimator.estimator import SKGraphEstimator

model = SKGraphEstimator(model_structure=[
    {
        'type':'I',
        'branches':[
            [
                {'type':'C','filters':32,
                 'kernel_size':(1,1),
                 'activation':'relu'}
            ],

            [
                {'type':'C','filters':32,
                 'kernel_size':(3,3),
                 'padding':'same',
                 'activation':'relu'}
            ],

            [
                {'type':'C','filters':32,
                 'kernel_size':(5,5),
                 'padding':'same',
                 'activation':'relu'}
            ]
        ]
    },

    {'type':'GAP'},
    {'type':'D','units':1,'activation':'linear'}
]
,epochs=20,learning_rate=5e-3,random_state=42)
```

# Why I built this
Often, when performing research, it's useful to test different machine learning models. Whether that be through tweaking the hyperparameters or changing your framework, it was interesting to see how different models performed.

However, when trying to work with neural networks, I encountered a block. I either had to use sklearn's MLPRegressor, which significantly limits your ability, or attempt to work with KerasRegressor from scikeras, which still requires you to rewrite dozens of lines of code when attempting to use a different model architecture. Debugging was a chore, and for more complex networks like Inception, ResNet, and Xception, it was a pain to code and read.

This is why I created SKGraphEstimator. It allows you to easily specify complex model structures with something resembling a flow chart, making it significantly more user-friendly while maintaining most of Keras's versatility. You can create complex and more readable just by coding something like:
```python
[
    {'type':'C', 'filters':32,
     'kernel_size':(3,3),
     'padding':'same',
     'activation':'relu'},

    {
        'type':'R',
        'layers':[
            {'type':'C','filters':32,
             'kernel_size':(3,3),
             'padding':'same',
             'activation':'relu'},

            {'type':'C','filters':32,
             'kernel_size':(3,3),
             'padding':'same',
             'activation':'linear'}
        ],
        'final_activation':'relu'
    },

    {'type':'GAP'},
    {'type':'D','units':1,'activation':'linear'}
]
```
Then, with many model structures created already, you can run it through a GridSearchCV or add it to your code alongside any other sklearn regressor, and it will behave just fine.

# Attributes

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
- loss (str or callable, default='mse'): The loss function used. See Keras for custom ones
- metrics (list, tuple, dict, or None, default=None): The metrics tracked during training
- optimizer (str, default='adam'): The optimizer used in training. See Keras for possibilities
- learning_rate (float, default=1e-4): The learning rate for training
- random_state (int or None, default=None): The random state. Used for reproducible results
- shuffle (bool, default=True): Whether to shuffle the data before training
- is_classifier (bool, default=False): Whether the model is a classifier
                                       This will only change how the scoring behaves

# Public Methods

.build_model()
- This builds the model from model_structure declared during initialization
- Parameters: None
- :return (keras.Model): The fully built and compiled model

.fit()
- This trains the model on the given features and labels
- For vector valued outputs, this is identical to any sklearn .fit()
- Parameters:
  - X (array-like) - The features of shape (n_samples, ...)
  - y (array-like) - The labels of shape (n_samples, ...) or (n_samples,)
  - **fit_params - Any additional parameters used in Keras when calling keras.Model.fit(...)
- :return (self): The trained estimator

.predict()
- This predicts the labels given the features
- For vector valued outputs, this is identical to any sklearn .predict()
- Parameters:
  - X (array-like) - The features of shape (n_samples, ...)
- :return (numpy.ndarray): The labels of shape (n_samples, ...) or (n_samples,) (for one output case)

.score()
- Only for SKGraphClassifier and SKGraphRegressor
- Returns the scoring of the model if applicable
  - If you are using SKGraphClassifier, the accuracy score will be returned
  - If you are using SKGraphRegressor, the R^2 score will be returned
- Parameters
  - X (array-like) - The features of shape (n_samples, ...)
  - y (array-like) - The labels of shape (n_samples, ...) or (n_samples,)
- :return (float or ndarray of floats or None): The score or ndarray of scores