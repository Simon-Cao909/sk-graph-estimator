This file contains information on how to properly format model_structure to create your Keras model

To start, model_structure must be a list or tuple of dictionaries. The nth dictionary in this list or tuple denotes the nth hidden layer or block. 
*** The output layer should be included in model_structure ***

Each dictionary in this list requires a key 'type', which denotes the type of layer or block you'd like to add. The value associated with this key can be:
- 'D' or 'dense'
- 'd' or 'dropout'
- 'C' or 'conv', 'convolution', 'conv2d'
- 'CT' or 'conv_transpose', 'convolution_transpose', 'conv2dtranspose'
- 'GN' or 'group_norm', 'group_normalization'
- 'BN' or 'batch_norm', 'batch_normalization'
- 'MP' or 'max_pooling'
- 'GAP' or 'global_avg_pooling', 'global_average_pooling'
- 'F' or 'flat', 'flatten'
- 'UP' or 'upsampling', 'upsample', 'upsampling2d'
- 'custom'

More advanced types:
- 'R' or 'resnet', 'residual'
- 'I' or 'incep', 'inception'
- 'X' or 'xcep', 'xception'
- 'regressor'
- 'NN' or 'neural'

## For simple layers:
Then, this dictionary can either have a 'specs' key, or the hyperparameters can be put directly as keys. If a 'specs' key is used, the associated value should be a dictionary containing the hyperparameters as keys:
```python
{'type':...,
 'specs':{...}}
```
OR
```python
{'type':...,
 ...}
```
The names of the keys are the names of the hyperparameters used by Keras (req is for required, opt for optional) and are also listed below. The expected value type is given in brackets () after the key
- For 'D': Input MUST have shape (n_samples, n_features)
    - 'units' or 'neurons' (req | int): Number of neurons for the layer (ex. 32)
    - 'activation' (req | str or callable): Activation function of the layer (ex. 'relu')
- For 'd':
    - 'rate' (req | float from [0,1)): Dropout rate (ex. 0.1)
- For 'C' or 'CT': Input MUST have shape (n_samples,height,width,channels) or (n_samples,channels,height,width)
    - 'filters' (req | int): The number of filters (ex. 8)
    - 'kernel_size' (req | tuple): The size of the input passed to each neuron in the filter (ex. (3,3))
    - 'strides' (opt | tuple | default=(1,1)): How we shift the group of neurons to use as input for the next neuron
    - 'padding' (opt | str | default='valid'): Either 'valid' or 'same'
    - 'data_format' (opt | str | default=None (Keras sets it to 'channels_last')): Either 'channels_first' or 'channels_last'
    - 'activation' (req | str or callable): Activation function of the layer
- For 'GN':
    - 'groups' (opt | int | default=32)
    - 'axis' (opt | int | default=-1)
    - 'epsilon' (opt | float | default=0.001)
    - 'center' (opt | bool | default=True)
    - 'scale' (opt | bool | default=True)
- For 'BN':
    - Same as 'GN' but no 'groups'
- For 'MP':
    - 'pool_size' (opt | tuple | default=(2,2))
    - 'strides' (opt | tuple or None | default=None)
    - 'padding' (opt | str | default='valid')
- For 'GAP' or 'F':
    - 'data_format' (opt | str | default=None (Keras sets it to 'channels_last'))
- For 'UP':
    - 'size' (opt | tuple | default=(2,2))
    - 'data_format' (opt | str | default=None (Keras sets it to 'channels_last'))
- For 'custom':
    - 'layer' (keras.layers.Layer or callable taking KerasTensor as input)
    - You can use this to make custom blocks as well, if you'd like

## For advanced blocks and layers:
- For 'R'
    - This is a ResNet block. If you do not know what this is you can read more on it on https://en.wikipedia.org/wiki/Residual_neural_network
    - For your dictionary, you must either have it to be of the form:
    ```python
    {'type':'R', # or 'resnet' or 'residual'
     'specs':{
        'layers':[...],
        'final_activation':...,
        'allow_projection':...,
        }
    }
    ```
    OR
    ```python
    {'type':'R', # or 'resnet' or 'residual'
     'layers':[...],
     'final_activation':...,
     'allow_projection':...}
    ```
    - The 'layers' key must have a list or tuple of dictionaries as the associated value. This list takes on the same form as model_structure. Think of it as building another model inside this block
    - 'final_activation' (opt | str or callable | default='linear'): This is the activation function applied on x + F(x)
    - 'allow_projection' (opt | bool | default=True): If set to True, in the case F(x) and x are of different shapes, this will handle it by projecting x onto the shape F(x). If set to False, an error will be raised if x and F(x) are of different shapes
- For 'I'
    - This is an inception block. If you do not know what this is you can read more on it on https://en.wikipedia.org/wiki/Inception_(deep_learning_architecture)
    - For your dictionary, it must be of the form:
    ```python
    {'type':'I', # or 'incep' or 'inception'
     'specs':{
        'branches':[
            [#1],
            [#2],
            ...
        ]
     }
    }
    ```
    OR
    ```python
    {'type':'I', # or 'incep' or 'inception'
     'branches':[
        [#1],
        [#2],
        ...
     ]
    }
    ```
    - Note that the 'branches' key is required. The associated value should be a list of lists. The nth list in this list denotes the nth branch. Each list should be of the form of model_structure. Think of it as building multiple different models in different branches
        - Do not actually put #n in the list please. That's just to make the documentation clearer
    - At the end, the branch outputs will be concatenated
- For 'X'
    - This is an xception block. If you do not know what this is you can read more on it on https://arxiv.org/abs/1610.02357
    - For your dictionary, it must be of the form:
    ```python
    {'type':'X', # or 'xcep' or 'xception'
     'specs':{
        'xcep_specs':[...],
        'final_activation':...,
        'allow_projection':...
        }
    }
    ```
    OR 
    ```python
    {'type':'X', # or 'xcep' or 'xception'
     'xcep_specs':[...],
     'final_activation':...,
     'allow_projection':...}
    ```
    - 'xcep_specs' is a required key here. Unlike for 'R' and 'I', the value associated with it is NOT of the form model_structrue. Instead, it is a list of the form:
    ```python
    [
     {'filters':...,'kernel_size':...,'padding':...,'activation':...},
     {'filters':...,},
     {'filters':...,},
     ...
    ]
    ```
    Where each dictionary denotes the specs for the interior SeparableConv2D layer
        - 'filters' (req | int): denotes the number of filters for the SeparableConv2D layer
        - 'kernel_size' (req | tuple): denotes the kernel size for the SeparableConv2D layer
        - 'padding' (opt | str | default='same'): denotes the padding for the layer
        - 'activation' (req | str or callable): denotes the activation function for the layer
    - 'final_activation' (opt | str or callable | default='linear'): This is the activation function applied on x + F(x)
    - 'allow_projection' (opt | bool | default=True): If set to True, in the case F(x) and x are of different shapes, this will handle it by projecting x onto the shape F(x). If set to False, an error will be raised if x and F(x) are of different shapes
- For 'regressor'
    - This creates a layer that acts like an sklearn regressor. The input of this layer should be the input of the regressor, and the output will be the output of the regressor
    - It should be of the form:
    ```python
    {'type':'regressor',
     'specs':{'model':...}}
    ```
    OR
    ```python
    {'type':'regressor',
     'model':...}
    ```
    - 'model' (req | scikit-learn regressor): This is a fully trained sklearn regressor (ex. GradientBoostingRegressor)
    - * Note that as most sklearn models do not use gradients, backpropagation will STOP at this layer. As a result, all previous layers will NOT be trained. Because of this, it is advised to put this near the front of the model
    - * As a further note, if this layer is put in parallel with a standard Keras layer (ex. Dense) in an inception block, then the standard Keras layer and all other standard layers before it will still be trained. Issues only arise when the sklearn layer is in series with the other layers
    - This is primarily used for combining the outputs of different models
- For 'NN'
    - This allows you to put a custom pre-trained Keras model into the neural network
    - For your dictionary, it must be of the form:
    ```python
    {'type':'NN', # or 'neural'
     'specs':{'model':...,'freeze':...}
    }
    ```
    OR
    ```python
    {'type':'NN', # or 'neural'
     'model':...,
     'freeze':...
    }
    ```
    - 'model' (req | keras.Model): This is a pretrained Keras model you would like to insert into your model
    - 'freeze' (opt | bool | default=False): Whether to freeze the weights and biases of the model
    - This is often used for transfer learning
    - * Unlike the 'regressor' layer, this does NOT stop backpropagation
* Note: model_structure is recursive. This means you can nest ResNet blocks, Inception blocks, and Xception blocks easily


## EXAMPLES ##
# Dense network:
```python
model_structure = [
    {'type':'D', 'units':128, 'activation':'relu'},
    {'type':'d', 'rate':0.2},
    {'type':'D', 'units':64, 'activation':'relu'},
    {'type':'D', 'units':1, 'activation':'linear'}
]
```
# Dense ResNet:
```python
model_structure = [
    {'type':'D', 'units':128, 'activation':'relu'},

    {
        'type':'R',
        'layers':[
            {'type':'D', 'units':128, 'activation':'relu'},
            {'type':'D', 'units':128, 'activation':'linear'}
        ],
        'final_activation':'relu'
    },

    {'type':'D', 'units':1, 'activation':'linear'}
]
```
# Convolutional network
```python
model_structure = [
    {'type':'C', 'filters':32, 'kernel_size':(3,3),
     'padding':'same', 'activation':'relu'},

    {'type':'MP'},

    {'type':'C', 'filters':64, 'kernel_size':(3,3),
     'padding':'same', 'activation':'relu'},

    {'type':'GAP'},

    {'type':'D', 'units':1, 'activation':'linear'}
]
```
# Convolutional ResNet:
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
# Inception:
```python
[
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
```
# Xception:
```python
[
    {
        'type':'X',
        'xcep_specs':[
            {
                'filters':64,
                'kernel_size':(3,3),
                'activation':'relu'
            },
            {
                'filters':64,
                'kernel_size':(3,3),
                'activation':'linear'
            }
        ],
        'final_activation':'relu'
    },

    {'type':'GAP'},
    {'type':'D','units':1,'activation':'linear'}
]
```
# Joke model (it's theoretically possible)
```python
[
    {
        'type':'R',
        'layers':[
            {
                'type':'I',
                'branches':[
                    [
                        {
                            'type':'X',
                            'xcep_specs':[
                                {
                                    'filters':64,
                                    'kernel_size':(3,3),
                                    'activation':'relu'
                                }
                            ]
                        }
                    ],

                    [
                        {
                            'type':'R',
                            'layers':[
                                {
                                    'type':'I',
                                    'branches':[
                                        [{'type':'C','filters':64,
                                          'kernel_size':(1,1),
                                          'activation':'relu'}]
                                    ]
                                }
                            ]
                        }
                    ]
                ]
            }
        ]
    },

    {'type':'GAP'},
    {'type':'D','units':1,'activation':'linear'}
]
```