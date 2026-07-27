# sk-graph-estimator
SKGraphEstimator is a graph-based neural network framework built on top of Keras that lets one describe complex architectures as composable graph blocks. It is also fully compatible with scikit-learn's estimator API

## Features

Graph-based architecture DSL

ResNet, Inception, and Xception

Transfer learning

Embed pre-trained scikit-learn regressors into neural networks

Standard and variational autoencoders

scikit-learn BaseEstimator compatibility

## Getting started

```bash
git clone git@github.com:Simon-Cao909/sk-graph-estimator.git
cd sk-graph-estimator
pip install .
```

```python
from sk_graph_estimator.estimator import SKGraphEstimator
```

## Documentation

You can find documentation in docs/documentation.md or on https://sk-graph-estimator.readthedocs.io/en/latest/

## Example code
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

## Why I built this
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