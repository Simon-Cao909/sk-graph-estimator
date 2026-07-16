from tensorflow import keras
import tensorflow as tf
import numpy as np

class SKlearnLayer(keras.layers.Layer):
    '''
    Provides support for sklearn models to act as layers in
    a neural network
    '''
    def __init__(self, model):
        '''
        Attributes
        - model (scikit-learn regressor): A fully traiend sklearn model
                                          (ex. GradientBoostingRegressor)
        '''
        super().__init__()
        self.model = model
    
    def call(self, inputs):
        '''
        This method decides how an input is passed through the layer

        :param inputs (KerasTensor): The input tensor
        
        :return (KerasTensor): The tensor after the sklearn model is applied
        '''
        def regressor_pred(x):
            pred = self.model.predict(x)
            pred = np.asarray(pred, dtype=np.float32)

            if pred.ndim == 1:
                pred = pred.reshape(-1,1)

            return pred

        y = tf.numpy_function(
            regressor_pred,
            [inputs],
            tf.float32
        )

        y.set_shape((None,1))
        return y