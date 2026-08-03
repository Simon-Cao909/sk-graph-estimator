from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score
import numpy as np
from tensorflow.keras.utils import to_categorical

from .estimator import SKGraphEstimator

class SKGraphClassifier(SKGraphEstimator, ClassifierMixin):
    '''
    SKGraphClassifier is the classifier branch of SKGraphEstimator.

    The only changes are that .predict() now returns class labels and 
    .predict_proba() is a new method that returns probabilities
    '''
    scoring_func = staticmethod(accuracy_score)
    must_be_vector = True

    def _format_data(self,X,y=None):
        '''
        Returns
        -------
        X : np.ndarray or list of np.ndarray
            X as an array for single input or 
            a list of arrays for multi-input
        
        y : np.ndarray or list of np.ndarray
            y as one-hot encoded
        '''
        data = SKGraphEstimator._format_data(self,X,y)

        if y is not None:
            data = list(data)
            if self.is_multi_output_:
                self.classes_ = [np.unique(target) for target in data[1]]
                data[1] = [to_categorical(
                                np.searchsorted(classes,target),
                                len(classes)
                        )
                        for target, classes in zip(data[1],self.classes_)]
            else:
                self.classes_ = np.unique(data[1])
                data[1] = to_categorical(
                                np.searchsorted(self.classes_,data[1]),
                                len(self.classes_)
                        )

        return data

    def predict_proba(self,X):
        '''
        Predicts the probabilities of each classification given the features.

        Parameters
        ----------
        X : array-like
            The features of shape ``(n_samples, *input_shape_)`` for single input.

            A list of features of shape ``(n_samples, *input_shape_[i])`` for multi-input.
        
        Returns
        -------
        prob : numpy.ndarray
            The probabilities of shape ``(n_samples, classes_)`` for single output.

            A list of probabilities each of shape ``(n_samples, classes_[i])`` for multi-output
        
        Raises
        ------
        ValueError
            If the dimension of the features is not equal 
            to the dimension of the input.

            For multi-input, if the number of inputs is not 
            equal to the number of features

            For multi-input, if the number of samples is not 
            equal between arrays
        '''
        return SKGraphEstimator.predict(self,X)

    def predict(self,X):
        '''
        Predicts the class labels given the features.

        Parameters
        ----------
        X : array-like
            The features of shape ``(n_samples, *input_shape_)`` for single input.

            A list of features of shape ``(n_samples, *input_shape_[i])`` for multi-input.
        
        Returns
        -------
        y : numpy.ndarray
            The class labels of shape ``(n_samples,)`` for single output.

            A list of class labels each of shape ``(n_samples,)`` for multi-output
        
        Raises
        ------
        ValueError
            If the dimension of the features is not equal 
            to the dimension of the input.

            For multi-input, if the number of inputs is not 
            equal to the number of features

            For multi-input, if the number of samples is not 
            equal between arrays
        '''
        prob = self.predict_proba(X)
        
        if self.is_multi_output_:
            out = [self.classes_[np.argmax(x, axis=1)] for x in prob]
        else:
            out = self.classes_[np.argmax(prob, axis=1)]

        return out