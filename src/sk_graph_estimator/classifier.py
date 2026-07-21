from sklearn.base import ClassifierMixin
from .estimator import SKGraphEstimator
from sklearn.metrics import accuracy_score
import numpy as np

class SKGraphClassifier(SKGraphEstimator, ClassifierMixin):
    '''
    SKGraphRegressor is the classifier branch of SKGraphEstimator
    The only thing different is that it now supports .score()
    '''

    def score(self,X,y):
        '''
        Scores the model based on how it performs on given data

        :param X (array-like): The features of shape (n_samples, ...)
        :param y (array-like): The labels of shape (n_samples, ...) or (n_samples,)

        :return (float or ndarray of floats or None): The accuracy score or ndarray of scores
        '''
        if len(self.output_shape_) <= 2:
            return accuracy_score(np.argmax(y, axis=1),
                                  np.argmax(self.predict(X), axis=1))
        else:
            raise ValueError("Scoring is only defined for vector-valued outputs")