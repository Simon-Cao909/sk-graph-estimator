from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score

from .estimator import SKGraphEstimator

class SKGraphClassifier(SKGraphEstimator, ClassifierMixin):
    '''
    SKGraphRegressor is the classifier branch of SKGraphEstimator.

    The only thing different is the functionality of .score().
    '''
    scoring_func = staticmethod(accuracy_score)
    must_be_vector = True