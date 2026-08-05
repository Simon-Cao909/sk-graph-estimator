from sklearn.base import RegressorMixin
from sklearn.metrics import r2_score

from .estimator import DeepEstimator

class DeepRegressor(DeepEstimator, RegressorMixin):
    '''
    DeepRegressor is the regressor branch of DeepEstimator.

    The only thing different is the functionality of .score().
    '''
    scoring_func = staticmethod(r2_score)
    must_be_vector = True