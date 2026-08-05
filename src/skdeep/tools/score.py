import numpy as np

def compute_score(ytrue,ypred,scoring_func,weights=None,must_be_vector=False):
    if isinstance(ypred,(list,tuple)):
        if weights is None:
            weights = np.asarray([1.0/len(ypred)]*len(ypred))

        if not must_be_vector or all(np.asarray(y).ndim <= 2 for y in ypred):
            all_scores = [scoring_func(yt,yp) for yt,yp in zip(ytrue,ypred)]
            return np.average(all_scores,weights=weights)
        else:
            raise ValueError("Scoring is only defined for vector-valued outputs")
    else:
        ypred = np.asarray(ypred)
        if not must_be_vector or ypred.ndim <= 2:
            return scoring_func(ytrue,ypred)
        else:
            raise ValueError("Scoring is only defined for vector-valued outputs")

def neg_mse_score(ytrue,ypred):
    return -np.mean((ytrue - ypred) ** 2)