import numpy as np

def get_data():
    rng = np.random.default_rng(42)

    n = 5000

    X = rng.uniform(-3, 3, size=(n, 4))
    y = np.sin(X[:,0]) + X[:,1]**2 - 0.5*X[:,2]*X[:,3] + 0.3*np.cos(2*X[:,3])

    y += rng.normal(0, 0.2, size=n)

    return X,y