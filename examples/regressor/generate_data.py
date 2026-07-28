import numpy as np

def get_data():
    rng = np.random.default_rng(42)

    n = 5000

    X = rng.uniform(-3, 3, size=(n, 4))
    y = np.sin(X[:,0]) + X[:,1]**2 - 0.5*X[:,2]*X[:,3] + 0.3*np.cos(2*X[:,3])

    y += rng.normal(0, 0.2, size=n)

    return X,y

def get_multi_output_data():
    rng = np.random.default_rng(42)

    n = 5000

    X = rng.uniform(-3, 3, size=(n, 4))
    y1 = np.sin(X[:,0]) + X[:,1]**2 - 0.5*X[:,2]*X[:,3] + 0.3*np.cos(2*X[:,3])
    y2 = np.cos(X[:,0]) + X[:,1]**3 + 0.2*X[:,3]

    y1 += rng.normal(0, 0.2, size=n)
    y2 += rng.normal(0, 0.2, size=n)

    return X,y1,y2

def get_multi_input_data():
    rng = np.random.default_rng(42)

    n = 5000

    X1 = rng.uniform(-3, 3, size=(n, 4))
    X2 = rng.uniform(-1, 1, size=(n, 5))

    y = np.sin(X1[:,0]) + X2[:,1]**2 - 0.5*X1[:,2]*X2[:,3] + 0.3*np.cos(2*X1[:,3])

    y += rng.normal(0, 0.2, size=n)

    return X1,X2,y