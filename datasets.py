from sklearn import datasets as sklearn_ds
import numpy as np

def make(DSName,**kwargs):
    dataset_makers = {
        "moons" : sklearn_ds.make_moons,
        "circles": sklearn_ds.make_circles,
        "checkers": make_checkers,
        "elliptic_curve": make_elliptic_curve
    }
    return dataset_makers[DSName](**kwargs)

def make_checkers(n_samples=1000,margin = 0.8):
    x_train = np.random.uniform(low = 0, high = 3, size=(n_samples,2))
    y_train = np.floor(x_train)
    
    x_train = margin*(x_train - (y_train+0.5))

    x_train = x_train + (y_train+0.5)
    
    y_train = np.mod(y_train,2)
    y_train = np.logical_xor(y_train[:,0],y_train[:,1])
    return (x_train, y_train)

def make_elliptic_curve(n_samples=100,a=-1,b=1):
    x_train = np.random.uniform(low = -3, high = 3, size=(n_samples,2))
    y_coord = x_train[:,1]
    x_coord = x_train[:,0]
    y_train = 1*((y_coord**2-(x_coord**3+a*x_coord+b))>=0)
    return (x_train,y_train)