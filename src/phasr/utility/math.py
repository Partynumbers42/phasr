import numpy as np # type: ignore

def derivative(f,precision=1e-6):
    def df(x):
        h=np.abs(x)*precision+precision
        return (f(x+h)-f(x))/h
    return df
