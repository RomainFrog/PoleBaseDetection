def depth_regression(X):
    return 1221.1025860930577 -6.13183029e+00*X + 1.03173169e-02*X**2 -5.76531414e-06*X**3


def depth_rescaling(X, old_range=[-32,1221], new_range=[1,10]):
    new_min, new_max = new_range
    old_min, old_max = old_range
    return (X - old_min) * (new_max - new_min) / (old_max - old_min) + new_min
