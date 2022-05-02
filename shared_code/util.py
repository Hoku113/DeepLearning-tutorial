import numpy as np

def smooth_curve(x):
    """
    Using Smooth the lost function graph

    Reference: http://glowingpython.blogspot.jp/2012/02/convolution-with-numpy.html
    """

    window_len = 11
    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    w = np.kaiser(window_len, 2)
    y = np.convolve(w/w.sum(), s, mode='valid')
    return y[5:len(y)-5]