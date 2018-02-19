import numpy as np
from numba import jit
from scipy import signal

@jit
def difference_detection(im, t):
    th, tw = t.shape
    ih, iw = im.shape
    
    d = np.zeros_like(im)
    for i in range(ih-th):
        for j in range(iw-tw):
            d[i,j] = ((im[i:i+th, j:j+tw] - t)**2).sum()
    d /= d.max()
    d[ih-th:] = 1.0
    d[:, iw-tw:] = 1.0
    return 1-d

def matched_filter(im, t):
    # low response is better
    th, tw = t.shape
    ih, iw = im.shape
    
    d = signal.correlate2d(1-im, 1-t, mode='valid') / (1-t).sum()
    d = np.pad(d, ((0, th-1), (0, tw-1)), 'constant', constant_values=(0, 0))
    assert d.shape==im.shape
    return d

@jit
def cross_correlation(im, t):
    tdiff = t - t.mean()
    th, tw = t.shape
    ih, iw = im.shape
    
    d = np.zeros_like(im)
    for i in range(ih-th):
        for j in range(iw-tw):
            patch = im[i:i+th, j:j+tw]
            normer = np.sqrt(((patch - patch.mean())**2 * tdiff**2).sum()) + 1e-10
            d[i,j] =  ((patch - patch.mean()) * tdiff).sum()/normer
    return (d - d.min()) / (d.max() - d.min())
            
def draw_boxes(thresholded, pat_shape):
    th, tw = pat_shape
    boxes = np.zeros_like(thresholded)
    ay, ax = np.where(thresholded)
    for y,x in zip(ay, ax):
        boxes[y:y+th, x] = 1
        boxes[y:y+th, x+tw-1] = 1
        boxes[y+th-1, x:x+tw] = 1
        boxes[y, x:x+tw] = 1
    return boxes

def draw_targets(thresholded, d=1):
    targets = np.zeros(thresholded.shape)
    ay, ax = np.where(thresholded)
    for y,x in zip(ay, ax):
        targets[y-d//2:y+d//2+1, x-d//2:x+d//2+1] = 1
    return targets