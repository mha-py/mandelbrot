# Version vom 31.7.2020

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output, display



class Display:
    
    def __init__(self, doit=True):
        self.doit = doit
        
    def __enter__(self):
        if self.doit:
            clear_output(wait=True)
        
    def __exit__(self, typ, wert, traceback):
        if self.doit:
            display(plt.gcf())
            
            
def _arrange_pts(pts):
    '''Puts pts of whatever format into an array of n x 2
    More intelligent version of np.array(listofitems).'''
    if type(pts) is np.ndarray:
        return pts.reshape((-1, 2))
    if type(pts) in (tuple, list):
        if len(pts) == 0: return None
        if type(pts[0]) is np.ndarray and len(pts[0].shape) == 2:
            return np.stack([ _arrange_pts(ps) for ps in pts ])
        else:
            return np.array(pts).reshape((-1, 2))
            
## Wrapper für plt-Funktionen (plt.plot und co.)
            
def plot(pts, *args, **kwargs):
    #pts = np.array(pts)
    pts = _arrange_pts(pts)
    if pts is None: return
    plt.plot(pts[:,0], pts[:,1], *args, **kwargs)
            
            
def scatter(pts, *args, **kwargs):
    pts = _arrange_pts(pts)
    if pts is None: return
    plt.scatter(pts[:,0], pts[:,1], *args, **kwargs)
    
def arrow(pts, dr, *args, **kwargs):
    pts = _arrange_pts(pts)
    dr = _arrange_pts(dr)
    if pts is None: return
    plt.arrow(pts[:,0], pts[:,1], dr[:,0], dr[:,1], *args, **kwargs)
    
    
## Funktionen plotten
    
def plot_f(f, xs, *args, **kwargs):
    'Plots a function f(x) (or several functions if f is a list)'
    if type(f) not in (list, tuple): f = [f]
    for ff in f:
        ys = ff(xs)
        plt.plot(xs, ys, *args, **kwargs)
    
#def f_add(f, g, *args):
#    return lambda x: f(x) + g(x) #+ sum([h(x) for h in args])



## Bilder in realen Größe anzeigen
def showimg_actualsize(img, *args, **kwargs):
    'Creates a new figure and plots img into it in its actual size.'
    h, w = img.shape[:2]
    dpi = 80.
    fig = plt.figure(figsize=(w/dpi, h/dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(img, *args, **kwargs)
    
def downscale(img, f):
    'Scales down the image by a factor f (example f=2, f=4)'
    if len(img.shape)==2: return downscale_gray(img)
    h, w, c = img.shape
    h, w = (h//f)*f, (w//f)*f
    img = img[:h, :w] # crop to a multiple of f
    img = img.reshape((h//f, f, w//f, f, c)).mean((1, 3))
    return img
    
def downscale_gray(img, f):
    'Scales down the image by a factor f (example f=2, f=4)'
    h, w = img.shape
    h, w = (h//f)*f, (w//f)*f
    img = img[:h, :w] # crop to a multiple of f
    img = img.reshape((h//f, f, w//f, f)).mean((1, 3))
    return img

def upscale(img, f):
    'Scales up the image by a factor f'
    if len(img.shape)==2: return upscale_gray(img)
    h, w, c = img.shape
    img = img.reshape((h, 1, w, 1, c))
    img = np.tile(img, (1, f, 1, f, 1))
    img = img.reshape((h*f, w*f, c))
    return img

def upscale_gray(img, f):
    'Scales up the image by a factor f'
    h, w = img.shape
    img = img.reshape((h, 1, w, 1))
    img = np.tile(img, (1, f, 1, f))
    img = img.reshape((h*f, w*f))
    return img
    