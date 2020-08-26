'''
Several functions for computing the mandelbrot sets pixel values.
- Simple iteration
- Cuda implementation
- Reference series based iteration
- (todo) Derivatives del z / del c
(c) 20.8.2020 mha

TODO: - mandel_byreference: spinnt. 
      - bei allen funktionen: wenn nicht gebreakt wird, dann n füllen in array
      - ......
'''

import numpy as np
from numba import jit, njit, prange, cuda
import mpmath
from mpmath import mpf, mpc
import cmath


bound = 20

ln = np.log
log2 = np.log2


myjit = jit
#myjit = lambda f: f


########## helper function ##########
def meshgrid(xarr, yarr):
    '''Creates a meshgrid of xarr and yarr, i. e. a Cartesian product of both sets.
    '''
    x_arr = np.repeat(xarr[:, None], len(yarr), -1)
    y_arr = np.repeat(yarr[None, :], len(xarr), 0)
    return x_arr, y_arr


########## legacy ##########
'''
# basis implementation (f nicht inline)
@njit
def f(cx, cy, nmax):
    'Escape time for point (cx, cy)'
    re, im = 0, 0
    for n in range(nmax):
        re, im = re**2-im**2 + cx, 2*re*im + cy
        if re**2+im**2 > bound:
            r = re**2+im**2
            pseudo_n = n - log2(ln(r))
            return pseudo_n
    return n

@jit
def mandelbrot_slow(xarr, yarr, nmax):
    'Determines the escape time for each point in the meshgrid of xarr, yarr'
    mandel = np.empty((len(xarr), len(yarr)))
    for i in range(len(xarr)):
        for j in range(len(yarr)):
            mandel[i,j] = f(xarr[i], yarr[j], nmax)
    return np.array(mandel)''';



########## basis implementation ##########
@myjit
def mandelbrot_mesh(xarr, yarr, nmax):
    '''Determines the escape time for each point in the meshgrid of xarr, yarr.
    I. e. the points are the cartesian product of the sets xarr and yarr.
    '''
    mandel = nmax * np.ones((len(xarr), len(yarr)))
    for i in range(len(xarr)):
        for j in range(len(yarr)):
            cx, cy = xarr[i], yarr[j]
            re, im = 0, 0
            for n in range(nmax):
                re, im = re**2-im**2 + cx, 2*re*im + cy
                if re**2+im**2 > bound:
                    r = re**2+im**2
                    pseudo_n = n - log2(ln(r))
                    mandel[i,j] = pseudo_n
                    break
    return mandel

@myjit
def mandelbrot(xarr, yarr, nmax):
    '''Determines the escape time for each point in xy_arr.
    '''
    N, M = xarr.shape
    mandel = nmax * np.ones((N, M))
    for i in range(N):
        for j in range(M):
            cx, cy = xarr[i,j], yarr[i,j]
            re, im = 0., 0.
            for n in range(nmax):
                re, im = re**2-im**2 + cx, 2*re*im + cy
                if re**2+im**2 > bound:
                    r = re**2+im**2
                    pseudo_n = n - log2(ln(r))
                    mandel[i,j] = pseudo_n
                    break
    return mandel



########## cuda implementation ##########
@cuda.jit
def _mandelbrot_cuda(xarr, yarr, nmax, res_mandel):
    '''Determines the escape time for each point in the meshgrid of xarr, yarr
    Expecting res_mandel to be an array of shape of xarr or yarr'''
    
    N, M = xarr.shape
    
    tidx = cuda.threadIdx.x
    bidx = cuda.blockIdx.x
    bdim = cuda.blockDim.x
    widx = bidx*bdim + tidx
    
    if widx > N*M:
        return
    
    i, j = widx//M, widx%M
    cx, cy = xarr[i,j], yarr[i,j]
    re, im = 0., 0.
    res_mandel[i,j] = nmax
    for n in range(nmax):
        re, im = re**2-im**2 + cx, 2*re*im + cy
        if re**2+im**2 > bound:
            r = re**2+im**2
            pseudo_n = ( n - cmath.log(cmath.log(r))/cmath.log(2) ).real
            res_mandel[i,j] = pseudo_n
            break
                
def mandelbrot_cuda(xarr, yarr, nmax, nblocks=512):
    '''Determines the escape time for each point in the meshgrid of xarr, yarr.
    Wraps the actual cuda function
    '''
    res_mandel = np.zeros_like(xarr)
    _mandelbrot_cuda[nblocks, 512](xarr, yarr, nmax, res_mandel)
    return res_mandel



########## high precision Variante ##########
# verwendet hochpräzisionszahlen für x und y
from numba.typed import List
@myjit(nopython=False)
def f_highprecision(cx, cy, nmax):
    '''Mandelbrot iteration with high precision numbers.
    Accepts x and y as mpf numbers. Returns the series z_n.
    '''
    ###zre, zim = 0.*cx, 0.*cy
    zre, zim = cx, cy
    zre = 0*zre
    zim = 0*zim
    series = []
    ## series_derv = [] # derivative in c: dz_(i+1)/dc = 2*z_i*(dz_i/dc) + 1
    for n in range(nmax):
        series.append((float(zre), float(zim)))
        zre, zim = zre**2-zim**2 + cx, 2*zre*zim + cy
        r = float(zre)**2 + float(zim)**2 # low precision is enough
        if r > bound:
            pseudo_n = n - log2(ln(r))
            #pseudo_n = n - mpmath.log(mpmath.log(r))/mpmath.log(2)
            return pseudo_n, series
    return nmax, np.array(series)


########## By Reference ##########
# Verwendet die Iteration für die Differenz von delta z zu einem Referenzwert z-balken.
@myjit
def _f_resume(cx, cy, nstart, nmax, zx, zy):
    '''Helper function which resumes the series at z given into the function.
    This is needed when iteration by reference series is not possible anymore since
    the reference series diverges.
    You wanna choose a reference series which diverges late or not at all not prevent
    this function from being called.'''
    zre, zim = zx, zy
    for n in range(nstart, nmax):
        zre, zim = zre**2-zim**2 + cx, 2*zre*zim + cy
        if zre**2+zim**2 > bound:
            r = zre**2+zim**2
            pseudo_n = n - log2(ln(r))
            return pseudo_n
    return nmax

@myjit
def f_byreference(delx, dely, nmax, series):
    '''Calculates the series by using the difference del z = z - z^bar.
    series: a list containing z^bar_i (get it by f_highprecision)'''
    #re, im = delx, dely
    re, im = 0, 0
    for n in range(min(nmax, len(series)-1)):
        ref_re, ref_im = series[n]
        re, im = 2*(re*ref_re-im*ref_im) + (re**2-im**2) + delx,  2*(im*ref_re+re*ref_im) + (2*re*im) + dely
        if re**2+im**2 > 1e-4: break # go on normal
    # transform back to absolute coordinates instead of relative
    re = re + series[n+1][0]
    im = im + series[n+1][1]
    cx = series[1][0] + delx   # add the c of the reference (which is the 1st entry in the series)
    cy = series[1][1] + dely
    return _f_resume(cx, cy, n, nmax, re, im)

@myjit
def mandel_byreference(xarr, yarr, nmax, series):
    '''Calculates the mandelbrot figure by using a reference series (create it using f_highprecision).
    Using a reference series simulates high precision while still using double precision.
    '''
    N, M = xarr.shape
    mandel = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            cx, cy = xarr[i,j], yarr[i,j]
            mandel[i,j] = f_byreference(cx, cy, nmax, series)
    return mandel


@cuda.jit
def _mandel_byreference_cuda(xarr, yarr, nmax, series, res_mandel):
    '''Like `mandel_byreference` but on cuda devic
    Expects flattened xarr and yarr, and result will be also flattened
    '''
    N = len(xarr)
    ## requires: res_mandel = np.empty((N, M))
    
    tidx = cuda.threadIdx.x
    bidx = cuda.blockIdx.x
    bdim = cuda.blockDim.x
    widx = bidx*bdim + tidx
    
    if widx > N:
        return
    
    
    #### This is f_byreference ####
    delx, dely = xarr[widx], yarr[widx]
    
    re, im = 0, 0
    for n in range(nmax):
        if n >= len(series)-1: break
        ref_re, ref_im = series[n]
        re, im = 2*(re*ref_re-im*ref_im) + (re**2-im**2) + delx,  2*(im*ref_re+re*ref_im) + (2*re*im) + dely
        if re**2+im**2 > 1e-4: break # go on normal
    # transform back to absolute coordinates instead of relative
    re = re + series[n+1][0]
    im = im + series[n+1][1]
    cx = series[1][0] + delx   # add the c of the reference (which is the 1st entry in the series)
    cy = series[1][1] + dely
    
    ### This is f_resume ###
    pseudo_n = nmax
    zre, zim = re, im
    nstart = n
    for n in range(nstart, nmax):
        zre, zim = zre**2-zim**2 + cx, 2*zre*zim + cy
        if zre**2+zim**2 > bound:
            r = zre**2+zim**2
            #pseudo_n = n - log2(ln(r))
            pseudo_n = ( n - cmath.log(cmath.log(r))/cmath.log(2) ).real
            break
        
    ### Save this value ###
    res_mandel[widx] = pseudo_n
    
    
    
def mandel_byreference_cuda(xarr, yarr, nmax, series):
    '''Determines the escape time for each point in the meshgrid of xarr, yarr.
    Wraps the actual cuda function
    '''
    shape = xarr.shape
    xarr = xarr.flatten()
    yarr = yarr.flatten()
    N = len(xarr)
    res_mandel = np.zeros_like(xarr)
    _mandel_byreference_cuda[int(np.ceil(N/512)), 512](xarr, yarr, nmax, np.array(series), res_mandel)
    return res_mandel.reshape(shape)
    
    
    
    
    
    
    