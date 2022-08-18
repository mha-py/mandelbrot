'''
Several functions for computing the mandelbrot sets pixel values.
- Simple iteration
- Cuda implementation
- Reference series based iteration
- (todo) Derivatives del z / del c
(c) 28.8.2020 mha

'''

import numpy as np
from numba import jit, njit, prange, cuda
import mpmath
from mpmath import mpf, mpc
import cmath
from math import ceil


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


def mesh_antialiased(width, sx, sy):
    '''Creates a grid compatible with antialiasing
    '''
    # Coordinates, y-coordinate has factor 4
    sy *= 4
    xarr = (np.arange(sx) - sx/2)/sx + 1e-6 # dont hit exactly the middle (necessary?)
    yarr = (np.arange(sy) - sy/2)/sx + 1e-6

    # Create meshgrid: xarr[i] and yarr[j] become xarr[i,j] and yarr[i,j]
    xarr, yarr = meshgrid(xarr, yarr)
    
    # Shift the x coordinates to get the right antialiasing sample points
    pxsize = width/sx
    xarr[:, 0::4] += - 1/8 * pxsize
    xarr[:, 1::4] += + 3/8 * pxsize
    xarr[:, 2::4] += - 3/8 * pxsize
    xarr[:, 3::4] += + 1/8 * pxsize
    
    return xarr, yarr


def antialias(array):
    '''Takes the mean of 4 sample points to get antialiasing effect.
    This function works on images that coordinates come from `mesh_antialiased`.
    '''
    sx, sy = array.shape
    return array.reshape((sx, sy//4, 4)).mean(-1)  # subpixels were stored in four y values



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
                
                
def mandelbrot_cuda(xarr, yarr, nmax):      ### , nblocks=512
    '''Determines the escape time for each point in the meshgrid of xarr, yarr.
    Wraps the actual cuda function
    '''
    res_mandel = np.zeros_like(xarr)
    nblocks = ceil(len(xarr)*len(xarr[0])/512)
    _mandelbrot_cuda[nblocks, 512](xarr, yarr, nmax, res_mandel)
    return res_mandel




########## high precision Variante ##########
# verwendet hochpräzisionszahlen für x und y
@myjit(nopython=False)
def f_highprecision(cx, cy, nmax):
    '''Mandelbrot iteration with high precision numbers.
    Accepts x and y as mpf numbers. Returns the series z_n.
    Formulas:
    z0 <- z0**2+c
    z1 <- 2*z0*z1+1
    z2 <- 2*z0*z2+z1**2
    z3 <- 2*z0*z3+2*z1*z2
    z4 <- 2*z0*z4+2*z0*z3+z2**2
    with zi the ith derivative of z(c) (actually there is still a faculty factor).
    '''
    z0re, z0im = cx, cy
    z0re = 0*z0re
    z0im = 0*z0im
    z1re = z0re
    z1im = z0im
    z2re = z0re
    z2im = z0im
    z3re = z0re
    z3im = z0im
    series = []
    coeff1 = []
    coeff2 = [] # stores second derivative of z(c)/2
    coeff3 = [] #  stores third derivative of z(c)/6
    
    pseudo_n = nmax
    for n in range(nmax):
        series.append((float(z0re), float(z0im)))
        coeff1.append((float(z1re), float(z1im)))
        coeff2.append((float(z2re), float(z2im)))
        coeff3.append((float(z3re), float(z3im)))
        z0re, z0im = z0re**2-z0im**2 + cx, 2*z0re*z0im + cy
        z1re, z1im = 2*(z0re*z1re-z0im*z1im) + 1, 2*(z0im*z1re+z0re*z1im)
        z2re, z2im = 2*(z0re*z2re-z0im*z2im + z1re**2 - z1im**2), 2*(z0re*z2im+z0im*z2re + 2*z1re*z1im)
        z3re, z3im = 2*(z0re*z3re-z0im*z3im + z1re*z2re-z1im*z2im), 2*(z0re*z3im+z0im*z3re + z1re*z2im+z1im*z2re)
        
        r = float(z0re)**2 + float(z0im)**2 # low precision is enough here
        if r > bound:
            pseudo_n = n - log2(ln(r))
            break
        
    return pseudo_n, np.array(series), np.array(coeff1), np.array(coeff2), np.array(coeff3)


    
def predevelope_xy(xarr, yarr, series, coeff1, coeff2, coeff3):
    '''Uses the tailor expansion of a reference series to get a starting point for the mandelbrot iteration (without actually iterating -> fast)
    xarr, yarr: Relative coordinates, offset to the reference series
    series, der1 to der3: Reference series and its derivatives in c
    '''
    # Find out the maximal distance to the reference, w, and the iteration n for which the third order term becomes too big to have a good quadratic approximation
    w = np.max(xarr**2+yarr**2)**(1/2)
    for n in range(len(series)):
        if np.linalg.norm(coeff3[n]) * w**3 >= 1e-5:
            break
    
    d1 = coeff1[n,0] + 1j*coeff1[n,1]
    d2 = coeff2[n,0] + 1j*coeff2[n,1]
    zarr = xarr + 1j*yarr
    zarr = d1*zarr + d2*zarr**2
    return n, np.real(zarr), np.imag(zarr)




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
def f_byreference(delx, dely, nmax, series, nstart=0, xstart=None, ystart=None):
    '''Calculates the series by using the difference del z = z - z^bar.
    series: a list containing z^bar_i (get it by f_highprecision)'''
    #re, im = delx, dely
    re, im = 0, 0
    if nstart > 0:
        re, im = xstart, ystart
    for n in range(nstart, min(nmax, len(series)-1)):
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
def _mandel_byreference_cuda(xarr, yarr, nmax, series, nstart, xstart, ystart, res_mandel):
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
    if nstart > 0:
        re, im = xstart[widx], ystart[widx]
    
    for n in range(nstart, nmax):
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
    
    
    
def mandel_byreference_cuda(xarr, yarr, nmax, series, nstart=0, xstart=None, ystart=None):
    '''Determines the escape time for each point in the meshgrid of xarr, yarr.
    Wraps the actual cuda function
    '''
    shape = xarr.shape
    xarr = xarr.flatten()
    yarr = yarr.flatten()
    N = len(xarr)
    if nstart > 0:
        xstart = xstart.flatten()
        ystart = ystart.flatten()
    else:
        xstart = np.empty_like(xarr)
        ystart = np.empty_like(yarr)
        
    res_mandel = np.zeros_like(xarr)
    _mandel_byreference_cuda[int(np.ceil(N/512)), 512](xarr, yarr, nmax, np.array(series), nstart, xstart, ystart, res_mandel)
    return res_mandel.reshape(shape)
    
    
    
    