
import numpy as np
import torch


GPU = True



def np2t(*args):
    'Converts a numpy array to a torch array'
    res = [torch.from_numpy(np.array(x, dtype='float32')) for x in args]
    if GPU:
        res = [x.cuda() for x in res]
        
    if len(res)==1:
        return res[0]
    else:
        return res


def t2np(*args):
    'Converts a torch array to a numpy array'
    res = [x.detach().cpu().numpy() for x in args]
    
    if len(res)==1:
        return res[0]
    else:
        return res
    
    
def load_checkpoint(model, path):
    'Load with mismatched layer sizes'
    load_dict = torch.load(path)
    model_dict = model.state_dict()
    for k in model_dict.keys():
        if k in load_dict:
            if load_dict[k].shape == model_dict[k].shape:
                model_dict[k] = load_dict[k]
            else:
                print('Ignoring (since shape mismatch)', k)
        else:
            print('Ignoring (since not in data)', k)
    model.load_state_dict(model_dict)
    
    

#### Batchgenerator ####

def batchgenerator_v2(indarray, readout_xy, batchsize=32, permute=True):
    indarray = np.asarray(indarray)
    
    numind = len(indarray)
    while True:
        if permute:
            perm = np.random.permutation(numind)
        else:
            perm = np.arange(numind)
        for i in range((numind)//batchsize):
            inds = indarray[perm[i*batchsize:(i+1)*batchsize]]
            xs, ys = readout_xy(inds)
            yield xs, ys

            
def batchgenerator_v2_single(indarray, readout_xy, batchsize=32, permute=True):
    # for only xs output (instead of xs and ys)
    indarray = np.asarray(indarray)
    
    numind = len(indarray)
    while True:
        if permute:
            perm = np.random.permutation(numind)
        else:
            perm = np.arange(numind)
        for i in range((numind)//batchsize):
            inds = indarray[perm[i*batchsize:(i+1)*batchsize]]
            #xs, ys = readout_xy(inds)
            xs = readout_xy(inds)
            yield xs
            
            
def getbatchgen(indarray, readout_xy, batchsize=32, permute=True, single=False):
    '''Wrapper of the batchgenerator, which additionally gives the number of steps per epoche.
    This value is needed by the model.fit_generator function.'''
    perepoch = len(indarray) // batchsize
    if single:
        return batchgenerator_v2_single(indarray, readout_xy, batchsize, permute), perepoch
    else:
        return batchgenerator_v2(indarray, readout_xy, batchsize, permute), perepoch


def augment8(x):
    r = np.random.rand
    if r()<.5:
        x = x[:,::-1]
    if r()<.5:
        x = x[:,:,::-1]
    if r()<.5:
        x = x.swapaxes(2,1)
    return x


def augment8_dual(x, y):
    r = np.random.rand
    if r()<.5:
        x, y = x[:,::-1], y[:,::-1]
    if r()<.5:
        x, y = x[:,:,::-1], y[:,:,::-1]
    if r()<.5:
        x, y = x.swapaxes(2,1), y.swapaxes(2,1)
    return x, y



#### HSV and RGB ####

from colorsys import rgb_to_hsv, hsv_to_rgb
def rgb2hsv(img):
    shape = img.shape
    img = img.reshape(-1, 3)
    for k in range(img.shape[0]):
        img[k] = rgb_to_hsv(*img[k])
    return img.reshape(shape)
def hsv2rgb(img):
    shape = img.shape
    img = img.reshape(-1, 3)
    for k in range(img.shape[0]):
        img[k] = hsv_to_rgb(*img[k])
    return img.reshape(shape)


#### Blurring ####

from scipy import signal

def blur_imgs(imgs, radius):
    if radius == 0:
        return imgs
    n = int(2.5*radius)*2+1 # odd number important!
    kernel1D = signal.gaussian(n, std=radius)
    kernel = np.outer(kernel1D, kernel1D)
    kernel = kernel[None, :, :, None] # add dummy channel
    kernel = kernel / kernel.sum()
    imgs = signal.convolve(imgs, kernel, mode='full')
    imgs = imgs[:, n//2:-n//2+1, n//2:-n//2+1]
    return imgs



#### Bildverarbeitung und Augmentation ####

def downscale(imgs, f):
    b, w, h, c = imgs.shape
    imgs = imgs.reshape((b, w//f, f, h//f, f, c))
    imgs = imgs.mean(4).mean(2)
    return imgs

def fliprot(imgs):
    if np.random.rand()<0.5:
        imgs = imgs[:,::-1]
    if np.random.rand()<0.5:
        imgs = imgs[:,:,::-1]
    if np.random.rand()<0.5:
        imgs = imgs.transpose((0,2,1,3))
    return imgs

def fliprot_mult(imgs1, imgs2):
    if np.random.rand()<0.5:
        imgs1 = imgs1[:,::-1]
        imgs2 = imgs2[:,::-1]
    if np.random.rand()<0.5:
        imgs1 = imgs1[:,:,::-1]
        imgs2 = imgs2[:,:,::-1]
    if np.random.rand()<0.5:
        imgs1 = imgs1.transpose((0,2,1,3))
        imgs2 = imgs2.transpose((0,2,1,3))
    return imgs1, imgs2


def randomcrop_dual(y1, y2, h, w=None):
    if isinstance(w, type(None)):
        w = h
    if h<=1:
        h = int(h*y1.shape[1])
        w = int(w*y1.shape[2])
    dy = np.random.randint(y1.shape[1]-h)
    dx = np.random.randint(y1.shape[2]-w)
    y1 = y1[:, dy:dy+h, dx:dx+w, :]
    y2 = y2[:, dy:dy+h, dx:dx+w, :]
    return y1, y2



#############

import torch
from torch import nn
F = nn.functional
relu = F.relu


def l1l2(x, y, r=1/10):
    # Mixture of l1 and l2 lossfunction
    return (x-y)**2 + r*torch.abs(x-y)


def L1L2Balanced(x, beta=1):
    'Balanced between l1 and l2 loss. Beta is the weight for l2. l1 is smoothed around 0.'
    m = torch.abs(x).mean()
    eps = m/10
    l1 = torch.sqrt(x**2+eps**2)-eps
    l2 = 1/2*x**2
    return l1+beta*l2

l1l2bal = lambda x, y: L1L2Balanced(x-y)


def doubleleaky(x, alpha):
    x = torch.nn.functional.leaky_relu(x, alpha)
    x = 1.-x
    x = torch.nn.functional.leaky_relu(x, alpha)
    x = 1.-x
    return x


def torch_randomcrop(y1, h, w=None):
    'Crops a single image (not a batch)'
    if isinstance(w, type(None)):
        w = h
    if h<=1:
        h = int(h*y1.shape[1])
        w = int(w*y1.shape[2])
    dy = torch.randint(y1.shape[1]-h, size=(1,))
    dx = torch.randint(y1.shape[2]-w, size=(1,))
    y1 = y1[:, dy:dy+h, dx:dx+w]
    return y1

def torch_randomcrop_batch(y1, y2, h, w=None):
    if isinstance(w, type(None)):
        w = h
    dy = torch.randint(y1.shape[2]-h, size=(1,))
    dx = torch.randint(y1.shape[3]-w, size=(1,))
    y1 = y1[:, :, dy:dy+h, dx:dx+w]
    y2 = y2[:, :, dy:dy+h, dx:dx+w]
    return y1, y2



def update_mt(mt, n, tau):
    # updates the mean teacher by the network
    mtdict = mt.state_dict()
    ndict = n.state_dict()
    for k in mtdict.keys():
        mtdict[k] = tau * mtdict[k] + (1-tau) * ndict[k]
    mt.load_state_dict(mtdict)
    

    
class AttentionLayer(nn.Module):
    def __init__(self, n, m):
        super().__init__()
        
        self.qkv = nn.Conv1d(n, 2*m + n, 1) # all in one layer: q and k have n//8 channels, v has n channels
        self.nm = (n,m)
        
        ##self.postlayer = nn.Conv2d(n, n, 1)
        
        self.gamma = nn.Parameter(torch.tensor([1.])) #### instead of 0.0
    
    def forward(self, x):
        x0 = x
        
        b, c, h, w = x.shape
        n, m = self.nm
        
        x = relu(x)
        
        qkv = self.qkv(x.view(b, c, h*w))
        #q, k, v = qkv[:,:c//2], qkv[:,c//2:2*c//2], qkv[:,2*c//2:]
        q, k, v = torch.split(qkv, [m, m, n], dim=1)
        beta = torch.bmm(q.permute(0,2,1), k) # has dimensions b, h*w, h*w
        beta = beta / np.sqrt(m)
        
        beta = F.softmax(beta, dim=1)
        self.last_beta = beta
        
        o = torch.bmm(v, beta)
        o = o.reshape(b, c, h, w)
        ##o = self.postlayer(o)
        
        return x0 + self.gamma * o