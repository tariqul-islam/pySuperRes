import numpy as np

def matlab_possion_noise(sig, clip='True', seed=None):
    #poission noise model
    #Post of the matlab version of the code
    #usage: input: x
    # x_noisy = 10**12 * matlab_possion_noise(x*10**-12)
    if seed is not None:
        np.random.seed(seed)
    
    sizeS = sig.shape
    sig = sig.copy().reshape(-1)
    sig = sig*10**12
    
    b = np.zeros_like(sig)
    
    idx1 = np.where(sig<50)[0] #for python 3.0
    if len(idx1):
        g = np.exp(-sig[idx1])
        em = -np.ones_like(g)
        t = np.ones_like(g)
        idx2 = np.arange(0,len(idx1))
        
        #print(idx1, idx2)
        
        while len(idx2):
            em[idx2] = em[idx2]+1
            t[idx2] = t[idx2] * np.random.rand(len(idx2))
            idx2 = idx2[t[idx2]>g[idx2]]
        
        b[idx1] = em
        
    idx1 = np.where(sig>=50)[0]
    b[idx1] = np.round( sig[idx1] + np.sqrt(sig[idx1]) * np.random.randn(len(idx1)) )
    
    b = b.reshape(sizeS)
    
    if clip:
        b = np.maximum(0,np.minimum(b*10**-12,1))
    else:
        b = b*10**-12
    
    return b

