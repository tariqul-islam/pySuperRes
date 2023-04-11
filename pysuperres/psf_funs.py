import numpy as np
#import numba 
#from numba import prange

import warnings


def sinc(x, delay = 0, scale = 1.0):
    z = scale*(x-delay)
    idx = z!=0
    y = np.ones(x.shape)
    y[idx] = np.sin(z[idx])/z[idx]
    
    return y
    
    
def dsinc(x, delay = 0, scale = 1.0):
    z = scale*(x-delay)
    idx = z!=0
    y = np.zeros(x.shape)
    y[idx] = np.cos(z[idx])/z[idx] - np.sin(z[idx])/z[idx]**2
    
    return y

def ddsincsq(x, delay=0, scale = 1.0):

    z = scale*(x-delay)

    idx = np.abs(z)>10**-8
    y = np.ones(x.shape)
    y[idx] = np.sin(z[idx])/z[idx]

    dy = np.zeros(x.shape)
    dy[idx] = np.cos(z[idx])/z[idx] - np.sin(z[idx])/z[idx]**2

    ddy = np.full(x.shape, -1.0/3.0)
    ddy[idx] = - np.sin(z[idx])/z[idx] - 2 * np.cos(z[idx]) / z[idx]**2 + 2 * np.sin(z[idx])/z[idx]**3
    
    y_x = 2 * ( dy**2 + y * ddy )
    
    return y_x
    
    
def sincsq_fit(x,data):
    delay = 0
    sigma = 1
    b = 0.01
    N = x.shape
    
    for i in range(1000):
        z = (x-delay)/sigma
        idx = z!=0
        sinc = np.ones(N)
        sinc[idx] = np.sin(z[idx])/z[idx]
        sincsq = sinc * sinc
        
        dsinc = np.zeros(N)
        dsinc[idx] = np.cos(z[idx])/z[idx] - np.sin(z[idx])/z[idx]**2
        
        dsigma = dsinc * z * -1/sigma
        dL = 2 * (sincsq - data) * dsigma
        sigma = sigma - 0.001 * np.sum(dL)
        
    return sigma
        


def sincsq_1d(x,param2opt,psf_param=None,compute_grad=True,variables_to_opt=[1,1,1]):
    delay = param2opt[0] #param['delay']
    amp = param2opt[1] #param['amp']
    scale = param2opt[2] #param['scale']
    if psf_param is not None:
        sigma = psf_param
    else:
        sigma = 1.0
    z_del = (x-delay)/sigma
    z = scale*z_del

    idx = z!=0
    sinc = np.ones(x.shape)
    sinc[idx] = np.sin(z[idx])/z[idx]

    sincsq = sinc * sinc
    intensity = amp * sincsq
    
    if compute_grad:
        dsinc = np.zeros(x.shape)
        dsinc[idx] = np.cos(z[idx])/z[idx] - np.sin(z[idx])/z[idx]**2
        
        dsincsq = 2 * amp * sinc * dsinc

        grad = np.zeros( ( 3,len(x) ) )
        if variables_to_opt[0]:  #delay gradient  
            grad[0,:] = - scale * dsincsq / sigma
        else:
            grad[0,:] = 0

        if variables_to_opt[1]: #amplitude Graident
            grad[1,:] = sincsq
        else:
            grad[1,:] = 0

        if variables_to_opt[2]: #Scale Gradient
            grad[2,:] = dsincsq * z_del
        else:
            grad[2,:] = 0

    else:
        grad = []    
    

    return intensity, grad

def sincsq_pi_1d(x,param2opt,psf_param=None,compute_grad=True,variables_to_opt=[1,1,1]):
    delay = param2opt[0] #param['delay']
    amp = param2opt[1] #param['amp']
    scale = param2opt[2] #param['scale']
    if psf_param is not None:
        scale_t = psf_param[0]
    else:
        scale_t = p.pi

    z_del = scale_t*(x-delay)
    z = scale*z_del

    idx = z!=0
    sinc = np.ones(x.shape)
    sinc[idx] = np.sin(z[idx])/z[idx]

    sincsq = sinc * sinc
    intensity = amp * sincsq
    
    if compute_grad:
        dsinc = np.zeros(x.shape)
        dsinc[idx] = np.cos(z[idx])/z[idx] - np.sin(z[idx])/z[idx]**2
        
        dsincsq = 2 * amp * sinc * dsinc

        grad = np.zeros( ( 3,len(x) ) )
        if variables_to_opt[0]:  #delay gradient  
            grad[0,:] = - scale * scale_t * dsincsq
        else:
            grad[0,:] = 0

        if variables_to_opt[1]: #amplitude Graident
            grad[1,:] = sincsq
        else:
            grad[1,:] = 0

        if variables_to_opt[2]: #Scale Gradient
            grad[2,:] = dsincsq * z_del
        else:
            grad[2,:] = 0

    else:
        grad = []    
    

    return intensity, grad    
    
def fourier_1d(x,param2opt,psf_param,compute_grad=True,variables_to_opt=[1,1,1]):
    delay = param2opt[0] #delay
    amp = param2opt[1] #amplitude
    scale = param2opt[2] #scale
    
    z_del = x-delay
    z = scale*z_del
    
    A = psf_param[0]
    phi = psf_param[1]
    w0 = psf_param[2]
    
    intensity = np.zeros( len(x) )
    grad = np.zeros( ( 3,len(x) ) )
    
    for i in range(len(A)):
        ZZ = w0*i*z+phi[i]
        cos_part = A[i] * np.cos(ZZ)
        sin_part = A[i] * np.sin(ZZ) * w0 * i * amp
        
        In = amp * cos_part
        
        intensity += In
        
        if compute_grad:
            gr = np.zeros( ( 3,len(x) ) )
            
            if variables_to_opt[0]: #delay gradient
                gr[0:,] = scale * sin_part
            else:
                gr[0,:] = 0
            
            if variables_to_opt[1]: #amplitude gradient
                gr[1,:] = cos_part
            else:
                gr[1,:] = 0
            
            if variables_to_opt[2]: #scale gradient
                gr[2,:] = - z_del * sin_part
            else:
                gr[2,:] = 0
                
            grad += gr

    return intensity, grad

def expsq_1d(x,param2opt,psf_param=None,compute_grad=True,variables_to_opt=[1,1,1]):
    delay = param2opt[0] #param['delay']
    amp = param2opt[1] #param['amp']
    scale = param2opt[2] #param['scale']
    variance = psf_param[0]

    z_del = x-delay
    z = scale*z_del

    exp_fn = np.exp(-2 * z**2 / variance)

    intensity = amp * exp_fn
    
    if compute_grad:

        grad = np.zeros( ( 3,len(x) ) )
        if variables_to_opt[0]:  #delay gradient
            grad[0,:] = 4 * amp * scale / variance * z * exp_fn
        else:
            grad[0,:] = 0

        if variables_to_opt[1]: #amplitude Graident
            grad[1,:] = exp_fn
        else:
            grad[1,:] = 0

        if variables_to_opt[2]: #Scale Gradient
            grad[2,:] = - 4 * amp * z**2 / (scale * variance) * exp_fn
        else:
            grad[2,:] = 0

    else:
        grad = []

    return intensity, grad

def create_x_mat(x,n_coeff):
    n_len = len(x)

    x_mat = np.ones( ( n_len, n_coeff ) )
    
    for i in range(1,n_len):
        x_mat[i,:] = x_mat[i-1,:] * x
    
    return x_mat

def polynomial_1d(x,param2opt,psf_param=None,compute_grad=True,variables_to_opt=[1,1,1]):
    delay = param2opt[0] #param['delay']
    amp = param2opt[1] #param['amp']
    scale = np.array(param2opt[2]) #param['scale']
    coeff = psf_param#['coeff']
    #x_mat = psf_param['x_mat']

    z_del = x-delay
    z = scale*z_del

    z_del_mat = create_x_mat( z_del, len(coeff) )
    z_scale_mat = create_x_mat( scale, len(coeff) )
    z_is = psf_param['idx']

    array = coeff.dot(z_scale_mat * z_del_mat)
    intensity = amp * array
    
    if compute_grad:

        grad = np.zeros( ( 3,len(x) ) )
        if variables_to_opt[0]:  #delay gradient
            grad[0,:] = - amp * coeff[:,1:,].dot( z_is * z_scale_mat[1:,:] * z_del_mat[:-1, :] )
        else:
            grad[0,:] = 0

        if variables_to_opt[1]: #amplitude Graident
            grad[1,:] = array
        else:
            grad[1,:] = 0

        if variables_to_opt[2]: #Scale Gradient
            grad[2,:] = amp * coeff[:,1:].dot( z_is * z_scale_mat[:-1, :] * z_del_mat[1:, :] )
        else:
            grad[2,:] = 0

    else:
        grad = []

    return intensity, grad



def sinc2d(x, delay = [0,0], scale = 1.0):
    X = x[0]
    Y = x[1]
    delay_x = delay[0]
    delay_y = delay[1]
    
    Z_X = scale * (X-delay_x)
    idx_x = Z_X!=0
    Z_Y = scale * (Y-delay_y)
    idx_y = Z_Y!=0
    
    f1 = np.ones(X.shape) 
    f1[idx_x] = np.sin(Z_X[idx_x])/Z_X[idx_x]
    f2 = np.ones(X.shape) 
    f2[idx_x] = np.sin(Z_Y[idx_y])/Z_Y[idx_y]
    
    f = f1*f2
    
    return f

def sincsq_2d(x,param2opt,psf_param=None,compute_grad=True,variables_to_opt=[1,1,1]):
    delay_x = param2opt[0] #param['delay']
    delay_y = param2opt[1]
    amp = param2opt[2] #param['amp']
    scale = param2opt[3] #param['scale']x
    X = x[0]
    Y = x[1]

    Z_X = scale * (X-delay_x)
    idx_x = Z_X!=0
    Z_Y = scale * (Y-delay_y)
    idx_y = Z_Y!=0
    
    fx = np.ones(X.shape) 
    fx[idx_x] = np.sin(Z_X[idx_x])/Z_X[idx_x]
    fy = np.ones(X.shape) 
    fy[idx_x] = np.sin(Z_Y[idx_y])/Z_Y[idx_y]

    fxsq = fx * fx
    fysq = fy * fy

    sincsq = fxsq * fysq
    intensity = amp * sincsq
    
    if compute_grad:
        dfx = np.zeros(X.shape)
        dfx[idx_x] = np.cos(Z_X[idx_x])/Z_X[idx_x] - np.sin(Z_X[idx_x])/Z_X[idx_x]**2
        
        dfy = np.zeros(Y.shape)
        dfy[idx_y] = np.cos(Z_Y[idx_x])/Z_Y[idx_y] - np.sin(Z_Y[idx_x])/Z_Y[idx_x]**2
        
        dfxsq = 2 * fx * dfx
        dfysq = 2 * fy * dfy

        grad = np.zeros( ( 4, X.shape[0], X.shape[1] ) )
        if variables_to_opt[0]:  #delay gradient  
            grad[0,:] = - scale * amp * dfxsq * fysq
            grad[1,:] = - scale * amp * fxsq * dfysq

        if variables_to_opt[1]: #amplitude Graident
            grad[2,:] = sincsq

        if variables_to_opt[2]: #Scale Gradient
            grad[3,:] = amp * (dfxsq * Z_X * fysq + fxsq * dfysq * Z_Y)

    else:
        grad = []    
    

    return intensity, grad


def select_psf_1d(psf_name,psf_param):

    if psf_name == 'sinc':
        ftype = 'Sinc Intensity Function'
        psf_fn = sincsq_1d
        psf_param2 = {}
    elif psf_name == 'exp':
        ftype = "Esponential/Gaussian Intensity Function"
        psf_fn = expsq_1d
        psf_param2 = {}
        psf_param2['variance'] = psf_param
    elif psf_name == 'polynomial':
        ftype = 'Polynomial Intensity Function'
        psf_fn = polynomial_1d
        psf_param2 = {}
        psf_param2['coeff'] = np.array(psf_param).reshape(1,-1)
        psf_param2['idx'] = np.arange(1,len(psf_param)+1).reshape(-1,1)
        #psf_param2['x_mat'] = creat_x_mat(x,len(psf_param))
    else:
        psf_fn = None

    return psf_fn, psf_param2, ftype


def select_psf_2d(psf_name,psf_param):

    if psf_name == 'sinc':
        ftype = 'Sinc Intensity Function'
        psf_fn = sincsq_2d
        psf_param2 = {}
    elif psf_name == 'exp':
        ftype = "Esponential/Gaussian Intensity Function"
        psf_fn = None #to be implemented
        psf_param2 = {}
        psf_param2['variance'] = psf_param
    elif psf_name == 'polynomial':
        ftype = 'Polynomial Intensity Function'
        psf_fn = None #to be implemented
        psf_param2 = {}
        psf_param2['coeff'] = np.array(psf_param).reshape(1,-1)
        psf_param2['idx'] = np.arange(1,len(psf_param)+1).reshape(-1,1)
        #psf_param2['x_mat'] = creat_x_mat(x,len(psf_param))
    else:
        psf_fn = None

    return psf_fn, psf_param2, ftype
