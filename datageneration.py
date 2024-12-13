# import
import numpy as np
import warnings

funcs = {
    "linear": lambda x: x,
    "square": lambda x: x ** 2,
    "cos": lambda x: np.cos(x),
    "cube": lambda x: x ** 3,
    "tanh": lambda x: np.tanh(x),
}

func_names = ["linear", "square", "cos", "cube", "tanh"]



def data_gen(n_samples, dim, test_type, noise="gaussian", seed=None):
    if seed == None:
        np.random.seed()
    else:
        np.random.seed(seed)

    if noise == "gaussian":
        sampler = np.random.normal
    elif noise == "laplace":
        sampler = np.random.laplace
    elif noise == "uniform":
        sampler = np.random.uniform
    keys = np.random.choice(range(5), 2)
    pnl_funcs = [func_names[k] for k in keys]

    func1 = funcs[pnl_funcs[0]]
    func2 = funcs[pnl_funcs[1]]

    x = 0.25 * sampler(size=(n_samples, 1))
    y = 0.25 * sampler(size=(n_samples, 1))
    z = sampler(size=(n_samples, dim))
    m = np.mean(z, axis=1).reshape(-1, 1)
    x += m
    y += m
    x, y = func1(x), func2(y)

    if test_type:
        return x, y, z, pnl_funcs[0], pnl_funcs[1]
    else:
        eb = 0.5 * sampler(size=(n_samples, 1))
        x += eb
        y += eb
        return x, y, z, pnl_funcs[0], pnl_funcs[1]






def data_continuous_discrete(n_samples, dim, test_type, noise="gaussian", seed=None):
    if seed == None:
        np.random.seed()
    else:
        np.random.seed(seed)

    if noise == "gaussian":
        sampler = np.random.normal
    elif noise == "laplace":
        sampler = np.random.laplace
    elif noise == "uniform":
        sampler = np.random.uniform
        
        
    
    
    if dim ==1:
        z = sampler(size=(n_samples, 1))
        
        
        
        
        if test_type:
            eps1=0.33*sampler(size=(n_samples, 1))
            eps2=0.33*sampler(size=(n_samples, 1))
            x = z+eps1
            y = z+eps2
        else:
            eps1=0.33*sampler(size=(n_samples, 1))
            x = z+eps1
            y = z+eps1

        
            
    else:
        z = np.zeros(shape=(n_samples, dim))
        z[:,0:int(dim/2)]=sampler(size=(n_samples, int(dim/2)))
        z[:,int(dim/2):]=np.random.binomial(1,0.5,z[:,int(dim/2):].shape)*2-1
        
        
        
        
        if test_type:
            eps1=0.33*sampler(size=(n_samples, 1))
            eps2=0.33*sampler(size=(n_samples, 1))

            x=np.mean(z[:,0:int(2*dim/3)], axis=1).reshape(-1, 1)+eps1
            y=np.mean(z[:,0:int(2*dim/3)], axis=1).reshape(-1, 1)+eps2
        else:
            eps1=0.33*sampler(size=(n_samples, 1))
            x=np.mean(z[:,0:int(2*dim/3)], axis=1).reshape(-1, 1)+eps1
            y=np.mean(z[:,0:int(2*dim/3)], axis=1).reshape(-1, 1)+eps1
        


    return x, y, z, 'continuous_discrete', 'continuous_discrete'


