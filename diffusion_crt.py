# import
from nnlscit import *
from model import *
from network import *
import torch
import torch.nn as nn
import warnings
from tqdm import tqdm


def perform_diffusion_crt(xxx, yyy, zzz, xxx_crt, yyy_crt, zzz_crt, 
                  repeat=100, device=torch.device('cuda'), 
                  verbose=False, seed=None, stat='cmi',
                  centralize=False,sampling_model='ddpm',return_samples=False):
    '''
    xxx,yyy,zzz: triple used to train diffusion model, learning Y|Z. and when computing CMI, we use 1-nn to learn X|Z. 
    you can see that in line 49,136 in nnlscit.py. in our paper, we wrote using diffusion model to learn X|Z and when computing CMI, we use 1-nn to learn Y|Z. 
    but that doesn't effect the outcome because in terms of R.V., X and Y is changeable. 
    actually, we also tried to use both diffusion model and 1-nn to learn Y|Z (or X|Z), which will have a little bit worse and unstable performance.
    
    the size of xxx, yyy and zzz is [N,1], [N,1] and [N,dz]. the size of xxx_crt, yyy_crt, zzz_crt is [n,1], [n,1], [n,dz]. 
    although X and Y are 1-dimensional tabular data, you can change it to high dimensional unstructural data. 
    note that our statistics CMI can only compute tabular data (maybe high-dimensional). 
    For unstructural data, you can use a VAE to easily convert it into vectors, and use our CMI statistics, or just find some neural maps T(X,Y,Z) to conduct CRT. 
    
    xxx_crt,yyy_crt,zzz_crt: triple used to compute T(X,Y,Z).
    
    for your own dataset (X,Y,Z), you need to randomly split it into 2 parts (xxx,yyy,zzz) and (xxx_crt,yyy_crt,zzz_crt). we use equal sample size.
    
    device: if you have gpu, use device=torch.device('cuda'). if you don't have gpu, use device=torch.device('cpu').
    
    seed: we have a function to fix seed, see seed_everything() in model.py.
    
    repeat: B in the paper, the number of X^(b). repeat should be bigger than 1/alpha. e.g., when alpha=0.05, repeat should >=21.
    
    stat: the defalut is 'cmi': conditional mutual information (CMI), we also provided 'corr': pearson corr, 'rdc': Randomized Dependence Coefficient. 
    you can find more information in model.py. we found that 'cmi' has the best performance on controling type I error. 
    'rdc' is faster, but with a little bit worse performance.
    
    sampling_model: sampling_model='score' means using the training and sampling method in https://arxiv.org/abs/2011.13456. 
    sampling_model='ddpm' means using training and sampling method in https://arxiv.org/abs/2006.11239.
    sampling_model='ddim' means using training method of DDPM, and use sampling method in https://arxiv.org/abs/2010.02502.
    
    return_samples: whether return the last samples.
    
    note that we highly recommend users to use sampling_model='ddpm', because ddpm can provide smoother forward process and reverse process, 
    thus outputing better results. sampling_model='ddpm' is especially better when X,Y,Z are high dimensional. 
    '''

    
    # set seed
    if seed == None:
        np.random.seed()
    else:
        seed_everything(seed)  # note that if the version of python changes, the seeds will be different.
        
    #get dim and sample size
    dz = zzz.shape[1]
    num = zzz.shape[0]
    
    # centralize
    if centralize:
        xxx = (xxx - xxx.mean())/xxx.std()
        yyy = (yyy - yyy.mean())/yyy.std()
        zzz = (zzz - zzz.mean(axis=0))/zzz.std(axis=0)
        xxx_crt = (xxx_crt - xxx_crt.mean())/xxx_crt.std()
        yyy_crt = (yyy_crt - yyy_crt.mean())/yyy_crt.std()
        zzz_crt = (zzz_crt - zzz_crt.mean(axis=0))/zzz_crt.std(axis=0)

    dataset_x = torch.from_numpy(zzz[:int(num), :]).float() # learn dataset_y| dataset_x, i.e. yyy|zzz
    dataset_y = torch.from_numpy(yyy[:int(num), :]).float()


    dataset_x = dataset_x.to(device)
    dataset_y = dataset_y.to(device)

    

    num_steps = 1000
        
    
    if sampling_model=='ddpm' or sampling_model=='ddim':
        betas=make_beta_schedule(schedule="linear", num_timesteps=num_steps,start=1e-4, end=2e-2)
        alphas=1-betas
        alphas_bar=torch.cumprod(alphas,0).to(device)
        alphas_bar_sqrt=torch.sqrt(alphas_bar)
        one_minus_alphas_bar_sqrt=torch.sqrt(1-alphas_bar)
    
    # train conditional diffusion model
    batch_size = 2048 # big batch size means faster training speed
    dataloader = torch.utils.data.DataLoader(torch.cat([dataset_x, dataset_y], dim=1), batch_size=batch_size,
                                             shuffle=True, )
    if sampling_model=='score':
        model = ConditionalGuidedModel(num_steps, dz=dz).to(device)
    elif sampling_model=='ddpm' or sampling_model=='ddim':
        model = DiffusionModelWithEmbedding(input_dim=dataset_y.shape[1], 
                                time_steps=num_steps, embedding_dim=16,
                                cond_dim=dz).to(device)
    if sampling_model=='score':
        lr = 1e-2
    elif sampling_model=='ddpm' or sampling_model=='ddim':
        lr = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    num_epoch = 1500 
    if verbose == True:
        print('training diffusion model')
    ema2 = EMA(model)
    for z in range(num_epoch):
        total_loss = 0
        for idx, batch in enumerate(dataloader):
            model.train()
            batch_x = batch[:, :dz]
            batch_y = batch[:, dz:]
            if sampling_model=='score':
                loss=score_loss_fn(model, batch_y, batch_x, num_steps,device)
            elif sampling_model=='ddpm' or sampling_model=='ddim':
                loss=diffusion_loss_fn(model, batch_y, batch_x,alphas_bar_sqrt, one_minus_alphas_bar_sqrt, num_steps,device)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            ema2.update(model)

    # crt
    if verbose == True:
        print('crting')
    
    # compute T(x,y,z)
    if stat =='cmi':

        original = NNCMI(xxx_crt, yyy_crt, zzz_crt, 1, 1, dz,      # if you want to try high-dim x,y, don't forgot to change here
                         classifier='xgb', normalize=False)
    elif stat =='corr':
        original = correlation(xxx_crt,yyy_crt)
        
    elif stat == 'rdc':
        original = rdc(xxx_crt,yyy_crt)
    count = 0
    for iiiii in range(repeat):
        with torch.no_grad():

            # sample pseudo_y
            if sampling_model=='score':

                y_seq_crt = score_sampler(model,yyy_crt.shape,torch.tensor(zzz_crt).float().to(device), 
                                           device,)
                y_seq_crt=[y_seq_crt]
            elif sampling_model=='ddpm':
                y_seq_crt = sample_from_diff(model, num_samples=zzz_crt.shape[0], 
                                    input_dim=yyy_crt.shape[1],cond=torch.tensor(zzz_crt).to(device).float(),
                                    alphas_bar_sqrt=alphas_bar_sqrt,
                                    one_minus_alphas_bar_sqrt=one_minus_alphas_bar_sqrt,
                                    betas=betas,
                                    num_steps=num_steps,
                                    device=device)
            elif sampling_model=='ddim':
                y_seq_crt = sample_from_ddim(model, 
                                             num_samples=zzz_crt.shape[0], 
                                             input_dim=yyy_crt.shape[1], 
                                             cond=torch.tensor(zzz_crt).to(device).float(), 
                                             alphas_bar_sqrt=alphas_bar_sqrt, 
                                             one_minus_alphas_bar_sqrt=one_minus_alphas_bar_sqrt, 
                                             num_steps=50, eta=0.0, device=device)
                
                
                
        if  stat =='cmi':
            # compute T(x,pseudo_y,z). again, we note that when computing CMI, we use 1-nn to learn x|z, which is equivalent with our paper. see nnlscit.py
            crt_stat = NNCMI(xxx_crt, y_seq_crt[-1].detach().cpu().numpy(),
                             zzz_crt, 1, 1, dz, classifier='xgb',
                             normalize=False)
        elif  stat =='corr':
            crt_stat = correlation(xxx_crt,y_seq_crt[-1].detach().cpu().numpy())
            
        elif  stat =='rdc':    
            crt_stat = rdc(xxx_crt,y_seq_crt[-1].detach().cpu().numpy())
        if crt_stat > original:
            count += 1

    if return_samples:
        return count/repeat, y_seq_crt[-1]
    else:
        return count/repeat


