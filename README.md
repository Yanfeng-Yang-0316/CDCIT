# CDCIT
This is the source code of [**Conditional Diffusion Models Based Conditional Independence Testing** (CDCIT)](https://arxiv.org/abs/2412.11744) acceepted by AAAI 2025. Our algorithm can test the conditional independence between random variable $X$, $Y$ given $Z$, i.e.:  

$H_0: X ⫫ Y|Z  \quad \text{  v.s. } \quad  H_1:X \not ⫫ Y|Z$  

which is an important problem in statistics, machine learning and causal structure learning. The random variables $X, Y, Z$ can represent gene expression levels, a characterization of a particular disease, or clinical information. And they may also be high-dimensional random variables. We utilize [Diffusion Models](https://arxiv.org/abs/2011.13456) and the [Conditional Randomization Test (CRT)](https://arxiv.org/abs/2304.04183) to test the conditional independence relationships.



## Usage
There is detailed description of function **perform_diffusion_crt()** in diffusion_crt.py. Show_case.ipynb also provides the usage.


## Acknowledgements
This code is built upon the [NNLSCIT](https://github.com/LeeShuai-kenwitch/NNLSCIT).


## Update
2025.4.19, highdim_XY.ipynb is updated to this repo. It can be used to test high-dimensional $X,Y,Z$. But you need to package it by yourself.

2025.6.30, you can use return_samples = True to check generated samples! More visualization of samples can be found in visualize.ipynb.

🎉🎉🎉🎉🎉🎉**2025.6.30, DDIM sampler is updated to highdim_XY.ipynb, diffusion_crt.py and model.py! Now you can use the faster DDIM sampler for simulation. Originally a CDCIT need 60s, but it only need 10s using DDIM!**

DDIM show case:
```python
total_x,total_y,total_z,_,_=data_gen(n_samples=1000, dim=20, test_type=True, noise='gaussian', seed=114)
xxx=total_x[:500,:]
yyy=total_y[:500,:]
zzz=total_z[:500,:]

xxx_crt=total_x[500:,:]
yyy_crt=total_y[500:,:]
zzz_crt=total_z[500:,:]
p_val=perform_diffusion_crt(xxx, yyy, zzz, xxx_crt, yyy_crt, zzz_crt, 
                            repeat=100, device=torch.device('cuda'), 
                            verbose=False, seed=1919, stat='cmi',sampling_model='ddim') # ← see here

print(p_val)
```
## Reference
```bibtex
@article{Yang_2025_cdcit, 
title={Conditional Diffusion Models Based Conditional Independence Testing}, 
volume={39}, 
url={https://ojs.aaai.org/index.php/AAAI/article/view/34354}, 
DOI={10.1609/aaai.v39i21.34354}, 
number={21}, 
journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
author={Yang, Yanfeng and Li, Shuai and Zhang, Yingjie and Sun, Zhuoran and Shu, Hai and Chen, Ziqi and Zhang, Renming}, 
year={2025}, 
pages={22020-22028} }



