# CDCIT
This is the source code of [**Conditional Diffusion Models Based Conditional Independence Testing** (CDCIT)](https://arxiv.org/abs/2412.11744) acceepted by AAAI 2025. Our algorithm can test the conditional independence between random variable $X$, $Y$ given $Z$, i.e.:  

$H_0: X ⫫ Y|Z  \text{ \quad v.s. \quad }  H_1:X \not ⫫ Y|Z$  

which is an important problem in statistics, machine learning and causal structure learning. The random variables $X, Y, Z$ can represent gene expression levels, a characterization of a particular disease, or clinical information. And they may also be high-dimensional random variables. We utilize [Diffusion Models](https://arxiv.org/abs/2011.13456) and the [Conditional Randomization Test (CRT)](https://arxiv.org/abs/2304.04183) to test the conditional independence relationships.



## Usage
There is detailed description of function **perform_diffusion_crt()** in diffusion_crt.py. Show_case.ipynb also provides the usage.


## Update
2024.4.19, highdim_XY.ipynb is updated to this repo. It can used to test high-dimensional $X,Y,Z$. But you need to package it by yourself.
