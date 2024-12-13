Underconstruction...
# CDCIT
This is the source code of **Conditional Diffusion Models Based Conditional Independence Testing** (CDCIT).


## Environment Requirements
numpy==1.23.5  
pandas==2.2.0  
random  
torch==1.13.1+cu117  
math  
datetime  
scipy==1.10.0  
xgboost==1.7.5  
scikit-learn==1.4.0  
warnings  
os  
copy  
matplotlib==3.7.0  
tqdm  



If you want to reproduce our simulation, see simulation_postnonlinear_mixedmodel.ipynb and read the codes in network.py. 
Function run_experiment_fork(...) is for the simulation under post-nonlinear model. 
Function run_experiment_con_and_dis(...) is for the simulation under mixed model. 

If you want to reproduce our real data analysis, see real_data_breast_cancer.ipynb and read the codes in network.py.
If you want to know more about how we pre-process the raw data, go to /real_data_breast_cancer/README.md.

If you want to apply our cit to your own triples (X,Y,Z), please read the description in function perform_diffusion_crt(...) and use it, see diffusion_crt.py.
