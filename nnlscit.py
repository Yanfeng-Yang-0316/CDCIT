# this code is from https://github.com/LeeShuai-kenwitch/NNLSCIT, with some modifications
# we mainly use NNCMI() in this file to compute CMI
from scipy import special, spatial
import numpy as np
import pandas as pd
import random
import torch
import math
import importlib
from datetime import datetime
from scipy.stats import norm
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.neighbors import NearestNeighbors
import warnings



eps = 1e-8

def split_XYZ(data, dx, dy):
    X = data[:, 0:dx]
    Y = data[:, dx:dx+dy]
    Z = data[:, dx+dy:]
    return X, Y, Z

def split_train_test(data):
    total_size = data.shape[0]
    train_size = int(2 * total_size / 3)   
    data_train = data[0:train_size, :]
    data_test = data[train_size:, :]
    return data_train, data_test

def normalize_data(data):
    data_norm = (data - np.mean(data, axis = 0))/(np.std(data, axis = 0))
    return data_norm


def gen_bootstrap(data):

    np.random.seed()
    random.seed()
    num_samp = data.shape[0]
    I = np.random.permutation(num_samp)
    data_new = data[I, :]
    return data_new


def mimic_knn(data_mimic, dx, dy, dz, Z_marginal):

    X_train, Y_train, Z_train  = split_XYZ(data_mimic, dx, dy)
    # print(Z_train)
    nbrs = NearestNeighbors(n_neighbors=1,algorithm='ball_tree').fit(Z_train)
    indx = nbrs.kneighbors(Z_marginal, return_distance=False).flatten()
    X_marginal = X_train[indx, :]
    return X_marginal


def shuffle_y(data, dx):
    X = data[:,0:dx]
    Y = data[:,dx:]
    Y = np.random.permutation(Y)
    return np.hstack((X, Y))


def log_mean_exp_numpy(fx_q, ax = 0):
    eps = 1e-8
    max_ele = np.max(fx_q, axis=ax, keepdims = True)
    return (max_ele + np.log(eps + np.mean(np.exp(fx_q-max_ele), axis = ax, keepdims=True))).squeeze()

    




def xgb_classifier(joint_train_data, joint_test_data, marginal_train_data, marginal_test_data):
    
    data_train_feature = np.vstack((joint_train_data, marginal_train_data))
    data_train_label = np.vstack((np.ones((len(joint_train_data), 1)), np.zeros((len(marginal_train_data), 1))))
    data_index = np.random.permutation(2*len(joint_train_data))
    data_train_feature = data_train_feature[data_index]
    data_train_label = data_train_label[data_index]

    data_test_feature = np.vstack((joint_test_data, marginal_test_data))
    data_test_label = np.vstack((np.ones((len(joint_test_data), 1)), np.zeros((len(marginal_test_data), 1))))
    data_test_index = np.random.permutation(2*len(joint_test_data))
    data_test_feature = data_test_feature[data_test_index]
    data_test_label = data_test_label[data_test_index]

    model = xgb.XGBClassifier(    
        #nthread=8,         
        learning_rate=0.01,
        n_estimators=100,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=1,
        colsample_bytree=1,
        objective="binary:logistic",
        scale_pos_weight=1,
        seed=11,
        eval_metric="error",
    )
    
    gbm = model.fit(data_train_feature, data_train_label)

    y_pred_pos_prob = gbm.predict_proba(joint_test_data) 
    y_pred_neg_prob = gbm.predict_proba(marginal_test_data) 

    rn_est_p = (y_pred_pos_prob[:, 1]+eps)/(1-y_pred_pos_prob[:, 1]-eps)
    finp_p = np.log(np.abs(rn_est_p))
    rn_est_q = (y_pred_neg_prob[:, 1] + eps) / (1 - y_pred_neg_prob[:, 1] - eps)
    finp_q = np.log(np.abs(rn_est_q))

    div_est = np.mean(finp_p) - log_mean_exp_numpy(finp_q)

    return div_est




def NNCMI(x, y, z, x_dim, y_dim, z_dim, classifier='xgb', normalize=False):
#     print(x.shape, y.shape, z.shape)
    data = np.hstack((x, y, z))
  
    if normalize:
        data = normalize_data(data)
    
    mimic_size = int(len(data)/2)
    data_mimic = data[0:mimic_size,:]    
    data_mine = data[mimic_size:,:]      
    X, Y, Z = split_XYZ(data_mine, x_dim, y_dim)   

    # print(Z)
    X_marginal = mimic_knn(data_mimic, x_dim, y_dim, z_dim, Z)   
    data_marginal = np.hstack((X_marginal, Y, Z))   
    
    data_train_joint, data_eval_joint = split_train_test(data_mine)   
    data_train_marginal, data_eval_marginal = split_train_test(data_marginal)  
    
    if classifier == 'xgb':
        cmi_est_t = xgb_classifier(data_train_joint, data_eval_joint, data_train_marginal, data_eval_marginal)
    elif classifier == 'lgb':
        cmi_est_t = lgb_classifier(data_train_joint, data_eval_joint, data_train_marginal, data_eval_marginal)
    else:
        tf.compat.v1.reset_default_graph()
        class_mlp_mi_xyz = Classifier_MI(data_train_joint, data_eval_joint, data_train_marginal, data_eval_marginal, x_dim)
        div_xyz_t = class_mlp_mi_xyz.train_classifier_MLP()
        cmi_est_t = div_xyz_t
    
    return cmi_est_t





def nnls_null_distribution(array, xyz, value, shuffle_neighbors=5, sig_samples=1000):
    
    dim, T = array.shape   

    x_indices = np.where(xyz == 0)[0]   
    y_indices = np.where(xyz == 1)[0]   
    z_indices = np.where(xyz == 2)[0]   
        
    seed = 42
    random_state = np.random.default_rng(seed)
    if len(z_indices) > 0 and shuffle_neighbors < T:
        
        z_array = np.fastCopyAndTranspose(array[z_indices, :])  
        tree_xyz = spatial.cKDTree(z_array)   
        neighbors = tree_xyz.query(z_array,
                                   k=shuffle_neighbors,
                                   p=2,
                                   eps=0.)[1].astype(np.int32)
        

        null_dist = np.zeros(sig_samples)   
        for sam in range(sig_samples):   
            for i in range(len(neighbors)):
                random_state.shuffle(neighbors[i])   
            #print('After randomly shuffling the k-nearest neighbor coordinates of zi, the neighbors are:')
            #print(neighbors)
            
            use_permutation = []
            for i in range(len(neighbors)):   
                use_permutation.append(neighbors[i, 0])  
            
            array_shuffled = np.copy(array)   
            for i in x_indices:    # y_indices = [1]
                array_shuffled[i] = array[i, use_permutation]   

            need_data = array_shuffled.T 
            x0, y0, z0 = split_XYZ(need_data, dx=1, dy=1)
            x0_dim = x0.shape[1]
            y0_dim = y0.shape[1]
            z0_dim = z0.shape[1]
            null_dist[sam] = NNCMI(x0, y0, z0, x0_dim, y0_dim, z0_dim, classifier='xgb', normalize=False)
                
    #print('Bth cmi results:')
    #print(null_dist)

    pval = (null_dist >= value).mean()

    return pval




def lpcmicit(x, y, z):

    x_dim = x.shape[1]
    y_dim = y.shape[1]
    z_dim = z.shape[1]
    real_cmi_value = NNCMI(x, y, z, x_dim, y_dim, z_dim, classifier='xgb', normalize=False)
    #print(real_cmi_value)
    
    real_data = np.hstack((x, y, z))
    data = real_data.T
    xyz0 = np.array([0, 1]+[2 for i in range(z_dim)])  
    
    # we always set sig_samples=200 and shuffle_neighbors=7
    p_value = nnls_null_distribution(array=data, xyz=xyz0, value=real_cmi_value, shuffle_neighbors=7, sig_samples=200)   
    
    return p_value