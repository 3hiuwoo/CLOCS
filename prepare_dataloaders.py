#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 23:22:28 2020

@author: Dani Kiyasseh
"""

import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset 
from prepare_dataset import my_dataset_contrastive


""" Functions in this script:
    1) load_initial_data_contrastive
"""
#%%
def load_initial_data_contrastive(basepath_to_data,phases,fraction,inferences,batch_size,modality,acquired_indices,acquired_labels,modalities,dataset_name,input_perturbed=False,perturbation='Gaussian',leads='ii',labelled_fraction=1,unlabelled_fraction=1,downstream_task='contrastive',class_pair='',trial='CMC',nviews=1):    
    """ Control augmentation at beginning of training here """ 
    # resize = False
    # affine = False
    # rotation = False
    # color = False    
    # perform_cutout = False
    # operations = {'resize': resize, 'affine': affine, 'rotation': rotation, 'color': color, 'perform_cutout': perform_cutout}    
    shuffles = {'train1':True,
                'train2':False,
                'val': False,
                'test': False}
    
    # fractions = {'fraction': fraction,
    #              'labelled_fraction': labelled_fraction,
    #              'unlabelled_fraction': unlabelled_fraction}
    
    # acquired_items = {'acquired_indices': acquired_indices,
    #                   'acquired_labels': acquired_labels}
    # dataset = {phase:my_dataset_contrastive(basepath_to_data,dataset_name,phase,inference,fractions,acquired_items,modalities=modalities,task=downstream_task,input_perturbed=input_perturbed,perturbation=perturbation,leads=leads,class_pair=class_pair,trial=trial,nviews=nviews) for phase,inference in zip(phases,inferences)}                                        
    
    X_train, _, _, y_train, _, _ = load_data(dataset_name, trial=trial)
    assert len(phases) == 1
    dataset = {phase: TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)) for phase in phases}
    
    
#    if 'train' in phases:
#        check_dataset_allignment(mixture,dataset_list)
    operations = {}    
    dataloader = {phase:DataLoader(dataset[phase],batch_size=batch_size,shuffle=shuffles[phase],drop_last=True) for phase in phases}
    return dataloader,operations


def load_data(dataset_name, trial='CMSC'):
    if dataset_name == 'chapman':
        X_train, X_val, X_test, y_train, y_val, y_test = load_chapman()
        
        if trial in ['CMSC', 'CMSMLC']:
            X_train = segment(X_train, 900)
            nviews = X_train.shape[1]
            
        if trial == 'CMSMLC':
            X_train = X_train.transpose(0, 3, 1, 2).reshape(X_train.shape[0], -1, 900, 1).squeeze(-1).transpose(0, 2, 1)
            
        elif trial == 'CMSC':
            X_train = X_train.transpose(0, 3, 1, 2).reshape(-1, nviews, 900, 1).squeeze(-1).transpose(0, 2, 1)
            y_train = y_train.repeat(X_train.shape[0]/y_train.shape[0], 0)
            
        else:
            raise ValueError('Trial not supported!')
    else:
        raise ValueError('Dataset not found!')
        
    return X_train, X_val, X_test, y_train, y_val, y_test
    
    
    
    
def load_chapman(root='/root/MCP/chapman', split=None):
    data_path = os.path.join(root, 'feature')
    label_path = os.path.join(root, 'label', 'label.npy')
    
    labels = np.load(label_path)
    
    pids_sb = list(labels[np.where(labels[:, 0]==0)][:, 1])
    pids_af = list(labels[np.where(labels[:, 0]==1)][:, 1])
    pids_gsvt = list(labels[np.where(labels[:, 0]==2)][:, 1])
    pids_sr = list(labels[np.where(labels[:, 0]==3)][:, 1])
    
    train_ids = pids_sb[:-500] + pids_af[:-500] + pids_gsvt[:-500] + pids_sr[:-500]
    valid_ids = pids_sb[-500:-250] + pids_af[-500:-250] + pids_gsvt[-500:-250] + pids_sr[-500:-250]
    test_ids = pids_sb[-250:] + pids_af[-250:] + pids_gsvt[-250:] + pids_sr[-250:]
    
    filenames = []
    for fn in os.listdir(data_path):
        filenames.append(fn)
    filenames.sort()
    
    train_trials = []
    train_labels = []
    valid_trials = []
    valid_labels = []
    test_trials = []
    test_labels = []
    
    for i, fn in enumerate(tqdm(filenames)):
        label = labels[i]
        feature = np.load(os.path.join(data_path, fn))
        for trial in feature:
            if i+1 in train_ids:
                train_trials.append(trial)
                train_labels.append(label)
            elif i+1 in valid_ids:
                valid_trials.append(trial)
                valid_labels.append(label)
            elif i+1 in test_ids:
                test_trials.append(trial)
                test_labels.append(label)
                
    X_train = np.array(train_trials)
    X_val = np.array(valid_trials)
    X_test = np.array(test_trials)
    y_train = np.array(train_labels)
    y_val = np.array(valid_labels)
    y_test = np.array(test_labels)
    
    if split:
        X_train, y_train = segment(X_train, y_train, split)
        X_val, y_val = segment(X_val, y_val, split)
        X_test, y_test = segment(X_test, y_test, split)
        
    return X_train, X_val, X_test, y_train, y_val, y_test


def segment(X, sample):
    '''
    segment the trial to non-overlapping samples
    '''
    length = X.shape[1]
    batch_size = X.shape[0]
    assert length % sample == 0
    nsample = length / sample
    
    samples = X.reshape(batch_size, -1, sample, X.shape[-1])
    return samples


def create_views(X, y, nviews=2, flatten=False):
    '''
    create the views for the contrastive learning
    '''
    length = X.shape[1]
    nchannels = X.shape[2]
    
    X = X.reshape(-1, nviews, length, nchannels)
    y = y.reshape(-1, nviews, y.shape[-1])
    
    if flatten:
        X = X.transpose(3, 0, 1, 2).reshape(-1, nviews, length, 1)
        y = np.tile(y, (nchannels, 1, 1))
        
    return X, y
# %%

