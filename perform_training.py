#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 23:24:46 2020

@author: Dani Kiyasseh
"""

import torch
from tqdm import tqdm
from prepare_miscellaneous import obtain_contrastive_loss, flatten_arrays, calculate_auc, change_labels_type

#%%
""" Functions in this script:
    1) contrastive_single
    2) one_epoch_contrastive
    3) one_epoch_finetuning
    4) finetuning_single
"""
#%%

def contrastive_single(phase,inference,dataloaders,model,optimizer,device,weighted_sampling,epoch_count=None,new_task_epochs=None,trial=None,save_path_dir=None): #b/c it is single, models_list contains one model only
    """ One Epoch's Worth of Training for Contrastive Learning Paradigm """
    running_loss = 0.0
    outputs_list = []
    labels_list = []
    modality_list = []
    indices_list = []
    task_names_list = []
    pids_list = []
    batch_num = 0
    batch = 0
    # for inputs,labels,pids,modality,task_names,indices in tqdm(dataloaders[phase]):
    for inputs, y in tqdm(dataloaders[phase]):
        batch += 1
        """ Send Data to Device """
        inputs = inputs.to(device)
        labels = y[:, 0].to(device)
        pids = y[:, 1].detach().numpy()
        
        with torch.set_grad_enabled('train1' in phase):# and inference == False): #('train' in phase and inference == False)
            outputs = model(inputs) #(BxHx2) in CPPC, (BxHx12) in CMLC, (BxHx24) in CMSMLC

            loss = obtain_contrastive_loss(outputs,pids,trial)
        
        """ Backpropagation and Update Step """
        if phase == 'train1': #only perform backprop for train1 phase           
            loss.backward()
            #for param in model.parameters():
            #    print(param.grad)
            
            """ Network Parameters """
            if isinstance(optimizer,tuple):
                optimizer[0].step()
                """ Task-Instance Parameters """
                optimizer[1].step()                
                optimizer[0].zero_grad()
                optimizer[1].zero_grad()
            else:
                optimizer.step()
                optimizer.zero_grad()
        
        """ Calculate Metrics """
        running_loss += loss.item() * inputs.shape[0]
        if labels.data.dtype != torch.long:
            labels.data = labels.data.type(torch.long)

        outputs_list.append(outputs.cpu().detach().numpy())
        labels_list.append(labels.cpu().detach().numpy())
        modality_list.append('')
        indices_list.append('')
        task_names_list.append('')
        pids_list.append(pids)
        batch_num += 1
    
    outputs_list, labels_list, modality_list, indices_list, task_names_list, pids_list = flatten_arrays(outputs_list,labels_list,modality_list,indices_list,task_names_list,pids_list)
    epoch_loss = running_loss / len(dataloaders[phase].dataset)
    return epoch_loss, outputs_list, labels_list, modality_list, indices_list, task_names_list, pids_list

def one_epoch_contrastive(weighted_sampling,phase,inference,dataloader,model,optimizer,device,bptt_steps=0,epoch_count=None,new_task_epochs=None,trial=None,save_path_dir=None):
    epoch_loss, outputs_list, labels_list, modality_list, indices_list, task_names_list, pids_list = contrastive_single(phase,inference,dataloader,model,optimizer,device,weighted_sampling,epoch_count=epoch_count,new_task_epochs=new_task_epochs,trial=trial,save_path_dir=save_path_dir)
    return {"epoch_loss": epoch_loss}, outputs_list, labels_list, modality_list, indices_list, task_names_list, pids_list

def finetuning_single(phase,inference,dataloaders,model,optimizer,device,weighted_sampling,criterion,classification,epoch_count=None,new_task_epochs=None,trial=None,save_path_dir=None): #b/c it is single, models_list contains one model only
    """ One Epoch's Worth of Training for Contrastive Learning Paradigm """
    running_loss = 0.0
    outputs_list = []
    labels_list = []
    modality_list = []
    indices_list = []
    task_names_list = []
    pids_list = []
    batch_num = 0
    batch = 0
    for inputs,labels,pids,modality,task_names,indices in tqdm(dataloaders[phase]):
        batch += 1
        """ Send Data to Device """
        inputs = inputs.to(device)
        labels = labels.to(device)
        labels = change_labels_type(labels,criterion)
        with torch.set_grad_enabled('train1' in phase):# and inference == False): #('train' in phase and inference == False)
            outputs = model(inputs)
            loss = criterion(outputs,labels)
        
        """ Backpropagation and Update Step """
        if phase == 'train1': #only perform backprop for train1 phase           
            loss.backward()
            
            """ Network Parameters """
            if isinstance(optimizer,tuple):
                optimizer[0].step()
                """ Task-Instance Parameters """
                optimizer[1].step()                
                optimizer[0].zero_grad()
                optimizer[1].zero_grad()
            else:
                optimizer.step()
                optimizer.zero_grad()
        
        """ Calculate Metrics """
        running_loss += loss.item() * inputs.shape[0]
        if labels.data.dtype != torch.long:
            labels.data = labels.data.type(torch.long)

        outputs_list.append(outputs.cpu().detach().numpy())
        labels_list.append(labels.cpu().detach().numpy())
        modality_list.append(modality)
        indices_list.append(indices)
        task_names_list.append(task_names)
        pids_list.append(pids)
        batch_num += 1
    
    outputs_list, labels_list, modality_list, indices_list, task_names_list, pids_list = flatten_arrays(outputs_list,labels_list,modality_list,indices_list,task_names_list,pids_list)
    epoch_loss = running_loss / len(dataloaders[phase].dataset)
    epoch_auroc = calculate_auc(classification,outputs_list,labels_list,save_path_dir)
    return epoch_loss, epoch_auroc, outputs_list, labels_list, modality_list, indices_list, task_names_list, pids_list

def one_epoch_finetuning(weighted_sampling,phase,inference,dataloader,model,optimizer,device,criterion,classification,bptt_steps=0,epoch_count=None,new_task_epochs=None,trial=None,save_path_dir=None):
    epoch_loss, epoch_auroc, outputs_list, labels_list, modality_list, indices_list, task_names_list, pids_list = finetuning_single(phase,inference,dataloader,model,optimizer,device,weighted_sampling,criterion,classification,epoch_count=epoch_count,new_task_epochs=new_task_epochs,trial=trial,save_path_dir=save_path_dir)
    return {"epoch_loss": epoch_loss, 'epoch_auroc': epoch_auroc}, outputs_list, labels_list, modality_list, indices_list, task_names_list, pids_list
