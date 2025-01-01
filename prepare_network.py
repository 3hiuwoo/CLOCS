#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 23:14:18 2020

@author: Dani Kiyasseh
"""

import torch.nn as nn
import torch
from torch.nn import functional as F
import numpy as np

#%%
""" Functions in this scripts:
    1) cnn_network_contrastive 
    2) second_cnn_network
"""
    
#%%

c1 = 1 #b/c single time-series
c2 = 4 #4
c3 = 16 #16
c4 = 32 #32
k=7 #kernel size #7 
s=3 #stride #3
#num_classes = 3

class cnn_network_contrastive(nn.Module):
    
    """ CNN for Self-Supervision """
    
    def __init__(self,dropout_type,p1,p2,p3,nencoders=1,embedding_dim=256,trial='',device=''):
        super(cnn_network_contrastive,self).__init__()
        
        self.embedding_dim = embedding_dim
        
        if dropout_type == 'drop1d':
            self.dropout1 = nn.Dropout(p=p1) #0.2 drops pixels following a Bernoulli
            self.dropout2 = nn.Dropout(p=p2) #0.2
            self.dropout3 = nn.Dropout(p=p3)
        elif dropout_type == 'drop2d':
            self.dropout1 = nn.Dropout2d(p=p1) #drops channels following a Bernoulli
            self.dropout2 = nn.Dropout2d(p=p2)
            self.dropout3 = nn.Dropout2d(p=p3)
        
        self.relu = nn.ReLU()
        self.selu = nn.SELU()
        self.maxpool = nn.MaxPool1d(2)
        self.trial = trial
        self.device = device
        
        self.view_modules = nn.ModuleList()
        self.view_linear_modules = nn.ModuleList()
        for n in range(nencoders):
            self.view_modules.append(nn.Sequential(
            nn.Conv1d(c1,c2,k,s),
            nn.BatchNorm1d(c2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            self.dropout1,
            nn.Conv1d(c2,c3,k,s),
            nn.BatchNorm1d(c3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            self.dropout2,
            nn.Conv1d(c3,c4,k,s),
            nn.BatchNorm1d(c4),
            nn.ReLU(),
            nn.MaxPool1d(2),
            self.dropout3
            ))
            self.view_linear_modules.append(nn.Linear(c4*10,self.embedding_dim))
                        
    def forward(self,x):
        """ Forward Pass on Batch of Inputs 
        Args:
            x (torch.Tensor): inputs with N views (BxSxN)
        Outputs:
            h (torch.Tensor): latent embedding for each of the N views (BxHxN)
        """
        batch_size = x.shape[0]
        #nsamples = x.shape[2]
        nviews = x.shape[3]
        latent_embeddings = torch.empty(batch_size,self.embedding_dim,nviews,device=self.device)
        for n in range(nviews):       
            """ Obtain Inputs From Each View """
            h = x[:,:,:,n]
            
            if self.trial == 'CMC':
                h = self.view_modules[n](h) #nencoders = nviews
                h = torch.reshape(h,(h.shape[0],h.shape[1]*h.shape[2]))
                h = self.view_linear_modules[n](h)
            else:
                h = self.view_modules[0](h) #nencoder = 1 (used for all views)
                h = torch.reshape(h,(h.shape[0],h.shape[1]*h.shape[2]))
                h = self.view_linear_modules[0](h)

            latent_embeddings[:,:,n] = h
        
        return latent_embeddings


class CLEncoder:
    def __init__(self, embedding_dim=320):
        super(CLEncoder,self).__init__()
        
        self.embedding_dim = embedding_dim
        self.encoder = TSEncoder(1, embedding_dim)
        
    def forward(self,x):
        """ Forward Pass on Batch of Inputs 
        Args:
            x (torch.Tensor): inputs with N views (BxSxN)
        Outputs:
            h (torch.Tensor): latent embedding for each of the N views (BxHxN)
        """
        batch_size = x.shape[0]
        #nsamples = x.shape[2]
        nviews = x.shape[3]
        latent_embeddings = torch.empty(batch_size,self.embedding_dim,nviews,device=self.device)
        for n in range(nviews):       
            """ Obtain Inputs From Each View """
            h = x[...,n]
            h = h.unsqueeze(-1)
            
            h = self.encoder(h, mask='continuous')
            h = F.max_pool1d(h.transpose(1, 2), kernel_size=h.size(1))
            h = h.squeeze(-1)

            latent_embeddings[:,:,n] = h
        
        return latent_embeddings
        
class second_cnn_network(nn.Module):
    
    def __init__(self,first_model,noutputs,embedding_dim=256):
        super(second_cnn_network,self).__init__()
        self.first_model = first_model
        self.linear = nn.Linear(embedding_dim,noutputs)
        
    def forward(self,x):
        h = self.first_model(x)
        h = h.squeeze() #to get rid of final dimension from torch.empty before
        output = self.linear(h)
        return output
    
    
class SamePadConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1):
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        padding = self.receptive_field // 2
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups
        )
        self.remove = 1 if self.receptive_field % 2 == 0 else 0
        
    def forward(self, x):
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, : -self.remove]
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, final=False):
        super().__init__()
        self.conv1 = SamePadConv(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv2 = SamePadConv(out_channels, out_channels, kernel_size, dilation=dilation)
        self.projector = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels or final else None
    
    def forward(self, x):
        residual = x if self.projector is None else self.projector(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x + residual


class DilatedConvEncoder(nn.Module):
    def __init__(self, in_channels, channels, kernel_size):
        super().__init__()
        self.net = nn.Sequential(*[
            ConvBlock(
                channels[i-1] if i > 0 else in_channels,
                channels[i],
                kernel_size=kernel_size,
                dilation=2**i,
                final=(i == len(channels)-1)
            )
            for i in range(len(channels))
        ])
        
    def forward(self, x):
        return self.net(x)
    
    
class TSEncoder(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dims=64, depth=10, mask_mode='binomial'):
        super().__init__()
        self.input_dims = input_dims  # Ci
        self.output_dims = output_dims  # Co
        self.hidden_dims = hidden_dims  # Ch
        self.mask_mode = mask_mode
        self.input_fc = nn.Linear(input_dims, hidden_dims)
        self.feature_extractor = DilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * depth + [output_dims],  # a list here
            kernel_size=3
        )
        self.repr_dropout = nn.Dropout(p=0.1)
        
    def forward(self, x, mask=None):  # input dimension : B x O x Ci
        x = self.input_fc(x)  # B x O x Ch (hidden_dims)
        
        # generate & apply mask, default is binomial
        if mask is None:
            # mask should only use in training phase
            if self.training:
                mask = self.mask_mode
            else:
                mask = 'all_true'
        
        if mask == 'binomial':
            mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'channel_binomial':
            mask = generate_binomial_mask(x.size(0), x.size(1), x.size(2)).to(x.device)
        elif mask == 'continuous':
            mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'channel_continuous':
            mask = generate_continuous_mask(x.size(0), x.size(1), x.size(2)).to(x.device)
        elif mask == 'all_true':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        elif mask == 'all_false':
            mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
        elif mask == 'mask_last':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False
        else:
            raise ValueError(f'\'{mask}\' is a wrong argument for mask function!')

        # mask &= nan_masK
        # ~ works as operator.invert
        x[~mask] = 0

        # conv encoder
        x = x.transpose(1, 2)  # B x Ch x O
        x = self.repr_dropout(self.feature_extractor(x))  # B x Co x O
        x = x.transpose(1, 2)  # B x O x Co
        
        return x
    
    
def generate_continuous_mask(B, T, C=None, n=5, l=0.1):
    if C:
        res = torch.full((B, T, C), True, dtype=torch.bool)
    else:
        res = torch.full((B, T), True, dtype=torch.bool)
    if isinstance(n, float):
        n = int(n * T)
    n = max(min(n, T // 2), 1)
    
    if isinstance(l, float):
        l = int(l * T)
    l = max(l, 1)
    
    for i in range(B):
        for _ in range(n):
            t = np.random.randint(T-l+1)
            if C:
                # For a continuous timestamps, mask random half channels
                index = np.random.choice(C, int(C/2), replace=False)
                res[i, t:t + l, index] = False
            else:
                # For a continuous timestamps, mask all channels
                res[i, t:t+l] = False
    return res


def generate_binomial_mask(B, T, C=None, p=0.5):
    if C:
        return torch.from_numpy(np.random.binomial(1, p, size=(B, T, C))).to(torch.bool)
    else:
        return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)
