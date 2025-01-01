from __future__ import annotations
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from einops import rearrange, repeat, einsum

import csv

import torch
import random
import numpy as np
import torch.utils.data
import os
import logging
from datetime import datetime
import torch.nn.functional as F
import torch.nn as nn
import math
import os
import time
import copy

import sys

import h5py
import argparse
import configparser
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class ModelArgs:
    vocab_size: int = 1        # Input dimension for embedding
    d_model: int = 64         # Dimension of each model layer
    d_inner: int = 128        # Inner dimension for hidden layers
    n_layer: int = 4          # Number of ResidualBlock layers
    seq_in: int = 50          # Input sequence length
    bias: bool = True         # Whether to use bias in linear layers
    conv_bias: bool = True    # Whether to use bias in convolution layers
    d_conv: int = 3           # Convolution kernel size
    dt_rank: int = 8          # Rank for dynamic time dimension
    d_state: int = 16         # State space dimension


class Mamba(nn.Module):
    def __init__(self, args: ModelArgs,hid):
        """Full Mamba model."""
        super().__init__()
        self.args = ModelArgs

        self.nl=args.n_layer

        self.embedding = nn.Linear(args.vocab_size, args.d_model)
        self.layers = nn.ModuleList([ResidualBlock(args) for _ in range(args.n_layer)])

        self.layers2 = nn.ModuleList([ResidualBlock(args) for _ in range(args.n_layer)])

        #self.layers3 = nn.ModuleList([nn.Sequential(RMSNorm(args.seq_in),AVWGCN(args.seq_in,args.seq_in,2,args.d_model)) for _ in range(args.n_layer)])

        #self.layers3=nn.ModuleList([nn.Sequential(RMSNorm(args.seq_in),AVWGCN(args.seq_in,args.seq_in,2,args.d_model)) for _ in range(args.n_layer)])
        
        #self.layers4=nn.ModuleList([nn.Sequential(RMSNorm(args.seq_in),gconv(args.seq_in,hid,2,10,args.d_model),nn.ReLU(),gconv(hid,args.seq_in,2,10,args.d_model)) for _ in range(args.n_layer)])
      
       

        self.lin=nn.ModuleList([nn.Sequential(nn.LayerNorm(args.seq_in),nn.Linear(args.seq_in,hid),nn.ReLU(),nn.Linear(hid,args.seq_in))]+[nn.Sequential(RMSNorm(args.seq_in),nn.Linear(args.seq_in,hid),nn.ReLU(),nn.Linear(hid,args.seq_in)) for _ in range(args.n_layer-2)]+[nn.Sequential(RMSNorm(args.seq_in),nn.Linear(args.seq_in,hid),nn.ReLU(),nn.Linear(hid,args.seq_in))])
        
        #self.lin2=nn.ModuleList([nn.Sequential(RMSNorm(args.seq_in),nn.Linear(args.seq_in,hid),nn.ReLU(),nn.Linear(hid,args.seq_in))]+[nn.Sequential(RMSNorm(args.seq_in),nn.Linear(args.seq_in,hid),nn.ReLU(),nn.Linear(hid,args.seq_in)) for _ in range(args.n_layer-2)]+[nn.Sequential(RMSNorm(args.seq_in),nn.Linear(args.seq_in,hid),nn.ReLU(),nn.Linear(hid,args.seq_in))])
        
        
        self.norm_f = nn.LayerNorm(args.d_model)

        self.lm_head = nn.Linear(args.d_model, args.vocab_size)


        self.proj=nn.Sequential(nn.Linear(args.seq_in,hid),nn.ReLU(),nn.Linear(hid,args.seq_in))

        self.nnl=nn.LayerNorm(args.vocab_size)

    

    
      
        #self.proj=nn.Linear(2*ModelArgs.vocab_size, ModelArgs.vocab_size)
        #self.lm_head.weight = self.embedding.weight  # Tie output projection to embedding weights.
                                                     # See "Weight Tying" paper


    def forward(self, input_ids):
        """
        Args:
            input_ids (long tensor): shape (b, l)    (See Glossary at top for definitions of b, l, d_in, n...)

        Returns:
            logits: shape (b, l, vocab_size)

        Official Implementation:
            class MambaLMHeadModel, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py#L173

        """
        
        
        x = self.embedding(input_ids)

        x1=x
        x2=x
    

        for i in range(self.nl):
            
            x1 = self.layers[i](x1)
            x2=self.layers2[i](x2.flip([1]))
            
            x=x1+x2.flip([1])+x

            x=self.lin[i](x.permute(0,2,1)).permute(0,2,1)+x

            x1=x
            x2=x
            
            
            

        x = self.norm_f(x)
        
        logits = self.lm_head(x)

        return logits


    @staticmethod
    def from_pretrained(pretrained_model_name: str):
        """Load pretrained weights from HuggingFace into model.

        Args:
            pretrained_model_name: One of
                * 'state-spaces/mamba-2.8b-slimpj'
                * 'state-spaces/mamba-2.8b'
                * 'state-spaces/mamba-1.4b'
                * 'state-spaces/mamba-790m'
                * 'state-spaces/mamba-370m'
                * 'state-spaces/mamba-130m'

        Returns:
            model: Mamba model with weights loaded

        """
        from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
        from transformers.utils.hub import cached_file

        def load_config_hf(model_name):
            resolved_archive_file = cached_file(model_name, CONFIG_NAME,
                                                _raise_exceptions_for_missing_entries=False)
            return json.load(open(resolved_archive_file))


        def load_state_dict_hf(model_name, device=None, dtype=None):
            resolved_archive_file = cached_file(model_name, WEIGHTS_NAME,
                                                _raise_exceptions_for_missing_entries=False)
            return torch.load(resolved_archive_file, weights_only=True, map_location='cpu', mmap=True)

        config_data = load_config_hf(pretrained_model_name)
        args = ModelArgs(
            d_model=config_data['d_model'],
            n_layer=config_data['n_layer'],
            vocab_size=config_data['vocab_size']
        )
        model = Mamba(args)

        state_dict = load_state_dict_hf(pretrained_model_name)
        new_state_dict = {}
        for key in state_dict:
            new_key = key.replace('backbone.', '')
            new_state_dict[new_key] = state_dict[key]
        model.load_state_dict(new_state_dict)

        return model





class ResidualBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        """Simple block wrapping Mamba block with normalization and residual connection."""
        super().__init__()
        self.args = args
        self.mixer = MambaBlock(args)
        self.norm = nn.LayerNorm(args.d_model)


    def forward(self, x):
        """
        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)

        Returns:
            output: shape (b, l, d)

        Official Implementation:
            Block.forward(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L297

            Note: the official repo chains residual blocks that look like
                [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> ...
            where the first Add is a no-op. This is purely for performance reasons as this
            allows them to fuse the Add->Norm.

            We instead implement our blocks as the more familiar, simpler, and numerically equivalent
                [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> ....

        """
        output = self.mixer(self.norm(x)) 

        return output


class gconv(nn.Module):
    def __init__(self, inp, hid,embed,cheb_k,n):
        super(gconv, self).__init__()

        self.node_num=n

        self.inp=inp

        self.cheb_k=cheb_k

        self.adj=nn.Parameter(torch.randn(n,embed), requires_grad=True)
       


        self.weights_pool = nn.Parameter(torch.FloatTensor(embed, cheb_k, inp, hid))
        
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed,hid))
        
    def forward(self, x):
        #x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        #output shape [B, N, C]
        
        ADJ=F.softmax(F.relu(torch.mm(self.adj, self.adj.transpose(0, 1))), dim=1)


        

        
        support_set = [torch.eye(self.node_num).cuda(),ADJ]
    
        
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * ADJ, support_set[-1]) - support_set[-2])
        
        
        supports = torch.stack(support_set, dim=0)
        
        weights = torch.einsum('nd,dkio->nkio', self.adj, self.weights_pool)  #N, cheb_k, dim_in, dim_out
        bias = torch.matmul(self.adj, self.bias_pool)                       #N, dim_out
        x_g = torch.einsum("knm,bmc->bknc", supports, x)      #B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        out_6 = torch.einsum('bnki,nkio->bno', x_g, weights) + bias   #B,N,D_OUT

        return out_6

class AVWGCN(nn.Module):
    def __init__(self, dim_in, hid, cheb_k,n):
        super(AVWGCN, self).__init__()

        self.node_num=n

        self.inp=dim_in

        self.cheb_k = cheb_k
        self.node_embeddings = nn.Parameter(torch.randn(n,dim_in,dim_in), requires_grad=True)

        



        self.weights_pool = nn.Parameter(torch.FloatTensor(cheb_k,n,dim_in, hid))
        
        self.bias_pool = nn.Parameter(torch.FloatTensor(n, hid))
        
    def forward(self, x):
        #x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        #output shape [B, N, C]
        
        supports = F.softmax(F.relu(self.node_embeddings), dim=2)

        I=torch.eye(self.inp).cuda()

        I2=I[None,:,:].repeat(x.size(1),1,1)
        

        support_set = [I2, supports]
        
    
        
        
        supports = torch.stack(support_set, dim=0)
        
                              #N, dim_out
        x_g = torch.einsum("bnc,kncm->bknm", x, supports)      #B, cheb_k, N, dim_in
        #x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        x_gconv = torch.einsum('bknm,knmo->bno', x_g, self.weights_pool) + self.bias_pool     #b, N, dim_out
        return x_gconv



class MambaBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        """A single Mamba block, as described in Figure 3 in Section 3.4 in the Mamba paper [1]."""
        super().__init__()
        self.args = args

        self.embedding = nn.Linear(args.vocab_size, args.d_model)

      


        self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias)

        self.in_proj_r = nn.Linear(args.d_model, args.d_inner, bias=args.bias)

        self.conv1d = nn.Conv1d(
            in_channels=args.d_inner,
            out_channels=args.d_inner,
            bias=args.conv_bias,
            kernel_size=args.d_conv,
            groups=args.d_inner,
            padding=args.d_conv - 1,
        )

        

        # x_proj takes in `x` and outputs the input-specific Δ, B, C
        self.x_proj = nn.Linear(args.d_inner, args.dt_rank + args.d_state * 2, bias=False)

        self.norm_f = RMSNorm(args.d_model)

        self.lm_head = nn.Linear(args.d_model, args.vocab_size,bias=False)

        #self.x_proj_r = nn.Linear(args.d_inner, args.dt_rank + args.d_state, bias=True)

        #self.x_proj = FourierKANLayer(args.d_inner, args.dt_rank + args.d_state * 2, 100)

        # dt_proj projects Δ from dt_rank to d_in
        self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True)
        #self.dt_proj=FourierKANLayer(args.dt_rank, args.d_inner, 100)

        A = repeat(torch.arange(1, args.d_state + 1), 'n -> d n', d=args.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(args.d_inner))
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)


    def forward(self, x):
        """Mamba block forward. This looks the same as Figure 3 in Section 3.4 in the Mamba paper [1].

        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)

        Returns:
            output: shape (b, l, d)

        Official Implementation:
            class Mamba, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L119
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311

        """
        (b, l, d) = x.shape

        #x=self.embedding(x)

        x_and_res = self.in_proj(x)  # shape (b, l, 2 * d_in)
        (x, res) = x_and_res.split(split_size=[self.args.d_inner, self.args.d_inner], dim=-1)

        x = rearrange(x, 'b l d_in -> b d_in l')
    
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, 'b d_in l -> b l d_in')



        x = F.silu(x)

        gate=x*(1-F.sigmoid(res))

        

        y = self.ssm(x)
        y = y * F.silu(res)

        output = self.out_proj(y)

        #o1=self.norm_f(output)

        #o2=self.lm_head(o1)

        return output


    def ssm(self, x):
        """Runs the SSM. See:
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        Args:
            x: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)

        Returns:
            output: shape (b, l, d_in)

        Official Implementation:
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311

        """
        (d_in, n) = self.A_log.shape

        # Compute ∆ A B C D, the state space parameters.
        #     A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
        #     ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
        #                                  and is why Mamba is called **selective** state spaces)

        A = -torch.exp(self.A_log.float())  # shape (d_in, n)
        D = self.D.float()

        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)

        (delta, B, C) = x_dbl.split(split_size=[self.args.dt_rank, n, n], dim=-1)  # delta: (b, l, dt_rank). B, C: (b, l, n)
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)

        y = self.selective_scan(x, delta, A, B, C, D)  # This is similar to run_SSM(A, B, C, u) in The Annotated S4 [2]

        return y


    def selective_scan(self, u, delta, A, B, C, D):
        """Does selective scan algorithm. See:
            - Section 2 State Space Models in the Mamba paper [1]
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        This is the classic discrete state space formula:
            x(t + 1) = Ax(t) + Bu(t)
            y(t)     = Cx(t) + Du(t)
        except B and C (and the step size delta, which is used for discretization) are dependent on the input x(t).

        Args:
            u: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
            delta: shape (b, l, d_in)
            A: shape (d_in, n)
            B: shape (b, l, n)
            C: shape (b, l, n)
            D: shape (d_in,)

        Returns:
            output: shape (b, l, d_in)

        Official Implementation:
            selective_scan_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L86
            Note: I refactored some parts out of `selective_scan_ref` out, so the functionality doesn't match exactly.

        """
        (b, l, d_in) = u.shape
        n = A.shape[1]

        # Discretize continuous parameters (A, B)
        # - A is discretized using zero-order hold (ZOH) discretization (see Section 2 Equation 4 in the Mamba paper [1])
        # - B is discretized using a simplified Euler discretization instead of ZOH. From a discussion with authors:
        #   "A is the more important term and the performance doesn't change much with the simplification on B"
        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')

        # Perform selective scan (see scan_SSM() in The Annotated S4 [2])
        # Note that the below is sequential, while the official implementation does a much faster parallel scan that
        # is additionally hardware-aware (like FlashAttention).
        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
            ys.append(y)
        y = torch.stack(ys, dim=1)  # shape (b, l, d_in)

        y = y + u * D

        return y


class RMSNorm(nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))


    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output