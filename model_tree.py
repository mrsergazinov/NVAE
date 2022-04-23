import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import AutoEncoder

class iSHM_block(nn.Module):
    def __init__(self, input_dim, cp_k_max) -> None:
        super().__init__()
        self.fp = nn.Sequential(nn.Linear(np.prod(input_dim), cp_k_max), 
                                    nn.LogSigmoid())

    def forward(self, x) -> torch.Tensor:
        # probability of travelling to the right node
        if x.shape[0] == 0:
            return 0.0
        x = x.view(x.shape[0], -1)
        p = 0.68394 - (self.fp(-x).mean(dim=1) - 1.0).exp()
        return torch.clamp(p, min=1e-4, max=1-1e-4)

class TreeNVAE(nn.Module):
    def __init__(self, depth, cp_k_max, feat_dim, **kwargs):
        super(TreeNVAE, self).__init__()
        self.depth = depth
        self.cp_k_max = cp_k_max
        self.feat_dim = feat_dim

        if self.depth == 1:
            self.child_l = AutoEncoder(**kwargs)
            self.child_r = AutoEncoder(**kwargs)
        else:
            self.child_l = TreeNVAE(depth - 1, cp_k_max, **kwargs)
            self.child_r = TreeNVAE(depth - 1, cp_k_max, **kwargs)
        self.ishm_block = iSHM_block(feat_dim, cp_k_max)
        
    def forward(self, x, p_x_previous=None, **kwargs):
        p_r = self.ishm_block(x) 
        if p_x_previous is None:
            p_x_previous = torch.ones_like(p_r)

        if self.depth == 1:
            out_l = self.child_l(x, p_x_previous * (1.0 - p_r), **kwargs)
            out_r = self.child_r(x, p_x_previous * p_r, **kwargs)
            return out_l[-1] + out_r[-1]
        else:
            return self.child_r(x, p_x_previous * p_r, **kwargs) + self.child_l(x, p_x_previous **kwargs)
