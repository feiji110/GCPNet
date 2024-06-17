#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@File    :   baseModule.py
@Time    :   2023/03/03 11:43:29
@Author  :   Hengda.Gao
@Contact :   ghd@nudt.edu.com
'''
import warnings
from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F


from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn.conv  import MessagePassing
from torch_geometric.utils    import softmax as tg_softmax



class GCAO(MessagePassing):
    def __init__(self, dim, act='silu', batch_norm='False', batch_track_stats='False', dropout_rate=0.0, fc_layers=2, **kwargs):
        super(GCAO, self).__init__(aggr='add',flow='target_to_source', **kwargs)

        self.act          = act
        self.fc_layers    = fc_layers
        if batch_track_stats      == "False":
            self.batch_track_stats = False 
        else:
            self.batch_track_stats = True 

        self.batch_norm   = batch_norm
        self.dropout_rate = dropout_rate
 
        #####################  Graph Attention
        self.heads             = 4
        self.add_bias          = True
        self.neg_slope         = 0.2

        self.bn_node           = nn.BatchNorm1d(self.heads)
        self.W                 = Parameter(torch.Tensor(dim*2,self.heads*dim))
        self.att               = Parameter(torch.Tensor(1,self.heads,2*dim))
        self.dim               = dim

        if self.add_bias  : 
            self.bias = Parameter(torch.Tensor(dim))
        else              : 
            self.register_parameter('bias', None)
        ###################

        ###### Graph Convolution
        channels = dim # input vertex_dim
        if isinstance(channels, int):
            channels = (channels, channels) 
        # node * 2 + edge -> edge, since the embedding layer had map them into the same dimensional hidden_features, dim*3 -> dim is also OK
        self.lin_f = nn.Linear(sum(channels) + dim, dim, bias=True)
        self.lin_s = nn.Linear(sum(channels) + dim, dim, bias=True)
        if self.batch_norm:
            self.bn_edge = nn.BatchNorm1d(dim)    
        else:
            self.bn_edge = None

        ######
        self.reset_parameters() 
        # FIXED-lines -------------------------------------------------------------

    def reset_parameters(self):
        super().reset_parameters()
        glorot(self.W)
        glorot(self.att)
        zeros(self.bias)
        
        self.lin_f.reset_parameters()
        self.lin_s.reset_parameters()
        if self.batch_norm is not None:
            self.bn_edge.reset_parameters()
            self.bn_node.reset_parameters()


    def forward(self, node_feats, edge_index, edge_attr):
        node_feat = node_feats + self.propagate(edge_index, x=node_feats, edge_attr=edge_attr) # node attention
        edge_feat = edge_attr + self.update_edge(edge_index, node_feat, edge_attr) # edge convolution
        return node_feat, edge_feat

    def message(self, edge_index_i, x_i, x_j, edge_attr):
        '''
        node attention message
        '''
        node_i   = torch.cat([x_i,edge_attr],dim=-1)
        node_j   = torch.cat([x_j,edge_attr],dim=-1)
        
        node_i   = getattr(F, self.act)(torch.matmul(node_i,self.W))
        node_j   = getattr(F, self.act)(torch.matmul(node_j,self.W))
        node_i   = node_i.view(-1, self.heads, self.dim)
        node_j   = node_j.view(-1, self.heads, self.dim)

        alpha   = getattr(F, self.act)((torch.cat([node_i, node_j], dim=-1)*self.att).sum(dim=-1))
        alpha   = getattr(F, self.act)(self.bn_node(alpha))
        alpha   = tg_softmax(alpha,edge_index_i)

        alpha   = F.dropout(alpha, p=self.dropout_rate, training=self.training)
        node_j     = (node_j * alpha.view(-1, self.heads, 1)).transpose(0,1)

        return node_j

    def update(self, aggr_out):
        node = aggr_out.mean(dim=0)
        if self.bias is not None:  node = node + self.bias
        return node
    def update_edge(self, edge_index, x, edge_attr):
        '''edge convolution'''
        node_i        = x[edge_index[0]] # 节点i
        node_j        = x[edge_index[1]] # 节点j
        z   = torch.cat([node_i,node_j, edge_attr], dim=-1)
        message = self.lin_f(z).sigmoid() * F.softplus(self.lin_s(z))
        message = message if self.batch_norm is None else self.bn_edge(message)
        return message
    

class BaseModule(nn.Module, metaclass=ABCMeta):
    def __init__(self) -> None:
        super(BaseModule, self).__init__()



    @property
    @abstractmethod
    def target_attr(self):
        """Specifies the target attribute property for writing output to file"""
    
    @abstractmethod
    def forward(self):
        """The forward method for the model."""


    def __str__(self):
        # Prints model summary
        str_representation = "\n"
        model_params_list = list(self.named_parameters())
        separator = (
            "--------------------------------------------------------------------------"
        )
        str_representation += separator + "\n"
        line_new = "{:>30}  {:>20} {:>20}".format(
            "Layer.Parameter", "Param Tensor Shape", "Param #"
        )
        str_representation += line_new + "\n" + separator + "\n"
        for elem in model_params_list:
            p_name = elem[0]
            p_shape = list(elem[1].size())
            p_count = torch.tensor(elem[1].size()).prod().item()
            line_new = "{:>30}  {:>20} {:>20}".format(
                p_name, str(p_shape), str(p_count)
            )
            str_representation += line_new + "\n"
        str_representation += separator + "\n"
        total_params = sum([param.nelement() for param in self.parameters()])
        str_representation += f"Total params: {total_params}" + "\n"
        num_trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        str_representation += f"Trainable params: {num_trainable_params}" + "\n"
        str_representation += (
            f"Non-trainable params: {total_params - num_trainable_params}"
        )

        return str_representation
    
    def total_params(self):
        total_params = sum([param.nelement() for param in self.parameters()])
        return total_params
    

