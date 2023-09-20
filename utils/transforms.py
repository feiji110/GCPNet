#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@File    :   transforms.py
@Time    :   2023/05/05 12:41:00
@Author  :   Hengda.Gao
@Contact :   ghd@nudt.eddu.com
'''
import torch
from torch_sparse import coalesce
from utils.helpers import compute_bond_angles


class GetY(object):
    '''Specify target for prediction'''
    def __init__(self, index=0):
        self.index = index

    def __call__(self, data):
        if self.index != -1:
            data.y = data.y[0][self.index]  # data.y: (#crystals, #targets)
        return data


class GetAngle(object):
    '''Computes bond angles in the crystall'''
    def __call__(self, data):
        angles, idx_kj, idx_ji = compute_bond_angles(
            data.pos, data.cell_offsets, data.edge_index, data.num_nodes
        )
        data.angle_index  = torch.stack([idx_kj, idx_ji], dim=0)
        data.angle_attr   = angles.reshape(-1, 1)
        return data



class ToFloat(object):
    '''Converts all features in the crystall pattern graph to float'''
    def __call__(self, data):
        data.x = data.x.float()
        data.edge_attr = data.edge_attr.float()
        data.angle_attr = data.angle_attr.float()
        return data
