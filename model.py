#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@File    :   gcpnet.py
@Time    :   2023/04/03 09:10:32
@Author  :   Hengda.Gao
@Contact :   ghd@nudt.edu.com
'''

import torch
import torch.nn.functional as F
from torch.nn import Linear,Sequential

from torch_scatter import scatter
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool,DiffGroupNorm

import numpy as np
from typing import Literal, Optional
from baseModule import BaseModule, GCAO


class RBFExpansion(torch.nn.Module):
    r"""
    Expand interatomic distances with radial basis functions.
    Default: RBF Expansion on distances or angles to compute Gaussian distribution of embeddings.
    """

    def __init__(
            self,
            vmin: float = 0,
            vmax: float = 8,
            bins: int = 40,
            lengthscale: Optional[float] = None,
            type: str = "gaussian"
    ):
        """Register torch parameters for RBF expansion."""
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins
        self.centers = torch.linspace(vmin, vmax, bins)
        self.type = type

        if lengthscale is None:
            # SchNet-style: set lengthscales relative to granularity of RBF expansion
            self.lengthscale = torch.diff(self.centers).mean()
            self.gamma = 1.0 / self.lengthscale

        else:
            self.lengthscale = lengthscale
            self.gamma = 1.0 / (lengthscale**2)

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        """Apply RBF expansion to interatomic distance tensor."""
        base = self.gamma * (distance - self.centers.to(distance.device))
        switcher = {
            'gaussian': (-base ** 2).exp(),
            'quadratic': base ** 2,
            'linear': base,
            'inverse_quadratic': 1.0 / (1.0 + base ** 2),
            'multiquadric': (1.0 + base ** 2).sqrt(),
            'inverse_multiquadric': 1.0 / (1.0 + base ** 2).sqrt(),
            'spline': base ** 2 * (base + 1.0).log(),
            'poisson_one': (base - 1.0) * (-base).exp(),
            'poisson_two': (base - 2.0) / 2.0 * base * (-base).exp(),
            'matern32': (1.0 + 3 ** 0.5 * base) * (-3 ** 0.5 * base).exp(),
            'matern52': (1.0 + 5 ** 0.5 * base + 5 / 3 * base ** 2) * (-5 ** 0.5 * base).exp(),
        }
        result = switcher.get(self.type, None)
        if result.any():
            return result
        else:
            raise Exception("No Implemented Radial Basis Method")

class EmbeddingLayer(torch.nn.Module):
    r"""
    Embedding layer which performs nonlinear transform on atom, edge and triplet features
    """

    def __init__(self, input_features, output_features) -> None:
        super().__init__()

        self.mlp = Sequential(
            Linear(input_features, output_features),
            DiffGroupNorm(output_features, 6, track_running_stats=True),
            torch.nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

class GCPNetUpdate(torch.nn.Module):
    """
    Implementation of the GCGNNAT layer composed of EdgeGatedGraphConv steps
    """

    def __init__(self, hidden_features, dropout_rate) -> None:
        super().__init__()

        # Sequential GCPNet update layers
        # Overall mapping is input_features -> output_features
        self.bondAndAngleUpdate = GCAO(dim=hidden_features,dropout_rate=dropout_rate)
        self.bondAndAtomUpdate = GCAO(dim=hidden_features,dropout_rate=dropout_rate)

    def forward(
        self,
        g: Data,
        atom_feats: torch.Tensor,
        bond_attr: torch.Tensor,
        triplet_feats: torch.Tensor,
    ) -> torch.Tensor:
        # Perform sequential edge and node updates

        bond, triplet_feats = self.bondAndAngleUpdate(
            bond_attr,  g.angle_index, triplet_feats
        )

        atom_feats, bond_attr = self.bondAndAtomUpdate(atom_feats, g.edge_index, bond)

        # Return updated node, edge, and triplet embeddings
        return atom_feats, bond_attr, triplet_feats



class GCPNet(BaseModule):
    def __init__(self, 
                 data: Data, 
                 firstUpdateLayers: int=4,
                 secondUpdateLayers: int=4,
                 atom_input_features: int=106,
                 edge_input_features: int=50,
                 triplet_input_features: int=40,
                 embedding_features: int=64,
                 hidden_features: int=256,
                 output_features: int=1,
                 min_edge_distance: float=0.0,
                 max_edge_distance: float=8.0,
                 min_angle: float=0.0,
                 max_angle: float=torch.acos(torch.zeros(1)).item() * 2,
                 link: Literal["identity", "log", "logit"] = "identity",
                 dropout_rate=0.0,
                ) -> None: 
        super().__init__()

        self.data = data  ##Todo: for future utilization

        self.atom_embedding = EmbeddingLayer(atom_input_features, hidden_features)

        self.edge_embedding = torch.nn.Sequential(
            RBFExpansion(
                vmin=min_edge_distance, vmax=max_edge_distance, bins=edge_input_features
            ),
            EmbeddingLayer(edge_input_features, embedding_features),
            EmbeddingLayer(embedding_features, hidden_features), 
        )

        self.angle_embedding = torch.nn.Sequential(
            RBFExpansion(vmin=min_angle, vmax=max_angle, bins=triplet_input_features),
            EmbeddingLayer(triplet_input_features, embedding_features),
            EmbeddingLayer(embedding_features, hidden_features), 
        )

        # layer to perform atom, bond and andle updates on the graph by 2N GCAO
        self.firstUpdate = torch.nn.ModuleList(
            [GCPNetUpdate(hidden_features,dropout_rate) for _ in range(firstUpdateLayers)]
        )

        # layer to perform atom and bond updates on the graph by N GCAO
        self.secondUpdate = torch.nn.ModuleList(
            [
                GCAO(dim=hidden_features,dropout_rate=dropout_rate)
                for _ in range(secondUpdateLayers)
            ]
        )

        # simple hign-level output layer
        self.fc = Linear(hidden_features, output_features)

        switcher = {
            "identity": self._identity,
            "log": torch.exp,
            "logit": torch.sigmoid
        }
        self.link = switcher.get(link, None)
        if link == "log":
            avg_gap = 0.7
            self.fc.bias.data = torch.tensor(np.log(avg_gap), dtype=torch.float)
 
    def _identity(self, x):
        return x
    
    @property
    def target_attr(self):
        """Specifies the target attribute property for writing output to file"""
        return "y"

    def forward(self, g: Data) -> torch.Tensor:
        # unpack data
        atom_feats = g.x
        bond_attr = g.edge_attr
        triplet_feats = g.angle_attr

        # perform initial embedding
        atom_feats = self.atom_embedding(atom_feats)
        bond_attr = self.edge_embedding(bond_attr)
        triplet_feats = self.angle_embedding(triplet_feats)

        # perform sequential GCPNet updates
        for update in self.firstUpdate:
            atom_feats, bond_attr, triplet_feats = update(
                g, atom_feats, bond_attr, triplet_feats
            )

        for update in self.secondUpdate:
            atom_feats, bond_attr = update(
                atom_feats, g.edge_index, bond_attr
            )

        # for update in self.secondUpdate:
        #     out_nodes, _ = update(
        #         atom_feats, g.edge_index, bond_attr
        #     )
        #     atom_feats = atom_feats + out_nodes # skip connection on atom feats for each update

        # readout
        out = global_mean_pool(atom_feats, g.batch)
        out = self.fc(out)


        ########################
        ###  interpretability
        # node_feats = self.fc(node_feats) 
        # out = global_mean_pool(node_feats, g.batch)
        # 
        # def atomic_number_to_symbol(atomic_number):
        #     from pymatgen.core.periodic_table import Element
        #     return Element.from_Z(atomic_number).symbol
        # 
        #
        # with open('node_contribs.csv','a') as f:
        #     # print('heat of formation [eV/atom], id, atomic symbol', file=f)
        #     for i in range(node_feats.shape[0]):
        #         print('{0},{1},{2}'.format(  torch.squeeze(node_feats, -1)[i].item(),
        #                                        g.AB[i] , 
        #                                        atomic_number_to_symbol(g.z.tolist()[i]))
        #                                        ,file=f)
        ##################################


        # apply link function
        if self.link:
            out = self.link(out)

        return torch.squeeze(out, -1)
