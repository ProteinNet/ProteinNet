'''
    \file GraphEncoder.py
    
    \brief GraphEncoders for encoding proteins.

    \author MinGyu Choi chemgyu98@snu.ac.kr
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import Sequential, BatchNorm, GraphNorm, GCNConv, SAGEConv, GATConv, GINConv, global_mean_pool, SAGPooling, global_max_pool, global_add_pool, GraphMultisetTransformer
from preprocess.ogb_util import get_atom_feature_dims, get_bond_feature_dims
#from modules import GraphMultisetTransformer, spspmm
from preprocess.pyg_util import dense_to_sparse, to_dense_adj

# Packages for graph U net.
# from torch_sparse import spspmm
#from utils import TopKPooling
from torch_geometric.utils import (
    add_self_loops,
    remove_self_loops,
    sort_edge_index,
)
from torch_geometric.utils.repeat import repeat


class GraphEmb(torch.nn.Module):
    '''
        Embedding model for converting initial node/edge attribute embeddings into learnable vectors.
    '''
    def __init__(self, pDimNodeHidden, pDimEdgeHidden):
        super(GraphEmb, self).__init__()
        # Parameters
        self.pDimNodeHidden = pDimNodeHidden
        self.pDimEdgeHidden = pDimEdgeHidden
        full_atom_feature_dims = get_atom_feature_dims()
        full_bond_feature_dims = get_bond_feature_dims()

        # Layers
        self.lNodeEmb = torch.nn.ModuleList()
        for i, vNodeDim in enumerate(full_atom_feature_dims):
            lNodeEmb_ = torch.nn.Embedding(vNodeDim, pDimNodeHidden)
            torch.nn.init.xavier_uniform_(lNodeEmb_.weight.data)
            self.lNodeEmb.append(lNodeEmb_)
        
        self.lEdgeEmb = torch.nn.ModuleList()
        for i, vEdgeEmb in enumerate(full_bond_feature_dims):
            lEdgeEmb_ = torch.nn.Embedding(vEdgeEmb, pDimEdgeHidden)
            torch.nn.init.xavier_uniform_(lEdgeEmb_.weight.data)
            self.lEdgeEmb.append(lEdgeEmb_)

    def forward(self, dNodeAttr, dEdgeAttr):        
        vNodeAttr = torch.zeros(dNodeAttr.shape[0], self.pDimNodeHidden)
        vEdgeAttr = torch.zeros(dEdgeAttr.shape[0], self.pDimEdgeHidden)
        for i in range(dNodeAttr.shape[1]): vNodeAttr += self.lNodeEmb[i](dNodeAttr[:,i])
        for i in range(dEdgeAttr.shape[1]): vEdgeAttr += self.lEdgeEmb[i](dEdgeAttr[:,i])
        
        return vNodeAttr, vEdgeAttr
    
    def from_pretrained(self, model_file):
        self.lNodeEmb.load_state_dict(torch.load(model_file), strict=False)
        self.lEdgeEmb.load_state_dict(torch.load(model_file), strict=False)
        return
        


class GCNEnc(torch.nn.Module):
    r"""The graph convolutional operator from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper
    """
    def __init__(self, pNumLayers, pDim, pHDim, pEdgeHDim, pDropRatio, pNumLabels, pPoolRatio=0.0, pSumRes=True, pRetEmb=False, pGraphPred=False):
        super(GCNEnc, self).__init__()
        
        convLayers = []
        curDim, curHDim = pDim, pHDim
        for i in range(pNumLayers):
            convLayers.append((GCNConv(curDim, curHDim), 'x, edge_index -> x'))
            convLayers.append(BatchNorm(curHDim))
            convLayers.append(nn.ReLU(inplace=True))
            curDim = curHDim
            if i < 2: curHDim *= 2
        convLayers.append(nn.Linear(curHDim, pDim))
        self.convolution = Sequential('x, edge_index', convLayers)
        
        readoutLayers = []
        readoutLayers.append((GraphMultisetTransformer(pDim, pDim, pDim), 'x, batch, edge_index -> x'))
        readoutLayers.append(nn.Linear(pDim, pNumLabels))
        self.readout = Sequential('x, batch, edge_index', readoutLayers)


    def forward(self, x, edge_index, edge_weight, batch):
        x = self.convolution(x, edge_index)
        pred = self.readout(x, batch, edge_index)
        return pred
    
    def from_pretrained(self, model_file):
        self.convolution.load_state_dict(torch.load(model_file))
        self.readout.load_state_dict(torch.load(model_file))
        return

class SAGEEnc(torch.nn.Module):
    r"""The GraphSAGE operator from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper
    """
    def __init__(self, pNumLayers, pDim, pHDim, pEdgeHDim, pDropRatio, pNumLabels, pPoolRatio=0.0, pSumRes=True, pRetEmb=False, pGraphPred=False):
        super(SAGEEnc, self).__init__()
        
        convLayers = []
        curDim, curHDim = pDim, pHDim
        for i in range(pNumLayers):
            convLayers.append((SAGEConv(curDim, curHDim), 'x, edge_index -> x'))
            convLayers.append(BatchNorm(curHDim))
            convLayers.append(nn.ReLU(inplace=True))
            curDim = curHDim
            if i < 2: curHDim *= 2
        convLayers.append(nn.Linear(curHDim, pDim))
        self.convolution = Sequential('x, edge_index', convLayers)

        readoutLayers = []
        readoutLayers.append((GraphMultisetTransformer(pDim, pDim, pDim), 'x, batch, edge_index -> x'))
        readoutLayers.append(nn.Linear(pDim, pNumLabels))
        self.readout = Sequential('x, batch, edge_index', readoutLayers)

    def forward(self, x, edge_index, edge_weight, batch):
        x = self.convolution(x, edge_index)
        pred = self.readout(x, batch, edge_index)
        return pred
        
    def from_pretrained(self, model_file):
        self.convolution.load_state_dict(torch.load(model_file))
        self.readout.load_state_dict(torch.load(model_file))
        return

class GATEnc(torch.nn.Module):
    r"""The graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper
    """
    def __init__(self, pNumLayers, pDim, pHDim, pDropRatio, pNumLabels, pHeads=2, pPoolRatio=0.0, pSumRes=True, pRetEmb=False, pGraphPred=False):
        super(GATEnc, self).__init__()
        
        convLayers = []
        curDim, curHDim = pDim, pHDim
        for i in range(pNumLayers):
            convLayers.append((GATConv(curDim, curHDim//pHeads, heads=pHeads, dropout=pDropRatio), 'x, edge_index -> x'))
            convLayers.append(BatchNorm(curHDim))
            convLayers.append(nn.ReLU(inplace=True))
            curDim = curHDim
            if i < 2: curHDim *= 2
        convLayers.append(nn.Linear(curHDim, pDim))
        self.convolution = Sequential('x, edge_index', convLayers)

        readoutLayers = []
        readoutLayers.append((GraphMultisetTransformer(pDim, pDim, pDim), 'x, batch, edge_index -> x'))
        readoutLayers.append(nn.Linear(pDim, pNumLabels))
        self.readout = Sequential('x, batch, edge_index', readoutLayers)

    def forward(self, x, edge_index, edge_weight, batch):
        if edge_index.dtype != torch.long: edge_index = edge_index.to(torch.long)
        print(edge_index.shape)
        x = self.convolution(x, edge_index)#, edge_weight)
        pred = self.readout(x, batch, edge_index)
        return pred
    
    def from_pretrained(self, model_file):
        self.convolution.load_state_dict(torch.load(model_file))
        self.readout.load_state_dict(torch.load(model_file))
        return

class GINEnc(torch.nn.Module):
    r"""The graph isomorphism operator from the `"How Powerful are
    Graph Neural Networks?" <https://arxiv.org/abs/1810.00826>`_ paper
    """
    def __init__(self, pNumLayers, pDim, pHDim, pEdgeHDim, pDropRatio, pNumLabels, pPoolRatio=0.0, pSumRes=True, pRetEmb=False, pGraphPred=False):
        super(GINEnc, self).__init__()
        
        convLayers = []
        curDim, curHDim = pDim, pHDim
        for i in range(pNumLayers):
            convLayers.append((GINConv(nn.Sequential(nn.Linear(curDim, curHDim), nn.ReLU(), nn.Linear(curHDim, curHDim))), 'x, edge_index -> x'))
            convLayers.append(BatchNorm(curHDim))
            convLayers.append(nn.ReLU(inplace=True))
            curDim = curHDim
            if i < 2: curHDim *= 2
        convLayers.append(nn.Linear(curHDim, pDim))
        self.convolution = Sequential('x, edge_index', convLayers)

        readoutLayers = []
        readoutLayers.append((GraphMultisetTransformer(pDim, pDim, pDim), 'x, batch, edge_index -> x'))
        readoutLayers.append(nn.Linear(pDim, pNumLabels))
        self.readout = Sequential('x, batch, edge_index', readoutLayers)

    def forward(self, x, edge_index, edge_weight, batch):
        x = self.convolution(x, edge_index)
        pred = self.readout(x, batch, edge_index)
        return pred
    
    def from_pretrained(self, model_file):
        self.convolution.load_state_dict(torch.load(model_file))
        return

class GUEnc(torch.nn.Module):
    r"""The Graph U-Net model from the `"Graph U-Nets"
    <https://arxiv.org/abs/1905.05178>`_ paper which implements a U-Net like
    architecture with graph pooling and unpooling operations.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        out_channels (int): Size of each output sample.
        depth (int): The depth of the U-Net architecture.
        pool_ratios (float or [float], optional): Graph pooling ratio for each
            depth. (default: :obj:`0.5`)
        sum_res (bool, optional): If set to :obj:`False`, will use
            concatenation for integration of skip connections instead
            summation. (default: :obj:`True`)
        act (torch.nn.functional, optional): The nonlinearity to use.
            (default: :obj:`torch.nn.functional.relu`)
    """
    def __init__(self, pNumLayers, pDim, pHDim, pDropRatio, pNumLabels, pPoolRatio, pSumRes):
        super(GUEnc, self).__init__()
        self.lDownConvs = torch.nn.ModuleList()
        self.lPools     = torch.nn.ModuleList()
        self.lUpConvs   = torch.nn.ModuleList()
        self.pNumLayers = pNumLayers
        self.pSumRes    = pSumRes
        self.lAct       = F.relu

        self.lDownConvs.append(GCNConv(pDim, pHDim, improved=True))
        for i in range(pNumLayers):
            self.lPools.append(TopKPooling(pHDim, pPoolRatio))
            self.lDownConvs.append(GCNConv(pHDim, pHDim, improved=True))

        pDim = pHDim if pSumRes else 2*pHDim
        for i in range(pNumLayers - 1):
            self.lUpConvs.append(GCNConv(pDim, pHDim, improved=True))
        self.lUpConvs.append(GCNConv(pHDim, pDim, improved=True))
        
        self.reset_parameters()

        readoutLayers = []
        readoutLayers.append((GraphMultisetTransformer(pDim, pDim, pDim), 'x, batch, edge_index -> x'))
        readoutLayers.append(nn.Linear(pDim, pNumLabels))
        self.readout = Sequential('x, batch, edge_index', readoutLayers)

    def reset_parameters(self):
        for conv in self.lDownConvs:
            conv.reset_parameters()
        for pool in self.lPools:
            pool.reset_parameters()
        for conv in self.lUpConvs:
            conv.reset_parameters()


    def forward(self, x, edge_index, edge_weight, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        batch_orig = batch
        edge_weight = x.new_ones(edge_index.size(1))

        x = self.lDownConvs[0](x, edge_index, edge_weight)
        x = self.lAct(x)

        xs = [x]
        edge_indices = [edge_index]
        edge_weights = [edge_weight]
        perms = []

        for i in range(1, self.pNumLayers + 1):
            # TODO: Due to an spspmm error... fix this...
            # edge_index, edge_weight = self.augment_adj(edge_index, edge_weight, x.size(0))
            
            x, edge_index, edge_weight, batch, perm, _ = self.lPools[i - 1](x, edge_index, edge_weight, batch)

            x = self.lDownConvs[i](x, edge_index, edge_weight)
            x = self.lAct(x)

            if i < self.pNumLayers:
                xs += [x]
                edge_indices += [edge_index]
                edge_weights += [edge_weight]
            perms += [perm]

        for i in range(self.pNumLayers):
            j = self.pNumLayers - 1 - i

            res = xs[j]
            edge_index = edge_indices[j]
            edge_weight = edge_weights[j]
            perm = perms[j]

            up = torch.zeros_like(res)
            up[perm] = x
            x = res + up if self.pSumRes else torch.cat((res, up), dim=-1)

            x = self.lUpConvs[i](x, edge_index, edge_weight)
            x = self.lAct(x) if i < self.pNumLayers - 1 else x

        pred = self.readout(x, batch_orig, edge_index)
        
        return pred

    def augment_adj(self, edge_index, edge_weight, num_nodes):
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 num_nodes=num_nodes)
        edge_index, edge_weight = sort_edge_index(edge_index, edge_weight,
                                                  num_nodes)
        edge_index, edge_weight = spspmm(edge_index, edge_weight, edge_index,
                                         edge_weight, num_nodes, num_nodes,
                                         num_nodes)
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        return edge_index, edge_weight
        
    def from_pretrained(self, model_file):
        self.convolution.load_state_dict(torch.load(model_file))
        return

# class GINEEnc(torch.nn.Module):
#     def __init__(self, num_layers, dim, hidden_dim, drop_ratio, heads=1):
#         super(GINEEnc, self).__init__()
        
#         self.convolution = nn.Sequential([])
#         for i in range(num_layers-1):
#             curDim, curHDim = dim, hidden_dim
#             self.convolution.append(GCNConv(curDim, curHDim, heads=heads, dropout=drop_ratio))
#             self.convolution.append(nn.BatchNorm1d(curHDim))
#             self.convolution.append(nn.ReLU(inplace=True))
#             curDim = hidden_dim
#             if i < 2: curHDim *= 2
#         self.convolution.append(nn.Linear(curHDim, dim))

#     def forward(self, x, edge_index, edge_weight):
#         x = self.convolution(x, edge_index, edge_weight)
#         return x


class GUEnc_V(torch.nn.Module):
    r"""The Graph U-Net model from the `"Graph U-Nets"
    <https://arxiv.org/abs/1905.05178>`_ paper which implements a U-Net like
    architecture with graph pooling and unpooling operations.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        out_channels (int): Size of each output sample.
        depth (int): The depth of the U-Net architecture.
        pool_ratios (float or [float], optional): Graph pooling ratio for each
            depth. (default: :obj:`0.5`)
        sum_res (bool, optional): If set to :obj:`False`, will use
            concatenation for integration of skip connections instead
            summation. (default: :obj:`True`)
        act (torch.nn.functional, optional): The nonlinearity to use.
            (default: :obj:`torch.nn.functional.relu`)
    """
    def __init__(self, pNumLayers, pDim, pHDim, pEdgeHDim, pDropRatio, pNumLabels, pPoolRatio, pSumRes, pRetEmb=False, pGraphPred=False):
        super(GUEnc_V, self).__init__()
        self.lDownConvs = torch.nn.ModuleList()
        self.lPools     = torch.nn.ModuleList()
        self.lUpConvs   = torch.nn.ModuleList()
        self.lNorms     = torch.nn.ModuleList()
        self.pNumLayers = pNumLayers
        self.pEdgeHDim  = pEdgeHDim
        self.pSumRes    = pSumRes
        self.lAct       = F.leaky_relu
        self.pRetEmb    = pRetEmb
        self.pGraphPred = pGraphPred
        self.pDim = pDim

        # Virtual node
        self.vVirtualEmb = torch.nn.Embedding(1, pHDim)
        torch.nn.init.constant_(self.vVirtualEmb.weight.data, 0)
        self.lVirtualMLPs = torch.nn.ModuleList()
        for _ in range(pNumLayers-1):
            self.lVirtualMLPs.append(torch.nn.Sequential(torch.nn.Linear(pHDim, 2*pHDim), torch.nn.BatchNorm1d(2*pHDim), torch.nn.ReLU(), \
                                     torch.nn.Linear(2*pHDim, pHDim), torch.nn.BatchNorm1d(pHDim), torch.nn.ReLU()))

        self.lEdgeEmb = torch.nn.Linear(self.pEdgeHDim, 1)

        # DownConvolution layers.
        self.lDownConvs.append(GCNConv(pDim, pHDim))#, improved=True))
        self.lNorms.append(torch.nn.BatchNorm1d(pDim))
        for _ in range(pNumLayers):
            self.lPools.append(TopKPooling(pHDim, pPoolRatio))
            # self.lPools.append(SAGPooling(pHDim, pPoolRatio))
            self.lDownConvs.append(GCNConv(pHDim, pHDim))# , improved=True))
            self.lNorms.append(torch.nn.BatchNorm1d(pHDim))


        # UpConvolution layers.
        pDim = pHDim if pSumRes else 2*pHDim
        for _ in range(pNumLayers - 1):
            self.lUpConvs.append(GCNConv(pDim, pHDim, improved=True))
        self.lUpConvs.append(GCNConv(pHDim, pDim, improved=True))
        
        self.reset_parameters()

        # Readout layers.
        # if self.pRetEmb:
        self.readout = Sequential('x, batch', [(global_add_pool, 'x, batch -> x'),
                                                nn.Linear(self.pDim, self.pDim),
                                                nn.BatchNorm1d(self.pDim)])
        self.x_readout = torch.nn.Linear(pHDim, self.pDim)
        self.x_norm = nn.BatchNorm1d(self.pDim)

        if self.pGraphPred:
        # readoutLayers = []
        # readoutLayers.append((GraphMultisetTransformer(pDim, pDim, pDim), 'x, batch, edge_index -> x'))
        # readoutLayers.append(nn.Linear(pDim, pNumLabels))
        # self.graph_readout = Sequential('x, batch, edge_index', readoutLayers)

            self.graph_readout = nn.Linear(self.pDim, pNumLabels)

    def reset_parameters(self):
        for conv in self.lDownConvs:
            conv.reset_parameters()
        for pool in self.lPools:
            pool.reset_parameters()
        for conv in self.lUpConvs:
            conv.reset_parameters()


    def forward(self, x, edge_index, edge_weight, batch=None):
        if batch is None: batch = edge_index.new_zeros(x.size(0))
        batch_orig = batch
        
        # edge_weight = x.new_ones(edge_index.size(1))
        edge_weight = self.lEdgeEmb(edge_weight)
        edge_weight = edge_weight - torch.min(edge_weight)
        edge_weight = edge_weight / torch.max(edge_weight + 1e-7)

        x = self.lNorms[0](x)
        x = self.lDownConvs[0](x, edge_index, edge_weight)
        x = self.lAct(x)
        
        xs = [x]
        edge_indices = [edge_index]
        edge_weights = [edge_weight]
        perms = []
        # vVirtualEmbed = self.vVirtualEmb(torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))
        for i in range(1, self.pNumLayers + 1):
            # TODO: Due to an spspmm error... fix this...
            # edge_index, edge_weight = self.augment_adj(edge_index, edge_weight, x.size(0))
            x, edge_index, edge_weight, batch, perm, _ = self.lPools[i - 1](x, edge_index, edge_weight, batch)
            x = self.lNorms[i](x)
            x = self.lDownConvs[i](x, edge_index, edge_weight)

            if not i == self.pNumLayers:
                x = self.lAct(x)    
                
            if i < self.pNumLayers:
                xs += [x]
                edge_indices += [edge_index]
                edge_weights += [edge_weight]
            perms += [perm]

        # Return compressed representations.
        if self.pRetEmb or self.pGraphPred:
            x = self.x_readout(x)
            x = self.x_norm(x)
            
            if self.pRetEmb:
                return self.readout(x, batch), x, batch
            elif self.pGraphPred:
                pred = self.graph_readout(self.readout(x, batch))
                return pred  

        # if self.pGraphPred:
        #     pred = self.graph_readout(x, batch, edge_index)
        #     return pred
        

        for i in range(self.pNumLayers):
            j = self.pNumLayers - 1 - i

            res = xs[j]
            edge_index = edge_indices[j]
            edge_weight = edge_weights[j]
            perm = perms[j]

            up = torch.zeros_like(res)
            up[perm] = x
            x = res + up if self.pSumRes else torch.cat((res, up), dim=-1)

            x = self.lUpConvs[i](x, edge_index, edge_weight)
            x = self.lAct(x) if i < self.pNumLayers - 1 else x

        pred = self.readout(x, batch_orig, edge_index)
        
        return pred

    def augment_adj(self, edge_index, edge_weight, num_nodes):
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 num_nodes=num_nodes)
        dense_edge = to_dense_adj(edge_index)
        dense_2hop = torch.matmul(dense_edge, dense_edge).bool().float() - dense_edge # Only two-hop edges are remaining.
        dense_1hop = to_dense_adj(edge_index, edge_attr=edge_weight)
        dense = dense_1hop + dense_2hop
        edge_index, edge_weight = dense_to_sparse(dense)
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

        return edge_index, edge_weight

    def augment_adj_old(self, edge_index, edge_weight, num_nodes):
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 num_nodes=num_nodes)
        edge_index, edge_weight = sort_edge_index(edge_index, edge_weight,
                                                  num_nodes)
        edge_index, edge_weight = spspmm(edge_index, edge_weight, edge_index,
                                         edge_weight, num_nodes, num_nodes,
                                         num_nodes)
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        return edge_index, edge_weight
        
    def from_pretrained(self, model_file):
        self.lDownConvs.load_state_dict(model_file['lDownConvs'], strict=False)
        return
