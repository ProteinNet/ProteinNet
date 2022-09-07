'''
    \file dataset_HomologyTAPE.py
    
    \brief pytorch geometric Dataset object for HomologyTAPE finetuning tasks.

    \author MinGyu Choi chemgyu98@snu.ac.kr
'''

import os
import numpy as np
from tqdm import tqdm
import torch
from torch_geometric.data import Data, Dataset, DataLoader
from torch_geometric.utils import dense_to_sparse, to_dense_adj
from torch.utils.data.sampler import SubsetRandomSampler


class ProtClassHomologyTAPEDataset(Dataset):
    """
    ProtClass100 dataset class.
    Adopted & modified from https://github.com/phermosilla/IEConv_proteins
    """
    def __init__(self, pDataSplit="training", pDataPath="../data/HomologyTAPE", pMaxLen=1024, pAddVirtual=False, pDebug=None):
        
        super(Dataset, self).__init__()
        self.pMaxLen = pMaxLen
        self.pDataPath = pDataPath
        self.pDataSplit = pDataSplit
        self.pAddVirtual = pAddVirtual
        
        # Parse the target dirs and labels
        self.targetCodes, self.targetLabels = self.__parse_dataset_()
        
        # For debugging
        if not pDebug == None:       
            self.__preprocess_debug__(pDataPath, pDataSplit, pDebug)

        # Preprocessed file does not exist: preprocessing works.
        if not os.path.exists(f"{pDataPath}/processed"):
            print(f"[GraphConSeq Finetuning] Preprossesed file does not exist: preprocessing started.")
            self.out_dir = f"{pDataPath}/processed"
            os.mkdirs(self.out_dir, exists_ok=True)
            self.__preprocess__(pDataPath, pDataSplit)

    def __getitem__(self, index):
        vTargetCode, vTargetLabel = self.targetCodes[index], self.targetLabels[index]
        
        ## Load original whole protein graph (without virtual node)
        vProtGraph = torch.load(f"{self.pDataPath}/processed/{self.pDataSplit}/{vTargetCode}.pt")
        vProtGraph.y = vTargetLabel
        
        if self.pAddVirtual: vProtGraph = self.__add_virtual_node_(vProtGraph)
        
        return vProtGraph

    def __len__(self):
        return len(self.targetCodes)

    def __parse_dataset_(self):
        # Build the dictionary mapping classnames to integers.
        maxIndex = 0
        self.classes_ = {}
        with open(self.pDataPath + "/class_map.txt", 'r') as mFile:
            for line in mFile:
                lineList = line.rstrip().split('\t')
                self.classes_[lineList[0]] = int(lineList[1])
                maxIndex = max(maxIndex, int(lineList[1]))
        self.classesList_ = ["" for i in range(maxIndex+1)]
        for key, value in self.classes_.items():
            self.classesList_[value] = key

        # Parse the file list.
        targetCodes, targetLabels = [], []
        with open(f"{self.pDataPath}/{self.pDataSplit}.txt", 'r') as mFile:
            for curLine in mFile:
                splitLine = curLine.rstrip().split('\t')
                if os.path.exists(f"{self.pDataPath}/processed/{self.pDataSplit}/{splitLine[0]}.pt"): 
                    targetCodes.append(f"{splitLine[0]}")
                    targetLabels.append(self.classes_[splitLine[-1]])
        
        return targetCodes, targetLabels

    def __add_virtual_node_(self, pyg_data):
        edge_attr, edge_index, node_attr, y = pyg_data['edge_attr'], pyg_data['edge_index'], pyg_data['x'], pyg_data['y']
        
        ## Add virtual node attribute (would be converted to learnable tensor after loading to GPU.)
        virtual_node_attr = torch.zeros_like(node_attr[1]).unsqueeze(0)
        x = torch.concat([virtual_node_attr, node_attr], dim=0)

        ## Add additional edge type on existing edge attribute
        virtual_edge_type = torch.zeros_like(edge_attr.T[0]).unsqueeze(1)
        edge_attr = torch.concat([edge_attr, virtual_edge_type], dim=1)

        ### Add additional edge indices on edge index
        edge_index_ = (edge_index + 1) # virtual node = index 0
        other_node_indices = torch.from_numpy(np.asarray([i+1 for i in range(node_attr.shape[0])], dtype=int)).unsqueeze(0)
        virtual_node_indices = torch.zeros_like(other_node_indices)
        virtual_edge_indices = torch.concat([torch.concat([other_node_indices, virtual_node_indices], dim=0), torch.concat([virtual_node_indices, other_node_indices], dim=0)], dim=1)
        virtual_edge_indices = torch.concat([torch.zeros_like(virtual_edge_indices[:, 0]).unsqueeze(1), virtual_edge_indices], dim=1)
        edge_index = torch.concat([virtual_edge_indices, edge_index_], dim=1)

        ### Add additional edge attributes corresponding to the edge_attr
        virtual_attr_ = torch.zeros_like(edge_attr[0]).unsqueeze(0)
        virtual_attr_[0][-1] = 1 # virtual node type
        virtual_attr = virtual_attr_.repeat(virtual_edge_indices.shape[1], 1)
        edge_attr = torch.concat([virtual_attr, edge_attr], dim=0)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


class ProtClassHomologyTAPEDatasetWrapper(object):
    def __init__(self, batch_size, num_workers, valid_size, data_path, max_seq, virtual_node):
        super(object, self).__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.max_seq = max_seq
        self.add_virtual = virtual_node


    def get_data_loaders(self):
        train_dataset           = ProtClassHomologyTAPEDataset(pDataSplit="training",           pDataPath=self.data_path, pMaxLen=self.max_seq, pAddVirtual=self.add_virtual)
        valid_dataset           = ProtClassHomologyTAPEDataset(pDataSplit="validation",         pDataPath=self.data_path, pMaxLen=self.max_seq, pAddVirtual=self.add_virtual)
        test_family_dataset     = ProtClassHomologyTAPEDataset(pDataSplit="test_family",        pDataPath=self.data_path, pMaxLen=self.max_seq, pAddVirtual=self.add_virtual)
        test_superfamily_dataset= ProtClassHomologyTAPEDataset(pDataSplit="test_superfamily",   pDataPath=self.data_path, pMaxLen=self.max_seq, pAddVirtual=self.add_virtual)
        test_fold_dataset       = ProtClassHomologyTAPEDataset(pDataSplit="test_fold",          pDataPath=self.data_path, pMaxLen=self.max_seq, pAddVirtual=self.add_virtual)

        train_loader            = DataLoader(train_dataset, batch_size = self.batch_size, sampler=SubsetRandomSampler(list(range(len(train_dataset)))),
                                  num_workers=self.num_workers, drop_last=True)
        valid_loader            = DataLoader(valid_dataset, batch_size = self.batch_size, sampler=SubsetRandomSampler(list(range(len(valid_dataset)))),
                                  num_workers=self.num_workers, drop_last=True)
        test_family_loader      = DataLoader(test_family_dataset, batch_size = self.batch_size, sampler=SubsetRandomSampler(list(range(len(test_family_dataset)))),
                                  num_workers=self.num_workers, drop_last=True)
        test_superfamily_loader = DataLoader(test_superfamily_dataset, batch_size = self.batch_size, sampler=SubsetRandomSampler(list(range(len(test_superfamily_dataset)))),
                                  num_workers=self.num_workers, drop_last=True)
        test_fold_loader        = DataLoader(test_fold_dataset, batch_size = self.batch_size, sampler=SubsetRandomSampler(list(range(len(test_fold_dataset)))),
                                  num_workers=self.num_workers, drop_last=True)
        
        return train_loader, valid_loader, test_family_loader, test_superfamily_loader, test_fold_loader



if __name__ == "__main__":
    """
    Preprocessing
    """
    training         = ProtClassHomologyTAPEDataset()
    validation       = ProtClassHomologyTAPEDataset(pDataSplit="validation")
    test_family      = ProtClassHomologyTAPEDataset(pDataSplit="test_family")
    test_fold        = ProtClassHomologyTAPEDataset(pDataSplit="test_fold")
    test_superfamily = ProtClassHomologyTAPEDataset(pDataSplit="test_superfamily")