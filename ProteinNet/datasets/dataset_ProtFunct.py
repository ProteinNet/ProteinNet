'''
    \file dataset_ProtFunct.py
    
    \brief pytorch geometric Dataset object for ProtFunct finetuning tasks.

    \author MinGyu Choi chemgyu98@snu.ac.kr
'''

import os
import numpy as np
from tqdm import tqdm
import torch
from torch_geometric.data import Data, Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


class ProtClassProtFunctDataset(Dataset):
    """
    ProtClass100 dataset class.
    Adopted & modified from https://github.com/phermosilla/IEConv_proteins
    """
    def __init__(self, pDataSplit="training", pDataPath="../data/finetune/ProtFunct", pMaxLen=1024, pSpaRad=8,\
                pInterval=1.5, pAddVirtual=False, pDebug=None):
        
        super(Dataset, self).__init__()
        self.pMaxLen = pMaxLen
        self.pSpaRad = pSpaRad
        self.pDataPath = pDataPath
        self.pInterval = pInterval
        self.pDataSplit = pDataSplit
        self.pAddVirtual = pAddVirtual
        # Parse the target dirs and labels
        self.targetCodes, self.targetLabels = self.__parse_dataset_()
        
        # Preprocessed file does not exist: preprocessing works.
        if not os.path.exists(f"{pDataPath}/processed"):
            print(f"[GraphConSeq Finetuning] Preprossesed file does not exist: preprocessing started.")
            self.out_dir = f"{pDataPath}/processed"
            os.mkdirs(self.out_dir, exists_ok=True)
            self.__preprocess__(pDataPath, pDataSplit)

    def __getitem__(self, index):
        vTargetCode, vTargetLabel = self.targetCodes[index], self.targetLabels[index]
        
        # Load the protein graph (without virtual node) 
        vTargetCode, vTargetChain = vTargetCode.split('.')
        vProtGraph = torch.load(f"{self.pDataPath}/processed/{vTargetCode}-{vTargetChain}.pt")

        vProtGraph.edge_attr = vProtGraph.edge_attr.long() # erase this for publication.

        vProtGraph.y = vTargetLabel
        
        if self.pAddVirtual: vProtGraph = self.__add_virtual_node_(vProtGraph)
              
        return vProtGraph

    def __len__(self):
        return len(self.targetCodes)

    def __parse_dataset_(self):
        # Parse the target proteins.
        targetCodes = []
        with open(f"{self.pDataPath}/{self.pDataSplit}.txt", 'r') as mFile:
            for curLine in mFile:
                # some files are not converted.
                if os.path.exists(f"{self.pDataPath}/processed/{curLine.strip().split('.')[0]}-{curLine.strip().split('.')[1]}.pt"):
                    targetCodes.append(curLine.rstrip())

        # Load the labels.
        targetLabelDict = {}
        with open(f"{self.pDataPath}/chain_functions.txt", 'r') as mFile:
            for curLine in mFile:
                splitLine = curLine.rstrip().split(',')
                if splitLine[0] in targetCodes:
                    targetLabelDict[splitLine[0]] = int(splitLine[1])
        
        # Refactor to the list.
        targetLabels = []
        for targetCode in targetCodes:
            targetLabels.append(targetLabelDict[targetCode])

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
    
    def __preprocess__(self, pDataPath, pDataSplit):
        print(f"[GraphConSeq Finetuning] Preprocessing begins.")
        for (root, dirs, files) in os.walk(f"{pDataPath}/data"):
            if len(files) > 0:
                for (i, file_name) in enumerate(tqdm(files)):
                    # Cache or garbage files.
                    if '.hdf5' not in file_name:
                        continue
                    queryname = f"{file_name.split('.')[0]}.{file_name.split('.')[1]}"
                    
                    if os.path.exists(f"{self.out_dir}/{queryname}_whole.pt"): continue

                    # Read hdf5 type raw data
                    periodicTable_ = PyPeriodicTable()
                    curProtein = PyProtein(periodicTable_)
                    self.numAAs_ = len(periodicTable_.aLabels_) # default = 26
                    curProtein.load_hdf5(f"{pDataPath}/data/{file_name}")
                    
                    # Build pytorch geometric graphs
                    distMatrix = self.__calc_dist_matrix(curProtein.aminoPos_[0]) #                [N, N]
                    aa_type = torch.from_numpy(curProtein.aminoType_).unsqueeze(0) + 1 # ME    Encoding [1, N]
                    node_attr = self.__get_nodes__(aa_type, distMatrix)           # NotME Encoding
                    self.seqLength = len(node_attr)
                    edge_index, edge_attr = self.__get_edges__(torch.from_numpy(curProtein.aminoNeighs_), torch.from_numpy(curProtein.aminoNeighsHB_), distMatrix) # [2, E], [N,N]
                    
                    # TODO: Fragmentation
                    
                    whole_prot = Data(x=node_attr, edge_index=edge_index, edge_attr=edge_attr)
                    whole_seq  = Data(x=aa_type)
                    
                    torch.save(whole_prot, f"{self.out_dir}/{queryname}_whole.pt")
                    torch.save(whole_seq, f"{self.out_dir}/{queryname}_seq.pt")
                    torch.save(distMatrix, f"{self.dist_dir}/{queryname}.pt")
        print(f"[GraphConSeq Finetuning] Preprocessing finished.")


class ProtClassProtFunctDatasetWrapper(object):
    def __init__(self, batch_size, num_workers, valid_size, data_path, max_seq, virtual_node, interval):
        super(object, self).__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.max_seq = max_seq
        self.add_virtual = virtual_node
        self.interval = interval

    def get_data_loaders(self):
        train_dataset           = ProtClassProtFunctDataset(pDataSplit="training"   ,pDataPath=self.data_path, pMaxLen=self.max_seq, pAddVirtual=self.add_virtual, pInterval=self.interval)
        valid_dataset           = ProtClassProtFunctDataset(pDataSplit="validation" ,pDataPath=self.data_path, pMaxLen=self.max_seq, pAddVirtual=self.add_virtual, pInterval=self.interval)
        test_dataset            = ProtClassProtFunctDataset(pDataSplit="testing"    ,pDataPath=self.data_path, pMaxLen=self.max_seq, pAddVirtual=self.add_virtual, pInterval=self.interval)
        
        train_loader            = DataLoader(train_dataset, batch_size = self.batch_size, sampler=SubsetRandomSampler(list(range(len(train_dataset)))),
                                  num_workers=self.num_workers, drop_last=True)
        valid_loader            = DataLoader(valid_dataset, batch_size = self.batch_size, sampler=SubsetRandomSampler(list(range(len(valid_dataset)))),
                                  num_workers=self.num_workers, drop_last=True)
        test_loader             = DataLoader(test_dataset, batch_size = self.batch_size, sampler=SubsetRandomSampler(list(range(len(test_dataset)))),
                                  num_workers=self.num_workers, drop_last=True)
        
        return train_loader, valid_loader, test_loader

if __name__ == "__main__":
    """
    Preprocessing
    """
    training         = ProtClassProtFunctDataset(pDataSplit="training")
    validation       = ProtClassProtFunctDataset(pDataSplit="validation")
    test             = ProtClassProtFunctDataset(pDataSplit="testing")
    