import torch
import numpy as np
import pandas as pd
from torch.utils.data.sampler import SubsetRandomSampler
from torch_geometric.data import Data, Dataset, DataLoader, Batch
from torch_geometric.utils import subgraph

class ProteinDataset(Dataset):
    def __init__(self, data_path, max_seq, add_virtual):
        super(Dataset, self).__init__()
        self.data_path = data_path
        self.pros_list = pd.read_csv(f"{data_path}/pdbs.csv", index_col=False).values.squeeze()
        self.max_seq = max_seq
        self.add_virtual = add_virtual

    def add_virtual_node(self, pyg_data):
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

    def add_cls(self, pyg_data):
        node_attr = pyg_data['x']
        virtual_node_attr = torch.zeros_like(node_attr[0]).unsqueeze(0)
        x = torch.concat([virtual_node_attr, node_attr], dim=0)
        return Data(x=x)

    def pad_sequence(self, seq_data):
        ## TODO: implement padding, upto self.dim
        sequence = seq_data.x
        padded_sequence = torch.zeros_like(sequence[0]).unsqueeze(0).repeat(self.max_seq, 1)
        padded_sequence[:,-1] = 1 # padding
        padded_sequence[:len(sequence),:] = sequence
        return Data(x=padded_sequence)

    def truncate_overflow(self, graph, sequence):
        trnc_edge_index, trnc_edge_attr = subgraph([i for i in range(self.max_seq - 1)], graph.edge_index, graph.edge_attr)
        graph = Data(x=graph.x[:self.max_seq - 1, :], edge_index=trnc_edge_index, edge_attr=trnc_edge_attr, y=graph.y)
        
        sequence.x = sequence.x[:self.max_seq - 1, :]
        return graph, sequence

    def __getitem__(self, index):
        pdb_code = self.pros_list[index]
        # dist_matrix = torch.load(f"{self.data_path.split("_")[-1]}_dist/{pdb_code}.pt")
        
        ### Processing the whole protein
        # load original whole protein graph (without virtual node)
        whole_prot_graph = torch.load(f"{self.data_path}/{pdb_code}_whole.pt")
        whole_prot_seq_ = torch.load(f"{self.data_path}/{pdb_code}_seq.pt") 

        # simply truncate overflowed sequences
        if whole_prot_graph.num_nodes > self.max_seq - 1:
            whole_prot_graph, whole_prot_seq_ = self.truncate_overflow(whole_prot_graph, whole_prot_seq_)
        
        # add virtual node/token
        if self.add_virtual:
            whole_prot_graph = self.add_virtual_node(whole_prot_graph)
        whole_prot_seq_ = self.add_cls(whole_prot_seq_)
        
        # pad sequences
        whole_prot_seq = self.pad_sequence(whole_prot_seq_)
        
        '''
        ### Fragmentation
        # load fragemended data        
        seqfrag_batch = torch.load(f"{self.data_path}/{pdb_code}_seqfrag.pt")
        spafrag_batch = torch.load(f"{self.data_path}/{pdb_code}_seqfrag.pt")
        # add virtual node
        seqfrag_list = []
        spafrag_list = []
        for frag_data in seqfrag_batch.to_data_list()
            seqfrag_list.append(self.add_virtual(frag_data))
        for frag_data in spafrag_batch.to_data_list()
            spafrag_list.append(self.add_virtual(frag_data))
        '''        
        return whole_prot_seq, whole_prot_graph #, seqfrag_list, spafrag_list

    def __len__(self):
        return len(self.pros_list)

class ProteinDatasetWrapper(object):
    def __init__(self, batch_size, num_workers, valid_size, data_path, max_seq, virtual_node):
        super(object, self).__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.max_seq = max_seq
        self.add_virtual = virtual_node

    def get_data_loaders(self):
        train_dataset = ProteinDataset(data_path=self.data_path, max_seq=self.max_seq, add_virtual=self.add_virtual)
        train_loader, valid_loader = self.get_train_validation_data_loaders(train_dataset)
        return train_loader, valid_loader

    def get_train_validation_data_loaders(self, train_dataset):
        # obtain training indices that will be used for validation
        num_train = len(train_dataset)
        indices = list(range(num_train))
        np.random.shuffle(indices)

        split = int(np.floor(self.valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]

        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler,
                                  num_workers=self.num_workers, drop_last=True)
        valid_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
                                  num_workers=self.num_workers, drop_last=True)

        return train_loader, valid_loader