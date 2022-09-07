import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data.sampler import SubsetRandomSampler
from torch_geometric.data import Data, Dataset, DataLoader
from tape import ProteinBertModel, TAPETokenizer

class DNABindDataset(Dataset):
    def __init__(self, pDataPath="../data/finetune/dna", pTAPE=False):
        super(Dataset, self).__init__()

        self.pDataPath = pDataPath
        self.pTAPE = pTAPE
        if self.pTAPE:
            self.TAPE = ProteinBertModel.from_pretrained('bert-base')
            self.tokenizer = TAPETokenizer(vocab='iupac')

        # Parse the target dirs and labels
        self.targetCodes = self.__parse_dataset_()
        
        # Preprocessed file does not exist: preprocessing works.
        if not os.path.exists(f"{pDataPath}/processed"):
            print(f"[ProteinNet] Preprocess the dataset first!")

    def __getitem__(self, index):
        vTargetCode = self.targetCodes[index]

        # Load original whole protein graph (without virtual node).
        vProtGraph = torch.load(f"{self.pDataPath}/processed/{vTargetCode}.pt")

        # Use TAPE encoder as a node embedding.
        if self.pTAPE:
            token_ids = torch.tensor([self.tokenizer.encode(vProtGraph.x)])
            # Remove additional tokens. [output_dim = 768]
            vProtGraph.x=self.TAPE(token_ids)[0, 1:-2, :] 
        
        return vProtGraph

    def __len__(self):
        return len(self.targetCodes)

    def __parse_dataset_(self):
        # Build the list containing pdb codes.
        if os.path.exists(f"{self.pDataPath}/pdbs.csv"):
            return pd.read_csv(f"{self.pDataPath}/pdbs.csv", index_col=False).values.squeeze()
        
        print(f"[ProteinNet] Parsing appropriate proteins.")
        for (root, dirs, files) in os.walk(f"{self.pDataPath}/processed"):
            if len(files) > 0:
                targetCodes = []
                for (i, file_name) in enumerate(tqdm(files)):
                    # Cache or garbage files.
                    if '.pt' not in file_name: continue
                    targetCodes.append(file_name.split('.')[0])
        df = pd.DataFrame(targetCodes)
        df.to_csv(f"{self.pDataPath}/pdbs.csv", index=False)

        return pd.read_csv(f"{self.pDataPath}/pdbs.csv", index_col=False).values.squeeze()   


class DNABindDatasetWrapper(object):
    def __init__(self, batch_size, num_workers, valid_size, pDataPath, pTAPE):
        super(object, self).__init__()
        self.pDataPath = pDataPath
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.pTAPE = pTAPE

    def get_data_loaders(self):
        train_dataset    = DNABindDataset(pDataSplit='train',    pDataPath=self.pDataPath, pTAPE=self.pTAPE)
        test_129_dataset = DNABindDataset(pDataSplit='test_129', pDataPath=self.pDataPath, pTAPE=self.pTAPE)
        test_181_dataset = DNABindDataset(pDataSplit='test_181', pDataPath=self.pDataPath, pTAPE=self.pTAPE)
        

        train_loader        = DataLoader(train_dataset, batch_size = self.batch_size,    sampler=SubsetRandomSampler(list(range(len(train_dataset)))), 
                                    num_workers=self.num_workers, drop_last=True)
        test_129_loader     = DataLoader(test_129_dataset, batch_size = self.batch_size, sampler=SubsetRandomSampler(list(range(len(test_129_dataset)))), 
                                    num_workers=self.num_workers, drop_last=True)
        test_181_loader     = DataLoader(test_181_dataset, batch_size = self.batch_size, sampler=SubsetRandomSampler(list(range(len(test_181_dataset)))), 
                                    num_workers=self.num_workers, drop_last=True)
        
        return train_loader, test_129_loader, test_181_loader
