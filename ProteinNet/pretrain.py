'''
Licensed under the MIT License.

Copyright (c) ProteinNet Team.

Finetuning pretrained protein encoder for node classification tasks.
'''

import os
import yaml
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from utils import set_seed
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

class ProteinNet_pretrain(object):
    def __init__(self, dataset, config):
        self.pConfig = config
        self.pDataset = dataset
        self.pDevice = self._fGetDevice_()

    def _fGetDevice_(self):
        if torch.cuda.is_available() and self.pConfig['gpu'] != 'cpu':
            pDevice = self.pConfig['gpu']
            torch.cuda.set_device(pDevice)
        else:
            pDevice = 'cpu'
        print("[GraphConSeq Finetuning] Running on:", pDevice)
        return pDevice

    def _fGetGraphEncoder_(self, pGNNType, pGraphEncParam):
        # You can deploy any kind of graph encoders for further performance improvement;
        # Current implementation only includes GNNs, but Graphtransformers/GraphUNet will be the great starting point.
        if pGNNType == 'GCN':
            from models.GraphEncoder import GCNEnc
            mGraphEnc = GCNEnc(**pGraphEncParam)
        elif pGNNType == 'SAGE':
            from models.GraphEncoder import SAGEEnc
            mGraphEnc = SAGEEnc(**pGraphEncParam)
        elif pGNNType == 'GAT':
            from models.GraphEncoder import GATEnc
            mGraphEnc = GATEnc(**pGraphEncParam)
        elif pGNNType == 'GIN':
            from models.GraphEncoder import GINEnc
            mGraphEnc = GINEnc(**pGraphEncParam)
        else: 
            print(f"[ProteinNet Error] GNN Type is not appropriate!")
            return None
        return mGraphEnc


    def _step(self, mEmb, mEnc, dGraphs):
        x_, edge_index, edge_attr_, batch = dGraphs.x, dGraphs.edge_index, dGraphs.edge_attr, dGraphs.batch
        x, edge_attr = mEmb(x_, edge_attr_)
        pred = mEnc(x, edge_index, edge_attr, batch)
        return pred


    def train(self):
        # Import dataloaders.
        print("[ProteinNet Pretraining] Importing Dataloaders...")
        self.criterion = torch.nn.CrossEntropyLoss()
        pTrainLoader = self.pDataset.get_data_loaders()
        print(f"[ProteinNet Pretraining] Imported Dataloaders...")

        # Import Graph Encoder and Graph Embedder.
        print("[ProteinNet Pretraining] Importing GraphEncoder...")        
        pEmbConfig = self.pConfig['embed']
        pGraphConfig = self.pConfig['graph_encoder']

        pGraphEmbParam = {"pDimNodeEmb": pEmbConfig['init_node_dim'], "pDimEdgeEmb": pEmbConfig['init_edge_dim']}
        pGraphEncParam = {"pNumLayers": pGraphConfig['num_layers'], "pDim": pEmbConfig['init_node_dim'],
                          "pHDim": pGraphConfig['hidden_dim'], "pDropRatio": pGraphConfig['drop_ratio'], "pNumLabels": None}
        from models.GraphEncoder import GraphEmb
        mGraphEmb = GraphEmb(**pGraphEmbParam)
        mGraphEnc = self._fGetGraphEncoder_(self.pConfig['graph_encoder_type'],pGraphEncParam)

        mGraphEmb.to(self.pDevice)
        mGraphEnc.to(self.pDevice)
        
        print("[ProteinNet Pretraining] GraphEncoder Successfully imported!")
        print(f"[ProteinNet Pretraining] Number of Parameters: {sum(p.numel() for p in mGraphEmb.parameters() if p.requires_grad) + sum(p.numel() for p in mGraphEnc.parameters() if p.requires_grad)}")
        print("---------------------------------------------------------------")
            
        # Setup Optimizer.
        pOptimizer = torch.optim.Adam(
            list(mGraphEmb.parameters()) + list(mGraphEnc.parameters()), self.pConfig['finetune']['init_lr'],
            weight_decay=eval(self.pConfig['finetune']['weight_decay'])
        ) 
        pOptimizer.param_groups[0]['capturable'] = True

        pScheduler = CosineAnnealingLR(
            pOptimizer, T_max=self.pConfig['finetune']['epochs'], #-self.pConfig['warm_up'], 
            eta_min=0, last_epoch=-1
        )

        # Perform Pretraining.
        print("[Pretraining Pretraining] Pretraining Begins.")
        
        pBestValLoss = np.inf
        pBestMetrics = []
        pMetrics = []
        for pEpochCounter in range(self.pConfig['epochs']):
            print(f"[GraphConSeq Pretraining] Epoch {pEpochCounter} started.")
            pLosses = self._Epoch_(mConSeq, pTrainLoader, pValidLoader, pOptimizer, pEpochCounter)

            if pLosses[1][1] < pBestValLoss:
                pBestValLoss = pLosses[1][1]
                pBestLosses = pLosses
                if self.pConfig['finetune']['output_model_dir'] == 'None':
                    pOutDir = f"results/{self.pConfig['experiment_name']}/{self.pConfig['finetune']['task']}"
                else:
                    pOutDir = f"results/{self.pConfig['finetune']['output_model_dir']}"
                os.makedirs(pOutDir, exist_ok=True)

                # Save the best model.
                pBestModelDir = f"{pOutDir}/model_best.pth"
                pBestModelParam = {"mConSeq": mConSeq.state_dict()}
                torch.save(pBestModelParam, pBestModelDir)
                # Save the best metrics.
                pBestMetricDir = f"{pOutDir}/metrics_best.pickle"
                with open(pBestMetricDir, 'wb') as f:
                    pickle.dump(pMetrics, f)
                # Save the current configuration.
                pCurConfigDir = f"{pOutDir}/config.pickle"
                with open(pCurConfigDir, 'wb') as f:
                    pickle.dump(self.pConfig, f)

            for bn, (sequences, graphs) in enumerate(tqdm(pTrainLoader, total=len(pTrainLoader))):
                pOptimizer.zero_grad()
                sequences = sequences.to(self.pDevice)
                graphs = graphs.to(self.pDevice)

                # Train the model
                loss, con_loss, seq_loss = self._step(mConSeq, sequences, graphs)
                
                loss.backward()
                pOptimizer.step()
                n_iter += 1
                if n_iter % self.pConfig['log_every_n_steps'] == 0 and n_iter > 0:
                    print(f"epoch:{pEpochCounter+1}, step:{bn+1}, total_loss:{float(loss):.3f}, loss_con: {float(con_loss):.3f}, loss_seq: {float(seq_loss):.3f}")

            print(f"[GraphConSeq] Best Loss | Train={pBestMetrics[0]:.3f}, Val={pBestMetrics[1]:.3f}")
            

    def _Epoch_(self, mConSeq, pTrainLoader, pValidLoader, pOptimizer, pEpochCounter):
        # Train Cycle.
        pTrainLosses = self._train_(mConSeq, pTrainLoader, pOptimizer, pEpochCounter)
        
        #Evaluation Cycle.
        pValidLosses = self._eval_(mConSeq, pValidLoader)
        pLosses = [pTrainLosses, pValidLosses]
        return pLosses

    def _train_(self, mConSeq, pTrainLoader, pOptimizer, pEpochCounter):
        mConSeq.train()
        pTotalLoss, pTotalConLoss, pTotalSeqLoss = 0.0, 0.0, 0.0
        for pBatchNum, (dSeqs, dGraphs) in enumerate(tqdm(pTrainLoader, total=len(pTrainLoader))):
            dLabels = dGraphs.y
            dSeqs   = dSeqs.to(self.pDevice)
            dGraphs = dGraphs.to(self.pDevice)
            pOptimizer.zero_grad()

            pTrainLoss, pTrainConLoss, pTrainSeqLoss = self._step_(mConSeq, dSeqs, dGraphs)

            pTrainLoss.backward()
            pTotalLoss    += pTrainLoss.detach().item()
            pTotalConLoss += pTrainConLoss.detach().item()
            pTotalSeqLoss += pTrainSeqLoss.detach().item()
            pOptimizer.step()
            
            if pBatchNum % self.pConfig['log_every_n_steps'] == 0 and pBatchNum > 0:
                print(f"[GraphConSeq] Epoch: {pEpochCounter+1}, Step: {pBatchNum+1}, TotalLoss={float(pTrainLoss):.3f}")
        
        pTrainLoss = [pTotalLoss, pTotalConLoss, pTotalSeqLoss]
        return pTrainLoss

    def _eval_(self, mConSeq, pEvalLoader):
        mConSeq.eval()
        pTotalLoss, pTotalConLoss, pTotalSeqLoss = 0.0, 0.0, 0.0
        for pBatchNum, (dSeqs, dGraphs) in enumerate(tqdm(pEvalLoader, total=len(pEvalLoader))):
            dLabels = dGraphs.y
            dSeqs   = dSeqs.to(self.pDevice)
            dGraphs = dGraphs.to(self.pDevice)
            with torch.no_grad():
                pEvalLoss, pEvalConLoss, pEvalSeqLoss = self._step_(mConSeq, dSeqs, dGraphs)
            pTotalLoss    += pEvalLoss.detach().item()
            pTotalConLoss += pEvalConLoss.detach().item()
            pTotalSeqLoss += pEvalSeqLoss.detach().item()
            
        pEvalLoss = [pTotalLoss, pTotalConLoss, pTotalSeqLoss]
        return pEvalLoss

def main():
    parser = argparse.ArgumentParser(description='[ProteinNet Pretraining]')
    parser.add_argument('--config', default="config", help='path to the config file.')
    args = parser.parse_args()
    
    config = yaml.load(open(f"config/{args.config}.yaml", "r"), Loader=yaml.FullLoader)
    
    print(f"---------------------------------------------------------------")
    set_seed(config['seed'])
    torch.set_num_threads(config['num_workers'])

    data_root = f"../data/pretrain/{config['finetune']['task']}"
    __data_config = {
        "data_path" : f"{data_root}",
        "num_workers" : 0,
        "valid_size" : 0.05,
    }
    batch_size = int(config['batch_size'])
    dataset = config['dataset']

    from datasets.dataset import ProteinDatasetWrapper
    dataset = ProteinDatasetWrapper(batch_size, **__data_config)
    model = ProteinNet_pretrain(dataset, config)
    model.train()

if __name__ == '__main__':
    main()