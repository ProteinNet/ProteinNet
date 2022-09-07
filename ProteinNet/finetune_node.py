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
from sklearn.metrics import accuracy_score, f1_score, average_precision_score, roc_auc_score


class ProteinNet_finetune(object):
    def __init__(self, dataset, config):
        self.pConfig = config
        self.pDataset = dataset
        self.pDevice = self._fGetDevice_()
        self.pTask = config['finetune']['task']

    def _fGetDevice_(self):
        if torch.cuda.is_available() and self.pConfig['gpu'] != 'cpu':
            pDevice = self.pConfig['gpu']
            torch.cuda.set_device(pDevice)
        else:
            pDevice = 'cpu'
        print("[ProteinNet Finetuning] Running on:", pDevice)
        return pDevice

    def _fGetGraphEncoder_(self, pGNNType, pGraphEncParam):
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
        print("[ProteinNet Finetuning] Importing Dataloaders...")
        pTrainLoader, pValidLoader, pTestLoader = None, None, None
        self.criterion = torch.nn.CrossEntropyLoss()
        if self.pTask == "dna":
            pTrainLoader, pTest129Loader, pTest181Loader = self.pDataset.get_data_loaders()
            pNumLabels = 2
        elif self.pTask == "atp":
            pTrainLoader, pTestLoader = self.pDataset.get_data_loaders()
            pNumLabels = 2
        elif self.pTask == "HomologyTAPE": 
            pTrainLoader, pValidLoader, pTestFamilyLoader, pTestFoldLoader, pTestSuperfamilyLoader = self.pDataset.get_data_loaders()
            pNumLabels = 1196
        elif self.pTask == "ProtFunct":
            pTrainLoader, pValidLoader, pTestLoader = self.pDataset.get_data_loaders()
            pNumLabels = 384
        elif self.pTask == "ProteinsDB":
            pTrainLoaders, pValidLoaders = self.pDataset.get_data_loaders()
            pNumLabels = 2
        print(f"[ProteinNet Finetuning] Imported {self.pTask} Dataloaders...")

        # Import Graph Encoder and Graph Embedder.
        print("[ProteinNet Finetuning] Importing GraphEncoder...")        
        pEmbConfig = self.pConfig['embed']
        pGraphConfig = self.pConfig['graph_encoder']

        pGraphEmbParam = {"pDimNodeEmb": (pEmbConfig['num_nodes'] - 1) * pEmbConfig['num_shells'], "pDimEdgeEmb": pEmbConfig['num_edges'],
                          "pDimNodeHidden": pGraphConfig['node_dim'], "pDimEdgeHidden": pGraphConfig['edge_dim']}
        pGraphEncParam = {"pNumLayers": pGraphConfig['num_layers'], "pDim": pGraphConfig['node_dim'],
                          "pHDim": pGraphConfig['hidden_dim'], "pDropRatio": pGraphConfig['drop_ratio'], "pNumLabels":pNumLabels}
        from models.GraphEncoder import GraphEmb
        mGraphEmb = GraphEmb(**pGraphEmbParam)
        mGraphEnc = self._fGetGraphEncoder_(self.pConfig['graph_encoder_type'],pGraphEncParam)
        
        # Import Pretrained Model
        pPretrainedModelDir = self.pConfig['finetune']['pretrained_model_dir']
        if pPretrainedModelDir == "None":
            # No pretrained model provided.
            print(f"[ProteinNet Finetuning] No pretrained model found: Learn from the scratch.")
        else:
            # Pretrained model provided.
            print(f"[ProteinNet Finetuning] Import pretrained model: {self.pConfig['finetune']['pretrained_model_dir']}")
            mGraphEmb.from_pretrained(self.pConfig['finetune']['pretrained_model_dir'])
            mGraphEnc.from_pretrained(self.pConfig['finetune']['pretrained_model_dir'])
        mGraphEmb.to(self.pDevice)
        mGraphEnc.to(self.pDevice)
        print("[ProteinNet Finetuning] GraphEncoder Successfully imported!")
        print(f"[ProteinNet Finetuning] Number of Parameters: {sum(p.numel() for p in mGraphEmb.parameters() if p.requires_grad) + sum(p.numel() for p in mGraphEnc.parameters() if p.requires_grad)}")
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
        
        # Perform Finetuning.
        print("[ProteinNet Finetuning] Finetuning Begins.")
        pBestValAcc = 0.0
        pBestMetrics = []
        pMetrics = []
        for pEpochCounter in range(self.pConfig['finetune']['epochs']):
            if self.pTask == "HomologyTAPE":
                pMetrics = self._NormEpoch_(mGraphEmb, mGraphEnc, pTrainLoader, pValidLoader, [pTestFoldLoader, pTestSuperfamilyLoader, pTestFamilyLoader], pOptimizer, pEpochCounter)
                print(f"[ProteinNet] Epoch: {pEpochCounter+1} Accuracy | Train={pMetrics[0]['Acc']:.3f}, Val={pMetrics[1]['Acc']:.3f}, TestFold={pMetrics[2]['Acc']:.3f}, TestSuperFamily={pMetrics[3]['Acc']:.3f}, TestFamily={pMetrics[4]['Acc']:.3f}")
            elif self.pTask == "ProtFunct":
                pMetrics = self._NormEpoch_(mGraphEmb, mGraphEnc, pTrainLoader, pValidLoader, [pTestLoader], pOptimizer, pEpochCounter)
                print(f"[ProteinNet] Epoch: {pEpochCounter+1} Accuracy | Train={pMetrics[0]['Acc']:.3f}, Val={pMetrics[1]['Acc']:.3f}, Test={pMetrics[2]['Acc']:.3f}")
            elif self.pTask == "ProteinsDB":
                pMetrics = self._kFoldEpoch_(mGraphEmb, mGraphEnc, pTrainLoaders, pValidLoaders, pOptimizer, pEpochCounter, pFold=10)
                print(f"[ProteinNet] Epoch: {pEpochCounter+1} Accuracy | Train={pMetrics[0]['Acc']:.3f}, Val={pMetrics[1]['Acc']:.3f}")

            if pMetrics[1]['Acc'] > pBestValAcc:
                pBestValAcc = pMetrics[1]['Acc']
                pBestMetrics = pMetrics
                if self.pConfig['finetune']['output_model_dir'] == 'None':
                    pOutDir = f"results/{self.pConfig['experiment_name']}/{self.pConfig['finetune']['task']}"
                else:
                    pOutDir = f"results/{self.pConfig['finetune']['output_model_dir']}"
                os.makedirs(pOutDir, exist_ok=True)
                # Save the best model.
                pBestModelDir = f"{pOutDir}/model_best.pth"
                pBestModelParam = {"GraphEnc": mGraphEnc.state_dict()}
                torch.save(pBestModelParam, pBestModelDir)
                # Save the best metrics.
                pBestMetricDir = f"{pOutDir}/metrics_best.pickle"
                with open(pBestMetricDir, 'wb') as f:
                    pickle.dump(pMetrics, f)
                # Save the current configuration.
                pCurConfigDir = f"{pOutDir}/config.pickle"
                with open(pCurConfigDir, 'wb') as f:
                    pickle.dump(self.pConfig, f)
        
        if self.pTask == "HomologyTAPE":
            print(f"[ProteinNet] Best Accuracy | Train={pBestMetrics[0]['Acc']:.3f}, Val={pBestMetrics[1]['Acc']:.3f}, TestFold={pBestMetrics[2]['Acc']:.3f}, TestSuperFamily={pBestMetrics[3]['Acc']:.3f}, TestFamily={pBestMetrics[4]['Acc']:.3f}")
        elif self.pTask == "ProtFunct":
            print(f"[ProteinNet] Best Accuracy | Train={pBestMetrics[0]['Acc']:.3f}, Val={pBestMetrics[1]['Acc']:.3f}, Test={pBestMetrics[2]['Acc']:.3f}")
        elif self.pTask == "ProteinsDB":
            print(f"[ProteinNet] Best Accuracy | Train={pBestMetrics[0]['Acc']:.3f}, Val={pBestMetrics[1]['Acc']:.3f}")
        print(f"[ProteinNet] Finetuning Finished.")

    def _CalcMetrics_(self, preds, labels):
        Acc     = accuracy_score(labels, preds)
        F1      = f1_score(labels, preds, average='weighted')
        # ROCAUC  = roc_auc_score(labels, preds)
        # Pre     = average_precision_score(labels, preds)

        metrics = {'Acc': Acc, 'F1': F1} #, 'ROCAUC': ROCAUC, 'Pre': Pre}
        return metrics
        
    def _NormEpoch_(self, mGraphEmb, mGraphEnc, pTrainLoader, pValidLoader, pTestLoaders, pOptimizer, pEpochCounter):
        # Train Cycle
        pTrainLoss, pTrainMetrics = self._train_(mGraphEmb, mGraphEnc, pTrainLoader, pOptimizer, pEpochCounter)
        
        # Evaluation Cycle.
        # pTrainMetrics = self._eval_(mGraphEmb, mGraphEnc, pTrainLoader, pEpochCounter) # Skip this for efficiency.
        pValidMetrics = self._eval_(mGraphEmb, mGraphEnc, pValidLoader, pEpochCounter)
        pMetrics = [pTrainMetrics, pValidMetrics]

        for pTestLoader in pTestLoaders:
            pTestMetrics = self._eval_(mGraphEmb, mGraphEnc, pTestLoader, pEpochCounter)
            pMetrics.append(pTestMetrics)        
        return pMetrics

    def _kFoldEpoch_(self, mGraphEmb, mGraphEnc, pTrainLoaders, pValidLoaders, pOptimizer, pEpochCounter, pFold=10):
        pTrainMetrics, pValidMetrics = [], []
        for fold in range(pFold):
            pTrainLoader, pValidLoader = pTrainLoaders[fold], pValidLoaders[fold]
            # Train Cycle
            self._train_(mGraphEmb, mGraphEnc, pTrainLoader, pOptimizer, pEpochCounter)
            # Evaluation Cycle.
            pTrainMetrics_ = self._eval_(mGraphEmb, mGraphEnc, pTrainLoader, pEpochCounter)
            pValidMetrics_ = self._eval_(mGraphEmb, mGraphEnc, pValidLoader, pEpochCounter)
            pTrainMetrics.append(pTrainMetrics_)
            pValidMetrics.append(pValidMetrics_)
        pMetrics = [pTrainMetrics, pValidMetrics]        
        return  pMetrics

    def _eval_(self, mEmb, mEnc, pEvalLoader, pEpochCounter):
        mEmb.eval()
        mEnc.eval()
        pTotalPred, pTotalLabel = [], []
        for pBatchNum, (dGraphs) in enumerate(tqdm(pEvalLoader, total=len(pEvalLoader))):
            dLabels = dGraphs.y
            dGraphs = dGraphs.to(self.pDevice)
            with torch.no_grad():
                pEvalPred = self._step(mEmb, mEnc, dGraphs)
            pTotalPred.append(pEvalPred.cpu())
            pTotalLabel += dLabels.tolist()

        pTotalPred = np.argmax(torch.concat(pTotalPred, dim=0).detach().numpy(), axis=-1)
        pMetrics = self._CalcMetrics_(pTotalPred, pTotalLabel)
        return pMetrics

    def _train_(self, mEmb, mEnc, pTrainLoader, pOptimizer, pEpochCounter):
        mEmb.train()
        mEnc.train()
        pTotalLoss = 0.0
        pTotalPred, pTotalLabel = [], []
        for pBatchNum, (dGraphs) in enumerate(tqdm(pTrainLoader, total=len(pTrainLoader))):
            dLabels = dGraphs.y
            dGraphs = dGraphs.to(self.pDevice)
            pOptimizer.zero_grad()
            
            pTrainPred = self._step(mEmb, mEnc, dGraphs)
            pTrainLoss = self.criterion(pTrainPred.squeeze(), dLabels)

            pTrainLoss.backward()
            pOptimizer.step()
            pTotalLoss += pTrainLoss.detach().item()
            
            pTotalPred.append(pTrainPred.cpu())
            pTotalLabel += dLabels.tolist()
            if pBatchNum % self.pConfig['log_every_n_steps'] == 0 and pBatchNum > 0:
                print(f"[ProteinNet] Epoch: {pEpochCounter+1}, Step: {pBatchNum+1}, TotalLoss={float(pTrainLoss):.3f}")
            
        pTotalPred = np.argmax(torch.concat(pTotalPred, dim=0).detach().numpy(), axis=-1)
        pMetrics = self._CalcMetrics_(pTotalPred, pTotalLabel)
        return pTotalLoss, pMetrics

def main():
    parser = argparse.ArgumentParser(description='[ProteinNet Finetuning]')
    parser.add_argument('--config', default="config", help='path to the config file.')
    args = parser.parse_args()
    
    config = yaml.load(open(f"config/{args.config}.yaml", "r"), Loader=yaml.FullLoader)

    print(f"---------------------------------------------------------------")
    set_seed(config['finetune']['seed'])
    torch.set_num_threads(config['num_workers'])

    data_root = f"../data/finetune/{config['finetune']['task']}"
    __data_config = {
        "data_path" : f"{data_root}",
        "num_workers" : 0,
        "valid_size": 0.05,
        "pTAPE": config['finetune']['isTAPE']
    }
    batch_size = int(config['finetune']['batch_size'])
    curTask = config['finetune']['task']
    
    if curTask == "HomologyTAPE":
        from datasets.dataset_HomologyTAPE import ProtClassHomologyTAPEDatasetWrapper
        dataset = ProtClassHomologyTAPEDatasetWrapper(batch_size, **__data_config)
    elif curTask == "ProteinsDB":
        from datasets.dataset_ProteinsDB import ProtClassProteinsDBDatasetWrapper
        dataset = ProtClassProteinsDBDatasetWrapper(batch_size, **__data_config)
    elif curTask == "ProtFunct":
        from datasets.dataset_ProtFunct import ProtClassProtFunctDatasetWrapper
        dataset = ProtClassProtFunctDatasetWrapper(batch_size, **__data_config)
    else:
        print(f"[ProteinNet Error!] Provided task name is not appropriate.")
        return

    model = ProteinNet_finetune(dataset, config)
    model.train()

if __name__ == '__main__':
    main()