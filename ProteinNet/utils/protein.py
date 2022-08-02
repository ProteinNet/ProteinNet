'''
Licensed under the MIT License.

Copyright (c) ProteinNet Team.

Build protein graphs from the ODDT object.
'''

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj

from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector
from oddt.toolkits.rdk import readfile

from ProteinNet.utils.features import fODDT2NonCovAdj
from ProteinNet.utils.graph_util import dense_to_sparse


def mol_to_graph_data_obj_simple(mol):
    """ used in MoleculeDataset() class
    Converts rdkit mol objects to graph data object in pytorch geometric
    NB: Uses simplified atom and bond features, and represent as indices
    :param mol: rdkit mol object
    :return: graph data object with the attributes: x, edge_index, edge_attr """

    # atoms
    # num_atom_features = 2  # atom type, chirality tag
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = atom_to_feature_vector(atom)
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    if len(mol.GetBonds()) <= 0:  # mol has no bonds
        num_bond_features = 2  # bond type & direction
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)
    else:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = bond_to_feature_vector(bond)
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data


def PDB2graph(pPDBDir):
    vODDTobj    = next(readfile("pdb", pPDBDir))
    vRDKobj     = vODDTobj.Mol
    vRDKGraph   = mol_to_graph_data_obj_simple(vRDKobj)

    vCovBondAdj      = to_dense_adj(vRDKGraph.edge_index, edge_attr=vRDKGraph.edge_attr)
    vNonCovBondAdj   = fODDT2NonCovAdj(vODDTobj)
    vBondAdj         = torch.concat([vCovBondAdj, vNonCovBondAdj], dim=-1)
    
    vEdgeIndex, vEdgeAttr = dense_to_sparse(vBondAdj)
    
    vProGraph = Data(x=vRDKGraph.x, edge_index=vEdgeIndex, edge_attr=vEdgeAttr)

    return vProGraph