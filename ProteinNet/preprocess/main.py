'''
Licensed under the MIT License.

Copyright (c) ProteinNet Team.

Convert a single protein structure file (.pdb) to the pyg graph object.
'''

import numpy as np
import torch
import argparse
from torch_geometric.data import Data


def PDB2graph(pPDBDir, pLevel):
    from oddt.toolkits.rdk import readfile
    from pyg_util import dense_to_sparse, to_dense_adj
    from ogb_util import mol_to_graph_data_obj_simple
    
    # Convert .pdb to ODDT and RDK object, respectively.
    vODDTobj    = next(readfile('pdb', pPDBDir))
    vRDKobj     = vODDTobj.Mol
    
    # Convert RDK object to the pyg graph. Following the convention of ogb.
    vRDKGraph   = mol_to_graph_data_obj_simple(vRDKobj)

    # Some bizarre structures have missing edges.
    if vRDKGraph.edge_index.shape[1] == 0:
        print(f"edge cannot be parsed: {pPDBDir}")
        return None

    try:
        # Get the covalent bond edges. Following the convention of ogb.
        vCovBondAdj      = to_dense_adj(vRDKGraph.edge_index, edge_attr=vRDKGraph.edge_attr)

        # Get the non-covalent bond edges.
        from util import fODDT2NonCovAdjAtom
        vNonCovBondAdj   = fODDT2NonCovAdjAtom(vODDTobj)

        # Sometimes ODDT object has additional C at the last.
        if len(vCovBondAdj[0]) != len(vNonCovBondAdj[0]): vNonCovBondAdj = vNonCovBondAdj[:, :-1, :-1]

        vBondAdj         = torch.concat([vCovBondAdj, vNonCovBondAdj], dim=-1)
        vEdgeIndex, vEdgeAttr = dense_to_sparse(vBondAdj)
        vProGraph = Data(x=vRDKGraph.x, edge_index=vEdgeIndex, edge_attr=vEdgeAttr.long())
    except:
        print(f"bond size does not match: {pPDBDir}")
        return None

    # Get the AA sequence.
    from bio import residue_tChars
    sequence = []
    for res in vODDTobj.residues: sequence.append(residue_tChars.index(res.name.upper()))
    vProGraph.y = torch.tensor(sequence, dtype=torch.long)
    
    return vProGraph


if __name__ == "__main__":
    # Set the maximum working threads. 
    # Without this, a large number of threads will be generated and hamper the throughput.
    torch.set_num_threads(2)

    parser = argparse.ArgumentParser(description="Preprocess a bulk pdb structures.")
    parser.add_argument('--dirstr',   type=str, help='the directory where structure exists.')
    parser.add_argument('--dirout',   type=str, help='the directory where the graph will be saved.')
    parser.add_argument('--level',      type=str, help='the level of graph construction', default='aa')
    args = parser.parse_args()
    
    ProGraph = PDB2graph(args.dirstr, args.level)
    if ProGraph is not None: torch.save(ProGraph, args.dirout)

    import os
    os._exit(os.EX_OK)