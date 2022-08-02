'''
Licensed under the MIT License.

Copyright (c) ProteinNet Team.

Graph node and edge features for ProteinNet.
'''

import torch
import itertools
from oddt.interactions import hbonds, halogenbonds, pi_stacking, salt_bridges, hydrophobic_contacts, pi_cation

allowable_noncovalent_features={
    'possible_noncovalent_bond_type_list' : [
        'NONE', 
        'HYDROGEN',
        'HALOGEN',
        'PI-PI_STACKING',
        'SALT_BRIDGE',
        'HYDROPHOBIC',
        'PI_CATION',
        'misc'
    ],
    'possible_strict_list' : [
        'NONE',
        'NOTSTRICT'
        'STRICT',
        'PARALLEL',
        'PERPENDICULAR',
        'MISC'
    ]
}

def fODDT2NonCovAdj(pODDTObj):
    """
        Calculates non-covalent bonds from an oddt protein object.
        Input: ODDT object of a protein
        Return: Adjacency matrix [N, N, D]
            - N: # Atoms
            - D: # Bond types (default = 6)
    """
    # Build dictionary mapping centroid x coordinate to atom indices.
    pRingDicts = {}
    for (i, pRingDict) in enumerate(pODDTObj.ring_dict):
        pRingDicts[pRingDict['centroid'][0]] = pODDTObj.sssr[i]

    # Define the dense adjacency map.
    pNumAtoms = len(pODDTObj.atoms)
    pDenseAdj = torch.zeros(pNumAtoms, pNumAtoms, 2)
    
    # Type1: Hydrogen bondings
    pHyD, pHyA, pHyS = hbonds(pODDTObj, pODDTObj) # HB Donor index, HB Acceptor index, is_strict
    if not len(pHyD) == 0:
        for i in range(len(pHyD)):
            vIsStrict = 'NOTSTRICT'
            if pHyS[i]: 'STRICT'
            pDenseAdj[pHyD[i][0], pHyA[i][0], :] = \
                torch.Tensor([allowable_noncovalent_features['possible_noncovalent_bond_type_list'].index('HYDROGEN'), 
                              allowable_noncovalent_features['possible_strict_list'].index(vIsStrict)])

    # Type2: Halogen bondings
    pHaD, pHaA, pHaS = halogenbonds(pODDTObj, pODDTObj) # Ha Donor index, Ha Acceptor index, is_strict
    if not len(pHaD) == 0:
        for i in range(len(pHaD)):
            vIsStrict = 'NOTSTRICT'
            if pHaS[i]: 'STRICT'
            pDenseAdj[pHaD[i][0], pHaA[i][0], :] = \
                torch.Tensor([allowable_noncovalent_features['possible_noncovalent_bond_type_list'].index('HALOGEN'), 
                              allowable_noncovalent_features['possible_strict_list'].index(vIsStrict)])
            
    # Type5: Pi-Pi stacking
    pPP1, pPP2, pPPpa, pPPpe = pi_stacking(pODDTObj, pODDTObj) # Pi-Pi stacking molecule 1/2, is_strict_parallel, is_strict_perpendicular
    if not len(pPP1) == 0:
        for i in range(len(pPP1)):
            pairs = [pRingDicts[pPP1[i]['centroid'][0]], pRingDicts[pPP2[i]['centroid'][0]]]
            for (a1, a2) in list(itertools.product(*pairs)):
                vIsStrict = 'NOTSTRICT'
                if pPPpa[i]: 'PARALLEL'
                elif pPPpe[i]: 'PERPENDICULAR'
                pDenseAdj[a1, a2, :] = \
                torch.Tensor([allowable_noncovalent_features['possible_noncovalent_bond_type_list'].index('PI-PI_STACKING'), 
                              allowable_noncovalent_features['possible_strict_list'].index(vIsStrict)])

    # Type6: Salt bridges
    pSaC, pSaA = salt_bridges(pODDTObj, pODDTObj) # Cationic mol, Anionic mol
    if not len(pSaC) == 0:
        for i in range(len(pSaC)):
            pDenseAdj[pSaC[i][0], pSaA[i][0], :] = \
                torch.Tensor([allowable_noncovalent_features['possible_noncovalent_bond_type_list'].index('SALT_BRIDGE'), 
                              allowable_noncovalent_features['possible_strict_list'].index('NONE')])

    # Type7: Hydrophobic contacts
    pHp1, pHp2 = hydrophobic_contacts(pODDTObj, pODDTObj) # mol1, mol2
    if not len(pHp1) == 0:
        for i in range(len(pHp1)):
            pDenseAdj[pHp1[i][0], pHp2[i][0], :] = \
                torch.Tensor([allowable_noncovalent_features['possible_noncovalent_bond_type_list'].index('HYDROPHOBIC'), 
                              allowable_noncovalent_features['possible_strict_list'].index('NONE')])

    # Type8: Pi-Cation interactions
    pPCP, pPCC, pPCS = pi_cation(pODDTObj, pODDTObj) # Ring, Cation, is_Strict
    if not len(pPCP) == 0:
        for i in range(len(pPCP)):
            for atom in pRingDicts[pPCP[i]['centroid'][0]]:
                vIsStrict='NOTSTRICT'
                if pPCS[i]: vIsStrict = 'STRICT'
                pDenseAdj[atom, pPCC[i][0], :] = \
                torch.Tensor([allowable_noncovalent_features['possible_noncovalent_bond_type_list'].index('PI_CATION'), 
                              allowable_noncovalent_features['possible_strict_list'].index(vIsStrict)])

    return pDenseAdj.unsqueeze(0)