'''
Licensed under the MIT License.

Copyright (c) ProteinNet Team.

Calculate and annotate non-covalent bondings between atoms, from ODDT object.
'''

import torch
import itertools
from features import allowable_noncovalent_features
from oddt.interactions import hbonds, halogenbonds, pi_stacking, salt_bridges, hydrophobic_contacts, pi_cation

def fODDT2NonCovAdjAtom(pODDTObj):
    """
        Calculates non-covalent bonds from an oddt protein object.
        Input: ODDT object of a protein
        Return: Adjacency matrix [N, N, D]
            - N: # Atoms
            - D: # Bond types (default = 6)
    """
    # Build dictionary mapping centroid x coordinate to atom indices.
    vRingDicts = {}
    for (i, vRingDict) in enumerate(pODDTObj.ring_dict):
        vRingDicts[vRingDict['centroid'][0]] = pODDTObj.sssr[i]

    # Define the dense adjacency map.
    vNumAtoms = len(pODDTObj.atoms)
    vDenseAdj = torch.zeros(vNumAtoms, vNumAtoms, 2)
    
    # Type1: Hydrogen bondings
    vHyD, vHyA, vHyS = hbonds(pODDTObj, pODDTObj) # HB Donor index, HB Acceptor index, is_strict
    if not len(vHyD) == 0:
        for i in range(len(vHyD)):
            vIsStrict = 'NOTSTRICT'
            if vHyS[i]: 'STRICT'
            vDenseAdj[vHyD[i][0], vHyA[i][0], :] = \
                torch.Tensor([allowable_noncovalent_features['possible_noncovalent_bond_type_list'].index('HYDROGEN'), 
                              allowable_noncovalent_features['possible_strict_list'].index(vIsStrict)])

    # Type2: Halogen bondings
    vHaD, vHaA, vHaS = halogenbonds(pODDTObj, pODDTObj) # Ha Donor index, Ha Acceptor index, is_strict
    if not len(vHaD) == 0:
        for i in range(len(vHaD)):
            vIsStrict = 'NOTSTRICT'
            if vHaS[i]: 'STRICT'
            vDenseAdj[vHaD[i][0], vHaA[i][0], :] = \
                torch.Tensor([allowable_noncovalent_features['possible_noncovalent_bond_type_list'].index('HALOGEN'), 
                              allowable_noncovalent_features['possible_strict_list'].index(vIsStrict)])
            
    # Type5: Pi-Pi stacking
    vPP1, vPP2, vPPpa, vPPpe = pi_stacking(pODDTObj, pODDTObj) # Pi-Pi stacking molecule 1/2, is_strict_parallel, is_strict_perpendicular
    if not len(vPP1) == 0:
        for i in range(len(vPP1)):
            pairs = [vRingDicts[vPP1[i]['centroid'][0]], vRingDicts[vPP2[i]['centroid'][0]]]
            for (a1, a2) in list(itertools.product(*pairs)):
                vIsStrict = 'NOTSTRICT'
                if vPPpa[i]: 'PARALLEL'
                elif vPPpe[i]: 'PERPENDICULAR'
                vDenseAdj[a1, a2, :] = \
                torch.Tensor([allowable_noncovalent_features['possible_noncovalent_bond_type_list'].index('PI-PI_STACKING'), 
                              allowable_noncovalent_features['possible_strict_list'].index(vIsStrict)])

    # Type6: Salt bridges
    vSaC, vSaA = salt_bridges(pODDTObj, pODDTObj) # Cationic mol, Anionic mol
    if not len(vSaC) == 0:
        for i in range(len(vSaC)):
            vDenseAdj[vSaC[i][0], vSaA[i][0], :] = \
                torch.Tensor([allowable_noncovalent_features['possible_noncovalent_bond_type_list'].index('SALT_BRIDGE'), 
                              allowable_noncovalent_features['possible_strict_list'].index('NONE')])

    # Type7: Hydrophobic contacts
    vHp1, vHp2 = hydrophobic_contacts(pODDTObj, pODDTObj) # mol1, mol2
    if not len(vHp1) == 0:
        for i in range(len(vHp1)):
            vDenseAdj[vHp1[i][0], vHp2[i][0], :] = \
                torch.Tensor([allowable_noncovalent_features['possible_noncovalent_bond_type_list'].index('HYDROPHOBIC'), 
                              allowable_noncovalent_features['possible_strict_list'].index('NONE')])

    # Type8: Pi-Cation interactions
    vPCP, vPCC, vPCS = pi_cation(pODDTObj, pODDTObj) # Ring, Cation, is_Strict
    if not len(vPCP) == 0:
        for i in range(len(vPCP)):
            for atom in vRingDicts[vPCP[i]['centroid'][0]]:
                vIsStrict='NOTSTRICT'
                if vPCS[i]: vIsStrict = 'STRICT'
                vDenseAdj[atom, vPCC[i][0], :] = \
                torch.Tensor([allowable_noncovalent_features['possible_noncovalent_bond_type_list'].index('PI_CATION'), 
                              allowable_noncovalent_features['possible_strict_list'].index(vIsStrict)])

    return vDenseAdj.unsqueeze(0)

    
def fODDT2NonCovAdjAA(pODDTObj, pAtom2Res):
    """
        Calculates non-covalent bonds from an oddt protein object.
        Input: ODDT object of a protein
        Return: Adjacency matrix [N, N, D]
            - N: # Atoms
            - D: # Bond types (default = 6)
    """
    # Build dictionary mapping centroid x coordinate to atom indices.
    vRingDicts = {}
    for (i, vRingDict) in enumerate(pODDTObj.ring_dict):
        vRingDicts[vRingDict['centroid'][0]] = pODDTObj.sssr[i]

    

    # Define the dense adjacency map.
    vNumAA    = len(pODDTObj.residues)
    vDenseAdj = torch.zeros(vNumAA, vNumAA, 2)
    
    # Type1: Hydrogen bondings
    vHyD, vHyA, vHyS = hbonds(pODDTObj, pODDTObj) # HB Donor index, HB Acceptor index, is_strict
    if not len(vHyD) == 0:
        for i in range(len(vHyD)):
            vIsStrict = 'NOTSTRICT'
            if vHyS[i]: 'STRICT'
            try:
                vDenseAdj[pAtom2Res[vHyD[i][0]], pAtom2Res[vHyA[i][0]], :] = \
                    torch.Tensor([allowable_noncovalent_features['possible_noncovalent_bond_type_list'].index('HYDROGEN'), 
                                allowable_noncovalent_features['possible_strict_list'].index(vIsStrict)])
            except:
                continue

    # Type2: Halogen bondings
    vHaD, vHaA, vHaS = halogenbonds(pODDTObj, pODDTObj) # Ha Donor index, Ha Acceptor index, is_strict
    if not len(vHaD) == 0:
        for i in range(len(vHaD)):
            try:
                vIsStrict = 'NOTSTRICT'
                if vHaS[i]: 'STRICT'
                vDenseAdj[pAtom2Res[vHaD[i][0]], pAtom2Res[vHaA[i][0]], :] = \
                    torch.Tensor([allowable_noncovalent_features['possible_noncovalent_bond_type_list'].index('HALOGEN'), 
                                allowable_noncovalent_features['possible_strict_list'].index(vIsStrict)])
            except: continue
            
    # Type5: Pi-Pi stacking
    vPP1, vPP2, vPPpa, vPPpe = pi_stacking(pODDTObj, pODDTObj) # Pi-Pi stacking molecule 1/2, is_strict_parallel, is_strict_perpendicular
    if not len(vPP1) == 0:
        for i in range(len(vPP1)):
            pairs = [vRingDicts[vPP1[i]['centroid'][0]], vRingDicts[vPP2[i]['centroid'][0]]]
            for (a1, a2) in list(itertools.product(*pairs)):
                try:
                    vIsStrict = 'NOTSTRICT'
                    if vPPpa[i]: 'PARALLEL'
                    elif vPPpe[i]: 'PERPENDICULAR'
                    vDenseAdj[pAtom2Res[a1], pAtom2Res[a2], :] = \
                    torch.Tensor([allowable_noncovalent_features['possible_noncovalent_bond_type_list'].index('PI-PI_STACKING'), 
                                allowable_noncovalent_features['possible_strict_list'].index(vIsStrict)])
                except: continue

    # Type6: Salt bridges
    vSaC, vSaA = salt_bridges(pODDTObj, pODDTObj) # Cationic mol, Anionic mol
    if not len(vSaC) == 0:
        for i in range(len(vSaC)):
            try:
                vDenseAdj[pAtom2Res[vSaC[i][0]], pAtom2Res[vSaA[i][0]], :] = \
                    torch.Tensor([allowable_noncovalent_features['possible_noncovalent_bond_type_list'].index('SALT_BRIDGE'), 
                                allowable_noncovalent_features['possible_strict_list'].index('NONE')])
            except:
                continue

    # Type7: Hydrophobic contacts
    vHp1, vHp2 = hydrophobic_contacts(pODDTObj, pODDTObj) # mol1, mol2
    if not len(vHp1) == 0:
        for i in range(len(vHp1)):
            try:
                vDenseAdj[pAtom2Res[vHp1[i][0]], pAtom2Res[vHp2[i][0]], :] = \
                    torch.Tensor([allowable_noncovalent_features['possible_noncovalent_bond_type_list'].index('HYDROPHOBIC'), 
                                allowable_noncovalent_features['possible_strict_list'].index('NONE')])
            except: continue

    # Type8: Pi-Cation interactions
    vPCP, vPCC, vPCS = pi_cation(pODDTObj, pODDTObj) # Ring, Cation, is_Strict
    if not len(vPCP) == 0:
        for i in range(len(vPCP)):
            for atom in vRingDicts[vPCP[i]['centroid'][0]]:
                try:
                    vIsStrict='NOTSTRICT'
                    if vPCS[i]: vIsStrict = 'STRICT'
                    vDenseAdj[pAtom2Res[atom], pAtom2Res[vPCC[i][0]], :] = \
                    torch.Tensor([allowable_noncovalent_features['possible_noncovalent_bond_type_list'].index('PI_CATION'), 
                                allowable_noncovalent_features['possible_strict_list'].index(vIsStrict)])
                except: continue

    return vDenseAdj.unsqueeze(0)
