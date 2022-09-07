'''
Licensed under the MIT License.

Copyright (c) ProteinNet Team.

Convert protein structure file (.pdb) to the pyg graph object.
'''

# MANE: python preprocess.py --dataname mane --level atom --process 8


import os
import argparse
import subprocess
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess a bulk pdb structures.")
    parser.add_argument('--dataname',   type=str, help='the name of pretraining pdb dataset', default='mane')
    parser.add_argument('--level',      type=str, help='the level of graph construction', default='aa')
    parser.add_argument('--process',    type=int, help='the number of subprocesses', default=8)

    args = parser.parse_args()

    # Define generic directories.
    dirRoot = f"../data/pretrain/{args.dataname}"
    dirStr  = f"{dirRoot}/raw"
    dirOut  = f"{dirRoot}/processed"
    os.makedirs(dirOut, exist_ok=True)
    
    processes = set()
    for (root, dirs, files) in os.walk(dirStr):
        if len(files) > 0:
            print(f"[ProteinNet Preprocessing] Processing total {len(files)} files.")
            for (i, file_name) in enumerate(tqdm(files)):
                # not a pdb file
                if '.pdb' not in file_name: continue

                # Define specific directories.
                codePDB = file_name.split('.')[0]
                _dirStr = dirStr + '/' + codePDB + '.pdb'
                _dirOut = dirOut + '/' + codePDB + '.pt'
                
                # already preprocessed; skip this.
                if os.path.exists(_dirOut): continue

                # Multiprocessing
                processes.add(subprocess.Popen([f"python preprocess/main.py --dirstr {_dirStr} --dirout {_dirOut} --level {args.level}"], shell=True))
                if len(processes) == args.process:
                    os.wait()
                    processes.difference_update([p for p in processes if p.poll() is not None])