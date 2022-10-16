# ProteinNet

This is a working repository for ProteinNet, a machine learning model aimming to learn general representation of proteins via multimodal contrastive learning between structures and sequences.

Please make sure that your code "Should be executable" without any kind of bugs, before the git push argument.

The following will explain how to begin your project.

## Executing Pretraining Network.
    # current dir: ProteinNet/ProteinNet
    $ python3 -m finetune_node --config eTest

## Data Download and Preprocessing.
### Pretraining: MANE
1. Download the full entries of MANE dataset from [here](https://alphafold.ebi.ac.uk/download#mane-section). <br>
    `mkdir data/pretrain/mane/raw`<br>
    `cd data/pretrain/mane/raw` <br>
    `wget https://ftp.ebi.ac.uk/pub/databases/alphafold/v3/mane_overlap_v3.tar`

2. Place the dataset at data/pretrain/mane folder, then unzip, rename the directory as 'raw' <br>
    `tar -xvf mane_overlap_v3.tar raw`

3. Preprocessing PDB structures using our pretraining code. <br>
    ` # current dir: data/pretrain/mane/raw`<br>
    `cd ../../../../ProteinNet`<br>
    `python preprocess.py --dataname mane --level aa --process 8`

### Pretraining: Swissprot
1. Download the full entries of Swissprot dataset from [here](https://alphafold.ebi.ac.uk/download#swissprot-section). <br>
    `mkdir data/pretrain/swissprot/raw`<br>
    `cd data/pretrain/swissprot/raw` <br>
    `wget https://ftp.ebi.ac.uk/pub/databases/alphafold/v3/swissprot_pdb_v3.tar`

2. Place the dataset at data/pretrain/mane folder, then unzip, rename the directory as 'raw' <br>
    `tar -xvf swissprot_pdb_v3.tar raw`

3. Preprocessing PDB structures using our preprocessing code. <br>
    ` # current dir: data/pretrain/swissprot/raw`<br>
    `cd ../../../../ProteinNet`<br>
    `python preprocess.py --dataname swissprot --level aa --process 8`

### Finetuning: DNA/ATP binding site prediction.
Raw data for DNA/ATP binding site prediction is already provided through the github.<br>
But their file format is not appropriate for our purpose - we should convert them into the structure (.pdb) files.

Cause most of developers are not familer to AlphaFold2, I plan to run them instead.<br> 
Currently DNA binding site prediction data is available by me; please contact me to download it.

1. Place the entries of DNABind dataset.<br>
    `mkdir data/finetune/dnabind/raw`<br>
    `cd data/finetune/dnabind/raw`<br>
    ` # Locate the data`

2. Preprocessing PDB structures using our preprocessing code. <br>
    ` # current dir: data/finetune/dna/raw`<br>
    `cd ../../../../ProteinNet`<br>
    `python preprocess.py --dataname dna --level aa --process 8`

