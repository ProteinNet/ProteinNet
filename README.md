# ProteinNet

This is a working repository for ProteinNet, a machine learning model aimming to learn general representation of proteins via multimodal contrastive learning between structures and sequences.

Please make sure that your code "Should be executable" without any kind of bugs, before the git push argument.

The following will explain how to begin your project.


## Data Download and Preprocessing.
### Pretraining: MANE
1. Download the full entries of MANE dataset from [here](https://alphafold.ebi.ac.uk/download#mane-section). <br>
    `mkdir data/pretrain/mane`<br>
    `cd data/pretrain/mane` <br>
    `wget https://ftp.ebi.ac.uk/pub/databases/alphafold/v3/mane_overlap_v3.tar`

2. Place the dataset at data/pretrain/mane folder, then unzip, rename the directory as 'raw' <br>
    `tar -xvf mane_overlap_v3.tar raw`

3. Preprocessing PDB structures using our pretraining code. <br>
    ` # current dir: data/pretrain/mane`<br>
    `cd ../../../ProteinNet`<br>
    `python preprocess.py --dataname mane --level atom --process 8`

### Pretraining: Swissprot
1. Download the full entries of Swissprot dataset from [here](https://alphafold.ebi.ac.uk/download#swissprot-section). <br>
    `mkdir data/pretrain/swissprot`<br>
    `cd data/pretrain/swissprot` <br>
    `wget https://ftp.ebi.ac.uk/pub/databases/alphafold/v3/swissprot_pdb_v3.tar`

2. Place the dataset at data/pretrain/mane folder, then unzip, rename the directory as 'raw' <br>
    `tar -xvf swissprot_pdb_v3.tar raw`

3. Preprocessing PDB structures using our pretraining code. <br>
    ` # current dir: data/pretrain/mane`<br>
    `cd ../../../ProteinNet`<br>
    `python preprocess.py --dataname swissprot --level atom --process 8`

### Finetuning: DNA/ATP binding site prediction.
Raw data for DNA/ATP binding site prediction is already provided through the github.<br>
But their file format is not appropriate for machine learning - we should reformulate them as NLP field does.

We would follow the convention of TAPE, which is the widely-used transformer architecture for proteins. [TAPE](https://github.com/songlab-cal/tape) <br>
Please be friendly with TAPE, since this model will be the base architecture for our project. <br>
We recommend you to carefully read TAPE paper before using it. [arXiv](https://arxiv.org/pdf/1906.08230.pdf)




We are planning to build our downstream codes 'on top of' the TAPE finetuning codes; it might reduce our work on defending our framework.


