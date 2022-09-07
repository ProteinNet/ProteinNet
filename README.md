# ProteinNet

This is a working repository for ProteinNet, a machine learning model aimming to learn general representation of proteins via multimodal contrastive learning between structures and sequences.

Please make sure that your code "Should be executable" without any kind of bugs, before the git push argument.

The following will explain how to begin your project.


## Data Download and Preprocessing.
### Pretraining: MANE
1. Download the full entries of MANE dataset from [here](https://alphafold.ebi.ac.uk/download#mane-section). <br>

2. Place the dataset at data/pretrain/mane folder, then unzip, rename the directory as 'raw' <br>
    `cd data/pretrain/mane` <br>
    `tar -xvf mane_overlap_v3.tar raw`

3. Preprocessing PDB structures using our pretraining code. <br>
    ` # current dir: data/pretrain/mane`<br>
    `cd ../../../ProteinNet`<br>
    `python preprocess.py --dataname mane --level atom --process 8`

### Finetuning: DNA/ATP binding site prediction.
Raw data for DNA/ATP binding site prediction is already provided through the github.<br>
But their file format is not appropriate for machine learning - we should treat them as NLP field does.
