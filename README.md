# MMO-VAE

## MMO-VAE: Variational AutoEncoder with Mask-Guided Multi-Objective Optimization for de novo Drug Design

This repository contains the implementation of MMO-VAE, a Variational AutoEncoder (VAE) with mask-guided multi-objective optimization for de novo drug design.

parer link : --

### Important Notes
- **This repository does not include code for evaluating the properties of SMILES. If you want more information, please refer to the paper.**  
- The code includes the implementation of only **Model A** 
- To use the code, preprocess your SMILES dataset using the `selfies` library: [Selfies GitHub Repository](https://github.com/aspuru-guzik-group/selfies).
- 

## File Descriptions

### Data Files (`/data`)
- `chembl_smiles.csv`  :  chembl Dataset sample
- `zinc15_smiles.csv`  :  zinc15 Dataset sample
- `chEMBL_sample.csv`  : Preprocessed chEMBL dataset sample.
- `chembl_selfies_tokens.txt` : SELFIES tokens extracted from the chEMBL dataset.
- `zinc15_sample.csv` : Preprocessed ZINC15 dataset sample.
- `zinc15_selfies_tokens.txt` : SELFIES tokens extracted from the ZINC15 dataset.

### Main Notebooks
0. **`preprocessing.ipynb`**
   - Preprocesses SMILES into SELFIES.
   - Identifies and stores the tokens present in the current dataset's SELFIES.
   

1. **`pretraining.ipynb`**
   - Implements KLD Sigmoid Annealing and Teacher Forcing techniques.
   - Pretrains the SELFIES-based model.
   - A pretrained model trained for 100 epochs on the ZINC15 dataset is available (see download link below).

2. **`finetuning.ipynb`**
   - Fine-tunes the Property Estimator and Mask Pooling Layer.
   - A fine-tuned model trained on the ZINC15 dataset is available (see download link below).

3. **`optimize.ipynb`**
   - Optimizes latent vectors using the final trained model.

### Core Scripts
- **`VAE.py`** : Implements the MMO-VAE model.
- **`utils.py`** : Contains helper functions used across the codebase.

## Model Download
models can be downloaded from the following link:  
[Google Drive - Models](https://drive.google.com/drive/folders/1riWfM56yNzfDVLCQWD6pNSIN10JksBhR?usp=sharing)

After downloading, place the models in the `/model/` folder as follows:

- **Pretrained VAE Model**: `./model/pretrained_vae_zinc15.pt`
- **Fine-tuned Model A**: `./model/finetuned_Model_A.pt`
