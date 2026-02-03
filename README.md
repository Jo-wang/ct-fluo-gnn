## Graph-based prediction of fluorescence chromaticity in CT complexes

This repository contains the code used in the study **“Graph-based prediction of fluorescence emission chromaticity coordinates in charge-transfer complexes”** by Sujlesh Sharma, Zixin Wang, Yadan Luo, and Thanh Vinh Nguyen.

The project implements a **multi-encoder Graph Isomorphism Network (GIN)** to predict the CIE 1931 chromaticity coordinates \((x, y)\) of Lewis base–Lewis acid charge-transfer (CT) complexes from molecular structure and experimental conditions, trained on a small, imbalanced dataset.

### Key ideas

- **Input**: SMILES strings for the fluorophore and Lewis acid, plus experimental **concentration** (and optionally solvent).
- **Model**: Multi-encoder GIN (`GINMultiGraphModelwoSol`) that:
  - Encodes the fluorophore graph with stacked GIN layers.
  - Encodes Lewis acid identity and concentration via learned embeddings.
  - Predicts \((x, y)\) chromaticity coordinates with a small MLP head.
- **Training protocol**:
  - 5‑fold cross‑validation on the training portion of the data.
  - Bayesian optimisation of architecture and training hyperparameters with Optuna.
  - Final model retrained with best hyperparameters and evaluated on a held‑out test set.

Reported performance in the paper: \(R^2 = 0.92\) (MAE = 0.03) for \(x\) and \(R^2 = 0.84\) (MAE = 0.05) for \(y\).

---

## Project structure

- `hyper_train.py` – Hyperparameter search with **Optuna** using 5‑fold CV.
- `best_hype_train.py` – Train/evaluate the best‑found configuration, save fold checkpoints, and write predictions for the test set.
- `dataset_aug.py` – Main dataset and graph construction utilities using **RDKit** and **torch_geometric** (`MoleculesDataset` for training).
- `dataset.py` – Additional dataset and augmentation helpers (alternative pipelines / legacy code).
- `models/Hype_GIN.py` – Definition of the multi‑encoder GIN model (`GINMultiGraphModelwoSol`).
- `data/` – CSV files with experimental data (see below).

---

## Data format

The training scripts expect a CSV file with one row per CT complex / condition. In `hyper_train.py` and `best_hype_train.py`, the file is read as:

- Path: `data/dataset.csv`
- Columns after loading:
  - `x` – CIE 1931 x chromaticity coordinate (float)
  - `y` – CIE 1931 y chromaticity coordinate (float)
  - `con` – Concentration (float)
  - `flu` – SMILES string of the fluorophore (Lewis base)
  - `levis` – SMILES string of the Lewis acid (may contain missing values; filled with `"C"` in the code)

Other CSVs in `data/` (e.g. `train_val.csv`, `test_pool.csv`) can be adapted to this format; the critical requirement is that the DataFrame passed to `MoleculesDataset` contains at least `["x", "y", "con", "flu", "levis"]` (and optionally `sol` for solvent if present).

> **Note**: By default, missing Lewis acid entries are replaced with `"C"` (treated as a special “no acid” token). Concentration is binned into two categories internally (0.5 vs others) and encoded via an embedding layer.

---

## Installation

### 1. Create and activate an environment

Use your preferred environment manager, e.g. with `conda`:

```bash
conda create -n ct-fluo-gnn python=3.10
conda activate ct-fluo-gnn
```

### 2. Install dependencies

Install the core Python dependencies:

```bash
pip install torch torch-geometric optuna pandas numpy scikit-learn
```

You also need:

- **RDKit** (for chemistry / SMILES handling).
- **auglichem** (for `ATOM_LIST` and `CHIRALITY_LIST` utilities used in `dataset_aug.py`).

Example (using conda-forge):

```bash
conda install -c conda-forge rdkit
pip install auglichem
```

Ensure that the installed versions of `torch` and `torch-geometric` are compatible (see the official PyTorch Geometric installation instructions if needed).

---

## Running hyperparameter optimisation

`hyper_train.py` runs Optuna to search over model and training hyperparameters using 5‑fold cross‑validation on 90% of the data (with 10% held out for final testing).

From the project root:

```bash
python hyper_train.py
```

This script:

- Loads `data/dataset.csv`.
- Splits it into training and test (90% / 10%).
- Runs 5‑fold CV inside the training set.
- Optimises:
  - GIN depth and hidden sizes
  - Pooling type (`add`, `mean`, `max`)
  - Activation (`ReLU`, `leaky_ReLU`)
  - Learning rate, batch size, dropout, etc.
- Prints the **best hyperparameters** and the **best cross‑validated validation loss**.

You can adjust `n_trials` in `hyper_train.py` to trade off search time vs thoroughness.

---

## Training and evaluation with the best hyperparameters

Once good hyperparameters are selected, you can train and evaluate the final model using `best_hype_train.py`:

```bash
python best_hype_train.py
```

This script:

- Uses the fixed hyperparameters defined at the top of `best_hype_train.py`.
- Loads `data/dataset.csv`, fills missing `levis` as `"C"`, and splits into train/test.
- Runs **5‑fold cross‑validation** with early stopping, saving:
  - Best model weights for each fold under `ckpt/model_fold{fold}.pth`.
- Selects the fold with the lowest validation MAE and:
  - Reloads that checkpoint.
  - Evaluates on the held‑out test set.
  - Writes per‑sample predictions to:
    - `result/BO_CV_2025_GIN_full_wo_sol.csv`
      - Columns: `epoch`, `predicted_x`, `predicted_y`, `gt_x`, `gt_y`, `SMILES_flu`, `SMILES_acid`, `concentration`.
- Prints the average test MAE across the test set.

Make sure the `ckpt/` and `result/` directories exist (create them if needed).

---

## Adapting to new datasets

To apply this model to new CT complexes:

- **Prepare a CSV** with the same columns (`x`, `y`, `con`, `flu`, `levis`), where:
  - `x`, `y` are experimental chromaticity coordinates.
  - `flu` / `levis` are valid SMILES for the fluorophore and Lewis acid.
  - `con` is the concentration value.
- Point `hyper_train.py` / `best_hype_train.py` to your new file (modify the `pd.read_csv` path).
- Optionally, adjust:
  - The concentration binning rule in `GINMultiGraphModelwoSol.forward`.
  - The special token used for missing acids (currently `"C"`).

---

## Citation

If you use this code or model in your work, please cite:

> S. Sharma, Z. Wang, Y. Luo, T. V. Nguyen,  
> **Graph-based prediction of fluorescence emission chromaticity coordinates in charge-transfer complexes**.  
> Schools of Chemistry (UNSW) and Electrical Engineering and Computer Science (UQ).

