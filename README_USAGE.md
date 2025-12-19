Usage
=====

Overview: This repository contains code to preprocess chemical SMILES data, compute physicochemical descriptors, train several baseline and graph neural network models, run meta-models, and perform active learning (AL) experiments using multiple strategies (random, uncertainty, margin, entropy). The main orchestration scripts live in `codes/` and example data live under `data/`.

Create a virtual environment with Python 3.11.
Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

- Preprocess data: edit `codes/preprocess.py` to change the `file_name` variable to your input CSV (it writes canonical SMILES to a cleaned CSV). Then run:

```bash
python codes/preprocess.py
```

- Compute physicochemical descriptors: edit `codes/physicochemical_properties.py` to set the input CSV path (cleaned output from preprocessing) and the desired output path, then run:

```bash
python codes/physicochemical_properties.py
```

- Output descriptor files (examples in `data/`) are used as model inputs (descriptor CSVs usually contain `PUBCHEM_CID`, descriptor columns and `Label`).

Train individual models (examples): each model script accepts `--subset`, `--test`, `--output_folder`, and `--iter` (some graph models accept `--epochs` and `--batch_size`). Example commands:

# Descriptor-based models (use descriptor CSVs)
```bash
python codes/attention.py --subset data/random42/x_subset_descriptor.csv --test data/x_test_descriptor.csv --output_folder results/attention_iter1 --iter 1
python codes/cnn.py       --subset data/random42/x_subset_descriptor.csv --test data/x_test_descriptor.csv       --output_folder results/cnn_iter1       --iter 1
```

# Graph-based models (use graph CSVs with SMILES column)
```bash
python codes/gcn.py      --subset data/random42/x_subset.csv --test data/x_test.csv --output_folder results/gcn_iter1 --iter 1
python codes/gmpnn_att.py--subset data/random42/x_subset.csv --test data/x_test.csv --output_folder results/gmpnn_iter1 --iter 1
```

Meta-models: after running the baseline models above, run meta input and meta models to build meta-features and train meta-models. Example orchestration (see `codes/brute_force_meta.py` / `codes/al_random.py` for examples):

```bash
python codes/meta_input.py --output_folder results/brute_force --models "attention cnn gcn gmpnn_att" --train_csv data/x_train.csv --test_csv data/x_test.csv
python codes/meta_cnn.py --input results/brute_force/meta_input_trainval.csv --test results/brute_force/meta_input_test.csv --output_folder results/brute_force/meta_cnn --model_type cnn --iter 1
python codes/meta_attention.py --input results/brute_force/meta_input_trainval.csv --test results/brute_force/meta_input_test.csv --output_folder results/brute_force/meta_attention --model_type attention --iter 1
```

Active Learning workflows: high-level AL loops are provided in:
- `codes/al_random.py` — random selection loop
- `codes/al_uncertainty.py` — uncertainty strategy loop (uses `codes/uncertainty_cal.py`)
- `codes/al_margin.py` — margin strategy loop (uses `codes/margin_cal.py`)
- `codes/al_entropy.py` — entropy strategy loop (uses `codes/entropy_cal.py`)

Each AL script follows a similar structure: it expects a `data/` folder with initial subset and pool files (descriptors and graph CSVs). Before running, edit the top of the AL script to point to your `data_folder_path` and `base_output_dir` if necessary. Example:

```bash
python codes/al_random.py
```

The AL scripts internally call the strategy calculators in `codes/` (`random_cal.py`, `uncertainty_cal.py`, `margin_cal.py`, `entropy_cal.py`) and model training scripts; check the top-level variables in each AL script to match your file layout.

Brute-force experiments (e.g., full data training): to run a brute-force training with a full data, prepare `data/x_train.csv` and `data/x_train_descriptor.csv` containing the 100% of the training set, then run:

```bash
python codes/brute_force_meta.py
```

This will run the baseline descriptor and graph models, then generate meta-inputs and train meta-models for comparison.

Conventional RF baseline (meta-RF with descriptors + graph representation):
- `codes/al_rf.py` runs an RF-based active-learning baseline that trains a meta-RF using descriptor features together with a numeric representation of the molecular graph. Because Random Forests cannot consume raw molecular graphs directly, first compute numeric graph features (node/edge summaries or graph descriptors) using `codes/graph_rf_cal.py` (or supply precomputed `data/graph_rf_*` CSVs). Configure the folder paths, iterations, and strategy constants at the top of `codes/al_rf.py` and then run:

```bash
python codes/al_rf.py
```

- For full-data (brute-force) RF training, run `codes/brute_force_rf.py` if present; otherwise adapt `codes/brute_force_meta.py` to call RF training on the full training set. This produces RF baselines trained on the complete dataset for comparison with AL runs.

Notes & tips:
- Most scripts write outputs to the folder provided by `--output_folder` or the hardcoded folders at top of scripts. Create these folders or update script variables as needed.
- Many helper scripts (preprocess, descriptor calculation) currently use hardcoded file paths in their `__main__` blocks — either edit those paths or wrap the functionality in a small driver script that sets the correct filenames.
- For reproducible AL runs, ensure random seeds / `random_state` values are set consistently in the scripts (many scripts already use fixed seeds like `42`).
- GPU: `gcn.py` and `gmpnn_att.py` will use CUDA if available; otherwise they fall back to CPU.
