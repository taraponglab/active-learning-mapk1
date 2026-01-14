# Usage + Model Update — Adding new data for retraining

This file is instructions for adding new data and running update/retraining experiments.

---

Adding new data for update / retraining

Follow this minimal workflow to add a new batch and update/retrain models reproducibly.

1) Place new raw data
- Create a subfolder `data/new_batch/` and add your raw CSV (one row per sample). Recommended filename (example): `data/new_batch/x_data.csv`.

2) Required columns in raw CSV
- `SMILES` (required) — structure string.
- `id` — unique identifier.
- Activity columns if labelled.

3) Preprocess the raw batch
- Edit `codes/preprocess.py` to point `file_name` to your `data/new_batch/x_data.csv`, then run:

```bash
python codes/preprocess.py
```

This will canonicalize SMILES, remove invalid rows, filter inorganic/mixtures, and optionally remove duplicates.

4) Deduplicate against existing data sets
- To avoid adding samples already in `data/x_train.csv` and `data/x_test.csv`, canonicalize both datasets and remove overlaps.

5) Compute descriptors for the filtered data
- Use `codes/physicochemical_properties.py` (edit input/output paths) or your preferred tool to produce `data/new_batch/x_data_descriptor.csv`.

6) Verify descriptor schema compatibility
- Ensure `x_data_descriptor.csv` columns match `data/x_train_descriptor.csv`.

7) Run update / retraining experiments
- To retrain full models on the expanded data set, append filtered labeled samples to `data/x_train.csv` and `data/x_train_descriptor.csv`, then run the desired training script.
- Train the AL framework with the same procedure that were explained in README_USAGE.md

Note:
To modify the model, download the script and adjust the parameters of each model according to your requirements.
