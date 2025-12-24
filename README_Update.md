# Usage + Model Update — Adding new data for retraining

This file is instructions for adding new data and running update/retraining experiments.

---

Adding new data for update / retraining

Follow this minimal workflow to add a new batch and update/retrain models reproducibly.

1) Place new raw data
- Create a subfolder `data/new_batch/` and add your raw CSV (one row per sample). Recommended filename: `data/new_batch/x_pool.csv`.

2) Required columns in raw CSV
- `SMILES` (required) — structure string.
- `id` or `mol_id` (recommended) — unique identifier.
- Activity columns if labelled.

3) Preprocess the raw batch
- Edit `codes/preprocess.py` to point `file_name` to your `data/new_batch/x_pool.csv`, then run:

```bash
python codes/preprocess.py
```

This will canonicalize SMILES, remove invalid rows, filter inorganic/mixtures, and optionally remove duplicates.

4) Deduplicate against existing training data
- To avoid adding samples already in `data/x_train.csv`, canonicalize both datasets and remove overlaps.

Example (pandas quick-check):

```bash
python - <<'PY'
import pandas as pd
from rdkit.Chem import AllChem as Chem

def canon(s):
    mol = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(mol, isomericSmiles=True) if mol else None

new = pd.read_csv('data/new_batch/x_pool.csv')
new['canonical'] = new['SMILES'].apply(canon)
train = pd.read_csv('data/x_train.csv')
train['canonical'] = train['SMILES'].apply(canon)
overlap = new[new['canonical'].isin(train['canonical'].dropna())]
print('overlap rows:', len(overlap))
new_filtered = new[~new['canonical'].isin(train['canonical'].dropna())]
new_filtered.to_csv('data/new_batch/x_pool_filtered.csv', index=False)
PY
```

5) Compute descriptors for the filtered pool
- Use `codes/physicochemical_properties.py` (edit input/output paths) or your preferred tool to produce `data/new_batch/x_pool_descriptor.csv`.

6) Verify descriptor schema compatibility
- Ensure `x_pool_descriptor.csv` columns match `data/x_train_descriptor.csv`. Quick check:

```bash
python - <<'PY'
import pandas as pd
train_cols = pd.read_csv('data/x_train_descriptor.csv', nrows=0).columns
new_cols = pd.read_csv('data/new_batch/x_pool_descriptor.csv', nrows=0).columns
print('missing:', set(train_cols)-set(new_cols))
print('extra:', set(new_cols)-set(train_cols))
PY
```

7) Run update / retraining experiments
- To retrain full models on the expanded dataset, append filtered labeled samples to `data/x_train.csv` and `data/x_train_descriptor.csv`, then run the desired training script.

Example: append and run brute-force metadata training

```bash
cat data/new_batch/x_pool_filtered.csv >> data/x_train.csv
cat data/new_batch/x_pool_descriptor.csv >> data/x_train_descriptor.csv
python codes/brute_force_meta.py
```