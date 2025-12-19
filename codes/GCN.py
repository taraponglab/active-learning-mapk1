import os
import pandas as pd
from datetime import datetime
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GraphConv, global_mean_pool
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")


# ======================
# --- Atom & Bond Features ---
# ======================
def atom_features(atom):
    return torch.tensor([
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        int(atom.GetChiralTag()),
        atom.GetTotalNumHs(),
        int(atom.GetHybridization()),
        atom.GetIsAromatic(),
        atom.GetMass(),
    ], dtype=torch.float)


def bond_features(bond):
    return torch.tensor([
        float(bond.GetBondTypeAsDouble()),
        bond.IsInRing(),
        int(bond.GetStereo()),
        bond.GetIsConjugated(),
    ], dtype=torch.float)


# ======================
# --- Convert SMILES to PyG Graph ---
# ======================
def mol_to_graph(smiles, label=None):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    x = torch.stack([atom_features(atom) for atom in mol.GetAtoms()])
    edge_index, edge_attr = [], []

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index += [[i, j], [j, i]]  # Undirected graph
        feat = bond_features(bond)
        edge_attr += [feat, feat]

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.stack(edge_attr)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    if label is not None:
        data.y = torch.tensor([label], dtype=torch.float)
    return data


def is_valid_molecule(smiles):
    if not isinstance(smiles, str) or '.' in smiles:
        return False
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None and any(atom.GetAtomicNum() == 6 for atom in mol.GetAtoms())


# ======================
# --- Data Loading & Preprocessing ---
# ======================
def load_data(file_path, smiles_col, label_col):
    df = pd.read_csv(file_path)
    df = df[df[smiles_col].apply(is_valid_molecule)].reset_index(drop=True)
    df = df.dropna(subset=[smiles_col])
    return df


def split_data(df, results_dir, smiles_col, label_col):
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df[label_col], random_state=42)
    train_df.to_csv(f"{results_dir}/train.csv", index=False)
    val_df.to_csv(f"{results_dir}/val.csv", index=False)
    return train_df, val_df


def convert_to_graphs(df, smiles_col, label_col):
    df['graph'] = df.apply(lambda row: mol_to_graph(row[smiles_col], row[label_col]), axis=1)
    return df['graph'].dropna().tolist()


# ======================
# --- GCN Model ---
# ======================
class GCNNClassifier(nn.Module):
    def __init__(self, node_dim, hidden_dim=64, num_layers=3):
        super(GCNNClassifier, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GraphConv(node_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GraphConv(hidden_dim, hidden_dim))

        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        return self.lin2(x).squeeze(1)


# ======================
# --- Evaluation ---
# ======================
def evaluate(model, loader, device):
    model.eval()
    y_true, y_logits = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            y_true.extend(batch.y.view(-1).cpu().numpy())
            y_logits.extend(logits.cpu().numpy())

    y_true = np.array(y_true)
    y_probs = torch.sigmoid(torch.tensor(y_logits)).numpy()
    y_pred = (y_probs >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_probs)
    auprc = average_precision_score(y_true, y_probs)
    mcc = matthews_corrcoef(y_true, y_pred)

    return acc, auc, auprc, mcc, y_probs, y_pred


def predict_and_save(model, loader, df, set_name, results_dir, device):
    acc, auc, auprc, mcc, y_prob, y_pred = evaluate(model, loader, device)
    cid_list = df["PUBCHEM_CID"].values

    prob_df = pd.DataFrame({
        "PUBCHEM_CID": cid_list,
        "y_prob": y_prob
    })
    prob_df.to_csv(f"{results_dir}/{set_name}_prob.csv", index=False)

    pred_df = pd.DataFrame({
        "PUBCHEM_CID": cid_list,
        "y_pred": y_pred
    })
    pred_df.to_csv(f"{results_dir}/{set_name}_pred.csv", index=False)

    return y_prob, y_pred


# ======================
# --- Main ---
# ======================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train GCN model for molecular properties")
    parser.add_argument("--subset", type=str, required=True, help="Path to subset CSV for training")
    parser.add_argument("--test", type=str, required=True, help="Path to test CSV")
    parser.add_argument("--output_folder", type=str, default=".", help="Folder to save outputs")
    parser.add_argument("--iter", type=int, default=1, help="Iteration number")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level: 0=silent, 1=verbose")

    args = parser.parse_args()

    verbose = args.verbose
    current_iteration = args.iter
    train_csv = args.subset
    test_csv = args.test
    results_dir = args.output_folder
    os.makedirs(results_dir, exist_ok=True)
    epochs = args.epochs
    batch_size = args.batch_size

    print(f"\n🚀 Starting PyTorch GCN training | Iteration {current_iteration}")
    print(f"Train CSV: {train_csv}")
    print(f"Test CSV: {test_csv}")
    print(f"Output folder: {results_dir}")
    print(f"Epochs: {epochs} | Batch size: {batch_size}\n")

    # --- Load data ---
    smiles_col = "SMILES"
    label_col = "Label"

    df = load_data(train_csv, smiles_col, label_col)
    print(f"Number of valid SMILES: {len(df)}")

    train_df, val_df = split_data(df, results_dir, smiles_col, label_col)
    test_df = pd.read_csv(test_csv)

    train_graphs = convert_to_graphs(train_df, smiles_col, label_col)
    val_graphs = convert_to_graphs(val_df, smiles_col, label_col)
    test_graphs = convert_to_graphs(test_df, smiles_col, label_col)

    print(f"📊 Train: {len(train_graphs)} | Val: {len(val_graphs)} | Test: {len(test_graphs)}")

    # --- DataLoaders ---
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size)
    test_loader = DataLoader(test_graphs, batch_size=batch_size)

    # --- Device & Model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    node_dim = train_graphs[0].x.shape[1]
    model = GCNNClassifier(node_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    pos_weight = torch.tensor([len(train_df[train_df[label_col] == 0]) / len(train_df[train_df[label_col] == 1])],
                              dtype=torch.float).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    start_train_time = time.time()
    # --- Training Loop ---
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch)
            loss = loss_fn(logits, batch.y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs

        avg_loss = total_loss / len(train_loader.dataset)

        if verbose:
            acc, auc, auprc, mcc, _, _ = evaluate(model, val_loader, device)
            print(f"Epoch {epoch:02d} | Train Loss: {avg_loss:.4f} | "
                  f"Val Acc: {acc:.4f} | AUROC: {auc:.4f} | "
                  f"AUPRC: {auprc:.4f} | MCC: {mcc:.4f}")
    end_train_time = time.time()
    training_time = end_train_time - start_train_time


    # --- Save Model ---
    torch.save(model.state_dict(), f"{results_dir}/gcn_model.pt")
   
    # === Final predictions on Test Set ===
    pred_start_time = time.time()

    # --- Predict & Save ---
    print("\n📊 Saving Predictions...")
    predict_and_save(model, train_loader, train_df, "train", results_dir, device)
    predict_and_save(model, val_loader, val_df, "val", results_dir, device)
    predict_and_save(model, test_loader, test_df, "test", results_dir, device)

    # --- Final Test Metrics ---
    acc, auc, auprc, mcc, _, _ = evaluate(model, test_loader, device)
    print(f"\n✅ Final Test Evaluation | Acc: {acc:.4f}, AUROC: {auc:.4f}, AUPRC: {auprc:.4f}, MCC: {mcc:.4f}")

    end_eval_time = time.time()
    evaluation_time = end_eval_time - pred_start_time

    # === Record Training Information ===
    total_time = training_time + evaluation_time
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # --- Save results summary ---
    result_row = {
        "Model": f"GCNN_iter{current_iteration}",
        "Accuracy": round(acc, 3),
        "AUROC": round(auc, 3),
        "AUPRC": round(auprc, 3),
        "MCC": round(mcc, 3)
    }

    result_file = os.path.join(results_dir, "result.csv")
    try:
        results_df = pd.read_csv(result_file)
    except FileNotFoundError:
        results_df = pd.DataFrame(columns=["Model", "Accuracy", "AUROC", "AUPRC", "MCC"])

    results_df = pd.concat([results_df, pd.DataFrame([result_row])], ignore_index=True)
    results_df.to_csv(result_file, index=False)

    print(f"✅ Results saved to {results_dir}/result.csv")

    training_info = {
        "Run_number": [current_iteration],
        "Total training time(s)": [total_time],
        "Model training time(s)": [training_time],
        "Prediction time(s)": [evaluation_time],
        "Epochs": [epochs],
        "Batch_size": [batch_size],
        "Test_size": [len(test_graphs)],
        "Timestamp": [timestamp]
    }

    # Convert dictionary to DataFrame
    training_info_df = pd.DataFrame(training_info)

    # Append to CSV (or create if not exists)
    training_info_path = os.path.join(results_dir, "training_info.csv")
    training_info_df.to_csv(
        training_info_path,
        mode='a',  # append mode
        header=not os.path.exists(training_info_path),
        index=False
    )