"""
predict_pool_all4_to_meta.py

Predict pool data with:
 - CNN (descriptor model)
 - Attention (descriptor model)
 - GCN (graph model)
 - GMPNN-Att (graph model)

Combine their probability outputs into a meta feature matrix:
 [cnn_prob, att_prob, gcn_prob, gmpnn_prob]

Scale meta features with meta_input_scaler and predict final output
with meta-CNN.

Saves:
 - pool_pred_cnn.csv
 - pool_pred_attention.csv
 - pool_pred_gcn.csv
 - pool_pred_gmpnn.csv
 - pool_meta_input.csv
 - pool_meta_predictions.csv
"""

import os
import argparse
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GraphConv, global_mean_pool, NNConv
from torch_geometric.utils import to_dense_batch
from tensorflow import keras
from keras.models import load_model

# ----------------------------
# Graph conversion utilities
# ----------------------------
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

def mol_to_graph(smiles, label=None):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    atoms = list(mol.GetAtoms())
    if len(atoms) == 0:
        return None
    x = torch.stack([atom_features(a) for a in atoms])
    edge_indices = []
    edge_attrs = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_features(bond)
        edge_indices += [[i, j], [j, i]]
        edge_attrs += [bf, bf]
    if len(edge_indices) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 4), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.stack(edge_attrs)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    if label is not None:
        data.y = torch.tensor([label], dtype=torch.float)
    return data

def is_valid_molecule(smiles):
    if not isinstance(smiles, str) or '.' in smiles:
        return False
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None and any(atom.GetAtomicNum() == 6 for atom in mol.GetAtoms())

# ----------------------------
# GCN model (matches your training)
# ----------------------------
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

# ----------------------------
# GMPNN-att model (matches your training)
# ----------------------------
class GMPNNClassifier(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim=64, num_heads=4):
        super(GMPNNClassifier, self).__init__()
        self.edge_net = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim * hidden_dim)
        )
        self.nnconv = NNConv(
            in_channels=node_dim,
            out_channels=hidden_dim,
            nn=self.edge_net,
            aggr='mean'
        )
        self.multihead_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.nnconv(x, edge_index, edge_attr)
        x = F.relu(x)
        x_dense, mask = to_dense_batch(x, batch)
        attn_output, _ = self.multihead_attn(x_dense, x_dense, x_dense, key_padding_mask=~mask)
        attn_output[~mask] = 0
        graph_embeddings = attn_output.sum(dim=1) / mask.sum(dim=1, keepdim=True)
        x = F.relu(self.lin1(graph_embeddings))
        return self.lin2(x).squeeze(1)

# ----------------------------
# Helpers
# ----------------------------
def torch_predict_probs(model, loader, device):
    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
    if len(all_probs) == 0:
        return np.zeros((0,), dtype=float)
    return np.concatenate(all_probs, axis=0)

# ----------------------------
# Main
# ----------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_folder = args.input_folder

    # Directories for each model inside input_folder
    cnn_dir = os.path.join(input_folder, "cnn")
    att_dir = os.path.join(input_folder, "attention")
    gcn_dir = os.path.join(input_folder, "gcn")
    gmpnn_att_dir = os.path.join(input_folder, "gmpnn_att")
    meta_dir = os.path.join(input_folder, "meta_attention")
    os.makedirs(input_folder, exist_ok=True)

    # ----------------------------
    # Load descriptor pool (for CNN & Attention)
    # expects PUBCHEM_CID and descriptor columns (no Label needed)
    # ----------------------------
    pool_desc = pd.read_csv(args.pool_desc)
    if "PUBCHEM_CID" not in pool_desc.columns:
        raise ValueError("pool_desc CSV must contain 'PUBCHEM_CID' column.")
    # keep descriptors columns (exclude PUBCHEM_CID and SMILES if present)
    desc_cols = [c for c in pool_desc.columns if c not in ("PUBCHEM_CID", "SMILES", "Label")]
    X_pool_desc = pool_desc[desc_cols].values

    # ----------------------------
    # Load SMILES pool (for GCN & GMPNN)
    # expects PUBCHEM_CID, SMILES
    # ----------------------------
    pool_graph_df = pd.read_csv(args.pool_graph)
    if "PUBCHEM_CID" not in pool_graph_df.columns or "SMILES" not in pool_graph_df.columns:
        raise ValueError("pool_graph CSV must contain 'PUBCHEM_CID' and 'SMILES' columns.")

    # ----------------------------
    # Load descriptor model scalers & models
    # ----------------------------
    cnn_input_scaler = joblib.load(os.path.join(cnn_dir, "scaler_model.joblib"))
    att_input_scaler = joblib.load(os.path.join(att_dir, "scaler_model.joblib"))

    cnn_model = load_model(os.path.join(cnn_dir, "cnn_model.keras"))
    att_model = load_model(os.path.join(att_dir, "attention_model.keras"))

    # ----------------------------
    # Predict CNN & Attention (probabilities)
    # ----------------------------
    X_pool_cnn_scaled = cnn_input_scaler.transform(X_pool_desc)
    cnn_pred = cnn_model.predict(X_pool_cnn_scaled)    # shape: (n_samples, 1) or (n_samples,)
    cnn_prob = np.asarray(cnn_pred).reshape(-1)
    cnn_label = (cnn_prob >= 0.5).astype(int) 

    # Save both probability and label
    df_cnn = pd.DataFrame({
        "PUBCHEM_CID": pool_desc["PUBCHEM_CID"].values,
        "cnn_prob": cnn_prob,
        "cnn_label": cnn_label
    })
    df_cnn.to_csv(os.path.join(args.input_folder, "pool_pred_cnn.csv"), index=False)

    # === Attention predictions ===
    X_pool_att_scaled = att_input_scaler.transform(X_pool_desc)
    att_pred = att_model.predict(X_pool_att_scaled)
    att_prob = np.asarray(att_pred).reshape(-1)
    att_label = (att_prob >= 0.5).astype(int)  # convert to 0/1

    # Save both probability and label
    df_att = pd.DataFrame({
        "PUBCHEM_CID": pool_desc["PUBCHEM_CID"].values,
        "att_prob": att_prob,
        "att_label": att_label
    })
    df_att.to_csv(os.path.join(args.input_folder, "pool_pred_attention.csv"), index=False)
    print("[INFO] Saved CNN & Attention predictions.")

    # ----------------------------
    # Prepare graphs for GCN & GMPNN
    # ----------------------------
    graphs = []
    graph_ids = []
    for idx, row in pool_graph_df.iterrows():
        smi = row["SMILES"]
        cid = row["PUBCHEM_CID"]
        if not isinstance(smi, str):
            continue
        try:
            g = mol_to_graph(smi, label=None)
        except Exception:
            g = None
        if g is None:
            continue
        g.PUBCHEM_CID = cid
        graphs.append(g)
        graph_ids.append(cid)

    if len(graphs) == 0:
        print("[WARN] No valid graphs created from pool_graph. GCN/GMPNN predictions will be empty.")
        gcn_prob = np.array([], dtype=float)
        gmpnn_prob = np.array([], dtype=float)
    else:
        batch_size = args.batch_size
        graph_loader = DataLoader(graphs, batch_size=batch_size)

        # ----------------------------
        # Load GCN model
        # ----------------------------
        node_dim = graphs[0].x.shape[1]
        gcn_model = GCNNClassifier(node_dim).to(device)
        gcn_state_path = os.path.join(gcn_dir, "gcn_model.pt")
        if not os.path.exists(gcn_state_path):
            raise FileNotFoundError(f"GCN model file not found: {gcn_state_path}")
        gcn_model.load_state_dict(torch.load(gcn_state_path, map_location=device))
        gcn_model.to(device)
        gcn_model.eval()

        gcn_prob = torch_predict_probs(gcn_model, graph_loader, device)  # aligns with graphs order

        # ----------------------------
        # Load GMPNN model
        # ----------------------------
        node_dim = graphs[0].x.shape[1]
        edge_dim = graphs[0].edge_attr.shape[1] if hasattr(graphs[0], "edge_attr") and graphs[0].edge_attr is not None else 4
        gmpnn_model = GMPNNClassifier(node_dim, edge_dim).to(device)
        gmpnn_state_path = os.path.join(gmpnn_att_dir, "gmpnn_att_model.pt")
        if not os.path.exists(gmpnn_state_path):
            raise FileNotFoundError(f"GMPNN model file not found: {gmpnn_state_path}")
        gmpnn_model.load_state_dict(torch.load(gmpnn_state_path, map_location=device))
        gmpnn_model.to(device)
        gmpnn_model.eval()

        gmpnn_prob = torch_predict_probs(gmpnn_model, graph_loader, device)

        # ----------------------------
        # Build dataframes for graph preds (order matches graph_ids)
        # ----------------------------
        # Convert torch_probs to numpy if needed
        gcn_prob = np.asarray(gcn_prob).reshape(-1)
        gmpnn_prob = np.asarray(gmpnn_prob).reshape(-1)

        # Convert probabilities to 0/1 labels
        gcn_label = (gcn_prob >= 0.5).astype(int)
        gmpnn_label = (gmpnn_prob >= 0.5).astype(int)

        # Save GCN predictions
        df_gcn = pd.DataFrame({
            "PUBCHEM_CID": graph_ids,
            "gcn_prob": gcn_prob,
            "gcn_label": gcn_label
        })
        df_gcn.to_csv(os.path.join(input_folder, "pool_pred_gcn.csv"), index=False)

        # Save GMPNN predictions
        df_gmpnn = pd.DataFrame({
            "PUBCHEM_CID": graph_ids,
            "gmpnn_att_prob": gmpnn_prob,
            "gmpnn_att_label": gmpnn_label
        })
        df_gmpnn.to_csv(os.path.join(input_folder, "pool_pred_gmpnn.csv"), index=False)

        print("[INFO] Saved GCN & GMPNN predictions with 0/1 labels.")

    # ----------------------------
    # Merge predictions by PUBCHEM_CID
    # Use inner join across the 4 predictions so meta receives only molecules with all 4 preds
    # ----------------------------
    # Start from descriptor predictions (they typically share the descriptor list)
    merged = df_cnn.merge(df_att, on="PUBCHEM_CID", how="inner")

    if len(graphs) > 0:
        merged = merged.merge(df_gcn, on="PUBCHEM_CID", how="inner")
        merged = merged.merge(df_gmpnn, on="PUBCHEM_CID", how="inner")
    else:
        # If graphs missing, fill gcn/gmpnn with NaN (and then drop because meta needs all 4)
        merged["gcn_prob"] = np.nan
        merged["gmpnn_att_prob"] = np.nan

    # Drop rows missing any baseline (meta requires all 4)
    merged = merged.dropna(subset=["cnn_prob", "att_prob", "gcn_prob", "gmpnn_att_prob"])
    if merged.shape[0] == 0:
        raise RuntimeError("No molecules have predictions from all 4 baselines. Cannot run meta prediction.")

    # ----------------------------
    # Build meta X matrix and save
    # ----------------------------
    meta_X = merged[["cnn_prob", "att_prob", "gcn_prob", "gmpnn_att_prob"]].values
    pd.DataFrame(meta_X, columns=["cnn_prob", "att_prob", "gcn_prob", "gmpnn_att_prob"]) \
        .to_csv(os.path.join(input_folder, "pool_meta_input.csv"), index=False)

    # ----------------------------
    # Load meta scaler & meta model; predict final
    # ----------------------------
    meta_input_scaler_path = os.path.join(meta_dir, "scaler_model.joblib")
    if not os.path.exists(meta_input_scaler_path):
        raise FileNotFoundError(f"Meta input scaler not found: {meta_input_scaler_path}")
    meta_input_scaler = joblib.load(meta_input_scaler_path)
    meta_X_scaled = meta_input_scaler.transform(meta_X)

    meta_model_path = os.path.join(meta_dir, "meta_attention_model.keras")
    if not os.path.exists(meta_model_path):
        raise FileNotFoundError(f"Meta model not found: {meta_model_path}")
    meta_model = load_model(meta_model_path)

    meta_pred = meta_model.predict(meta_X_scaled)    # probabilities or logits depending on your meta model
    # ensure 1D probabilities
    meta_prob = np.asarray(meta_pred).reshape(-1)
    meta_label = (meta_prob >= 0.5).astype(int) 

    # Save both probability and label
    df_meta = pd.DataFrame({
        "PUBCHEM_CID": pool_desc["PUBCHEM_CID"].values,
        "meta_prob": meta_prob,
        "meta_label": meta_label
    })
    df_meta.to_csv(os.path.join(args.input_folder, "pool_pred_meta.csv"), index=False)
    print("[INFO] Saved meta predictions.")
    
    # Attach final predictions
    merged["meta_prob"] = meta_prob

    # Save final CSV with PUBCHEM_CID and all probs
    out_cols = ["PUBCHEM_CID", "cnn_prob", "att_prob", "gcn_prob", "gmpnn_att_prob", "meta_prob"]
    merged[out_cols].to_csv(os.path.join(input_folder, "pool_meta_predictions.csv"), index=False)
    print("[✓] Saved final meta predictions to:", os.path.join(input_folder, "pool_meta_predictions.csv"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict pool: CNN, Attention, GCN, GMPNN -> Meta-Attention")
    parser.add_argument("--input_folder", type=str, required=True,
                        help="Iteration folder containing subfolders: cnn/, attention/, gcn/, gmpnn/, meta_attention/")
    parser.add_argument("--pool_desc", type=str, required=True,
                        help="CSV with descriptor features and PUBCHEM_CID (used by CNN & Attention).")
    parser.add_argument("--pool_graph", type=str, required=True,
                        help="CSV with PUBCHEM_CID and SMILES (used by GCN & GMPNN).")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for graph DataLoader")
    args = parser.parse_args()

    main(args)
