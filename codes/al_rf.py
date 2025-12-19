import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score, average_precision_score
from joblib import dump
from scipy.stats import entropy as scipy_entropy

# =========================================
# USER PARAMETERS
# =========================================
ITERATIONS = 5
POOL_SELECT_FRACTION = 0.05
RANDOM_STATE = 42
ACTIVE_LEARNING_STRATEGY = "random"  # "uncertainty", "margin", "entropy", "random"

# Folders
DESC_FOLDER = "data/descriptor_43"
GRAPH_FOLDER = "data/graph_rf_43"
OUTPUT_DIR = "random_rf_43"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Filenames
FILES = {
    "subset": "x_subset.csv",
    "pool": "x_pool.csv",
    "test": "x_test.csv"
}

# =========================================
# HELPER FUNCTIONS
# =========================================
def evaluate_model(y_true, y_prob):
    y_pred = (y_prob > 0.5).astype(int)
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUROC": roc_auc_score(y_true, y_pred),
        "AUPRC": average_precision_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred)
    }

def load_features(folder, file, drop_columns=None):
    df = pd.read_csv(os.path.join(folder, file)).reset_index(drop=True)
    if drop_columns is None:
        drop_columns = ["PUBCHEM_CID", "Label"]
        if "SMILES" in df.columns:
            drop_columns.append("SMILES")
    X = df.drop(columns=drop_columns, errors="ignore").values.astype(float)
    y = df["Label"].values if "Label" in df.columns else None
    return df, X, y

# ------------------------------
# Functions for modal-based strategies
# ------------------------------
def get_prob_0(df, col_name='y_pred_meta'):
    df['y_pred_meta_0'] = 1 - df[col_name]
    return df

def entropy_sampling(df, cols=['y_pred_meta_0', 'y_pred_meta']):
    proba = df[cols].values
    entropies = scipy_entropy(proba.T)
    df['strategy_value'] = entropies
    return df

def margin_sampling(df, cols=['y_pred_meta_0', 'y_pred_meta']):
    proba = df[cols].values
    part = np.partition(-proba, 1, axis=1)
    margins = -part[:,0] + part[:,1]
    df['strategy_value'] = margins
    return df

def uncertainty_sampling(df, col='y_pred_meta'):
    df['strategy_value'] = np.abs(df[col] - 0.5)
    return df

def active_learning_select(df, strategy="uncertainty", fraction=0.05):
    n_select = max(1, int(len(df) * fraction))
    if strategy == "random":
        selected_idx = np.random.choice(len(df), n_select, replace=False)
    elif strategy in ["uncertainty", "margin"]:
        selected_idx = df['strategy_value'].sort_values().index[:n_select]
    elif strategy == "entropy":
        selected_idx = df['strategy_value'].sort_values(ascending=False).index[:n_select]
    else:
        raise ValueError(f"Unknown strategy {strategy}")
    return selected_idx

# =========================================
# LOAD DATA
# =========================================
train_desc_df, X_train_desc, y_train = load_features(DESC_FOLDER, FILES["subset"])
pool_desc_df, X_pool_desc, y_pool = load_features(DESC_FOLDER, FILES["pool"])
test_desc_df, X_test_desc, y_test = load_features(DESC_FOLDER, FILES["test"])

train_graph_df, X_train_graph, _ = load_features(GRAPH_FOLDER, FILES["subset"])
pool_graph_df, X_pool_graph, _ = load_features(GRAPH_FOLDER, FILES["pool"])
test_graph_df, X_test_graph, _ = load_features(GRAPH_FOLDER, FILES["test"])

# Ensure labels align
assert len(y_train) == X_train_desc.shape[0] == X_train_graph.shape[0]
assert len(y_pool) == X_pool_desc.shape[0] == X_pool_graph.shape[0]

# =========================================
# ITERATIVE ACTIVE LEARNING LOOP
# =========================================
for iteration in range(1, ITERATIONS + 1):
    print(f"\n=== Iteration {iteration} | Strategy: {ACTIVE_LEARNING_STRATEGY} ===")
    iter_folder = os.path.join(OUTPUT_DIR, f"iteration_{iteration}")
    os.makedirs(iter_folder, exist_ok=True)
    
    # ------------------------------
    # SCALE FEATURES
    # ------------------------------
    scaler_desc = StandardScaler()
    X_train_desc_scaled = scaler_desc.fit_transform(X_train_desc)
    X_pool_desc_scaled  = scaler_desc.transform(X_pool_desc)
    X_test_desc_scaled  = scaler_desc.transform(X_test_desc)
    dump(scaler_desc, os.path.join(iter_folder, f"scaler_desc_{iteration}.joblib"))
    
    scaler_graph = StandardScaler()
    X_train_graph_scaled = scaler_graph.fit_transform(X_train_graph)
    X_pool_graph_scaled  = scaler_graph.transform(X_pool_graph)
    X_test_graph_scaled  = scaler_graph.transform(X_test_graph)
    dump(scaler_graph, os.path.join(iter_folder, f"scaler_graph_{iteration}.joblib"))
    
    # ------------------------------
    # TRAIN BASE RANDOM FORESTS
    # ------------------------------
    rf_desc = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    rf_desc.fit(X_train_desc_scaled, y_train)
    dump(rf_desc, os.path.join(iter_folder, f"rf_desc_{iteration}.joblib"))
    
    rf_graph = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    rf_graph.fit(X_train_graph_scaled, y_train)
    dump(rf_graph, os.path.join(iter_folder, f"rf_graph_{iteration}.joblib"))
    
    # ------------------------------
    # BASE RF PREDICTIONS
    # ------------------------------
    train_pred_desc  = rf_desc.predict_proba(X_train_desc_scaled)[:,1]
    train_pred_graph = rf_graph.predict_proba(X_train_graph_scaled)[:,1]
    
    test_pred_desc  = rf_desc.predict_proba(X_test_desc_scaled)[:,1]
    test_pred_graph = rf_graph.predict_proba(X_test_graph_scaled)[:,1]
    
    pool_pred_desc  = rf_desc.predict_proba(X_pool_desc_scaled)[:,1]
    pool_pred_graph = rf_graph.predict_proba(X_pool_graph_scaled)[:,1]
    
    # ------------------------------
    # META-RF TRAINING
    # ------------------------------
    meta_X_train = np.vstack([train_pred_desc, train_pred_graph]).T
    meta_y_train = y_train
    meta_rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    meta_rf.fit(meta_X_train, meta_y_train)
    dump(meta_rf, os.path.join(iter_folder, f"meta_rf_{iteration}.joblib"))
    
    # ------------------------------
    # META-RF PREDICTIONS
    # ------------------------------
    meta_test_X = np.vstack([test_pred_desc, test_pred_graph]).T
    meta_test_prob = meta_rf.predict_proba(meta_test_X)[:,1]
    test_metrics = evaluate_model(y_test, meta_test_prob)
    print(f"Meta-RF Test Metrics: {test_metrics}")
    pd.DataFrame([test_metrics]).to_csv(os.path.join(iter_folder, "test_metrics.csv"), index=False)
    
    # ------------------------------
    # ACTIVE LEARNING SELECTION
    # ------------------------------
    meta_pool_X = np.vstack([pool_pred_desc, pool_pred_graph]).T
    meta_pool_prob = meta_rf.predict_proba(meta_pool_X)[:,1]
    
    meta_pool_df = pd.DataFrame({
        "PUBCHEM_CID": pool_desc_df["PUBCHEM_CID"],
        "y_pred_meta": meta_pool_prob
    })
    meta_pool_df = get_prob_0(meta_pool_df)
    
    # Compute strategy values
    if ACTIVE_LEARNING_STRATEGY == "uncertainty":
        meta_pool_df = uncertainty_sampling(meta_pool_df)
    elif ACTIVE_LEARNING_STRATEGY == "margin":
        meta_pool_df = margin_sampling(meta_pool_df)
    elif ACTIVE_LEARNING_STRATEGY == "entropy":
        meta_pool_df = entropy_sampling(meta_pool_df)
    elif ACTIVE_LEARNING_STRATEGY == "random":
        pass
    
    # Select top fraction
    selected_idx = active_learning_select(meta_pool_df, strategy=ACTIVE_LEARNING_STRATEGY, fraction=POOL_SELECT_FRACTION)
    selected_desc = pool_desc_df.iloc[selected_idx].reset_index(drop=True)
    selected_graph = pool_graph_df.iloc[selected_idx].reset_index(drop=True)
    
    # Save predictions, strategy values, selected data
    meta_pool_df.to_csv(os.path.join(iter_folder, f"meta_pool_{ACTIVE_LEARNING_STRATEGY}.csv"), index=False)
    selected_desc.to_csv(os.path.join(iter_folder, f"selected_desc_{iteration}.csv"), index=False)
    selected_graph.to_csv(os.path.join(iter_folder, f"selected_graph_{iteration}.csv"), index=False)
    
    # Merge selected into training
    train_desc_df = pd.concat([train_desc_df, selected_desc], axis=0).reset_index(drop=True)
    train_graph_df = pd.concat([train_graph_df, selected_graph], axis=0).reset_index(drop=True)
    
    # Remove selected from pool
    pool_desc_df = pool_desc_df.drop(selected_idx).reset_index(drop=True)
    pool_graph_df = pool_graph_df.drop(selected_idx).reset_index(drop=True)
    
    # Update features for next iteration
    X_train_desc = train_desc_df.drop(columns=["PUBCHEM_CID","Label"]).values.astype(float)
    X_train_graph = train_graph_df.drop(columns=["PUBCHEM_CID","Label","SMILES"], errors="ignore").values.astype(float)
    y_train = train_desc_df["Label"].values
    
    X_pool_desc = pool_desc_df.drop(columns=["PUBCHEM_CID","Label"]).values.astype(float)
    X_pool_graph = pool_graph_df.drop(columns=["PUBCHEM_CID","Label","SMILES"], errors="ignore").values.astype(float)
    y_pool = pool_desc_df["Label"].values
    
    # Save iteration datasets
    train_desc_df.to_csv(os.path.join(iter_folder, "train_desc.csv"), index=False)
    train_graph_df.to_csv(os.path.join(iter_folder, "train_graph.csv"), index=False)
    pool_desc_df.to_csv(os.path.join(iter_folder, "pool_desc.csv"), index=False)
    pool_graph_df.to_csv(os.path.join(iter_folder, "pool_graph.csv"), index=False)
    
    print(f"Selected {len(selected_desc)} samples from pool for next iteration.")
    print(f"Remaining pool size: {len(pool_desc_df)}")
