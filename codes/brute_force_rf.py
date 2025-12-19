import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score, average_precision_score
from joblib import dump

# =========================================
# USER PARAMETERS
# =========================================
RANDOM_STATE = 42
OUTPUT_DIR = "rf/brute_force_rf_3"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Folders
DESC_FOLDER = "data/descriptor_42"
GRAPH_FOLDER = "data/graph_rf_42"

# Filenames
FILES = {
    "train": "x_train.csv",
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
        "AUPRC": average_precision_score(y_true, y_prob),
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

# =========================================
# LOAD DATA
# =========================================
train_desc_df, X_train_desc, y_train = load_features(DESC_FOLDER, FILES["train"])
test_desc_df, X_test_desc, y_test = load_features(DESC_FOLDER, FILES["test"])

train_graph_df, X_train_graph, _ = load_features(GRAPH_FOLDER, FILES["train"])
test_graph_df, X_test_graph, _ = load_features(GRAPH_FOLDER, FILES["test"])

# =========================================
# SCALE FEATURES
# =========================================
scaler_desc = StandardScaler()
X_train_desc_scaled = scaler_desc.fit_transform(X_train_desc)
X_test_desc_scaled  = scaler_desc.transform(X_test_desc)
dump(scaler_desc, os.path.join(OUTPUT_DIR, "scaler_desc.joblib"))

scaler_graph = StandardScaler()
X_train_graph_scaled = scaler_graph.fit_transform(X_train_graph)
X_test_graph_scaled  = scaler_graph.transform(X_test_graph)
dump(scaler_graph, os.path.join(OUTPUT_DIR, "scaler_graph.joblib"))

# =========================================
# TRAIN BASE RANDOM FORESTS
# =========================================
rf_desc = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
rf_desc.fit(X_train_desc_scaled, y_train)
dump(rf_desc, os.path.join(OUTPUT_DIR, "rf_desc.joblib"))

rf_graph = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
rf_graph.fit(X_train_graph_scaled, y_train)
dump(rf_graph, os.path.join(OUTPUT_DIR, "rf_graph.joblib"))

# =========================================
# BASE RF PREDICTIONS
# =========================================
train_pred_desc  = rf_desc.predict_proba(X_train_desc_scaled)[:,1]
train_pred_graph = rf_graph.predict_proba(X_train_graph_scaled)[:,1]

test_pred_desc  = rf_desc.predict_proba(X_test_desc_scaled)[:,1]
test_pred_graph = rf_graph.predict_proba(X_test_graph_scaled)[:,1]

# =========================================
# META-RF TRAINING
# =========================================
meta_X_train = np.vstack([train_pred_desc, train_pred_graph]).T
meta_y_train = y_train
meta_rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
meta_rf.fit(meta_X_train, meta_y_train)
dump(meta_rf, os.path.join(OUTPUT_DIR, "meta_rf.joblib"))

# =========================================
# META-RF PREDICTIONS & EVALUATION
# =========================================
meta_test_X = np.vstack([test_pred_desc, test_pred_graph]).T
meta_test_prob = meta_rf.predict_proba(meta_test_X)[:,1]

test_metrics = evaluate_model(y_test, meta_test_prob)
print(f"Meta-RF Test Metrics: {test_metrics}")

pd.DataFrame([test_metrics]).to_csv(os.path.join(OUTPUT_DIR, "test_metrics.csv"), index=False)
# =========================================
# SAVE PREDICTIONS & PROBABILITIES
# =========================================

# Train set
train_df_out = train_desc_df.copy()
train_df_out["pred_prob_desc"] = train_pred_desc
train_df_out["pred_prob_graph"] = train_pred_graph
train_df_out["pred_meta_prob"] = meta_X_train.mean(axis=1)  # optional, or use meta_rf.predict_proba(meta_X_train)[:,1]
train_df_out["pred_meta_class"] = (meta_rf.predict(meta_X_train)).astype(int)
train_df_out.to_csv(os.path.join(OUTPUT_DIR, "train_predictions.csv"), index=False)

# Test set
test_df_out = test_desc_df.copy()
test_df_out["pred_prob_desc"] = test_pred_desc
test_df_out["pred_prob_graph"] = test_pred_graph
test_df_out["pred_meta_prob"] = meta_test_prob
test_df_out["pred_meta_class"] = (meta_rf.predict(meta_test_X)).astype(int)
test_df_out.to_csv(os.path.join(OUTPUT_DIR, "test_predictions.csv"), index=False)

