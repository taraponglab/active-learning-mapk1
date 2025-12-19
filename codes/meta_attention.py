import os
import time
import argparse
import pandas as pd
import numpy as np
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Dense, Reshape, Attention
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score, average_precision_score
from joblib import dump
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# =========================================
# MODEL DEFINITIONS
# =========================================
def attention_model(fingerprint_length):
    input_layer = Input(shape=(fingerprint_length,))
    dense_layer = Dense(64, activation='relu')(input_layer)
    reshape_layer = Reshape((1, 64))(dense_layer)
    attention_layer = Attention(use_scale=True)([reshape_layer, reshape_layer])
    attention_output = Reshape((64,))(attention_layer)
    output_layer = Dense(1, activation='sigmoid')(attention_output)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# =========================================
# UTILITY FUNCTIONS
# =========================================
def split_data(df, label_col="Label"):
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df[label_col], random_state=42)
    return train_df, val_df

def evaluate_model(model, x, y_true):
    y_prob = model.predict(x)
    y_pred = (y_prob > 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    mcc = matthews_corrcoef(y_true, y_pred)
    return acc, auc, auprc, mcc, y_prob, y_pred

def save_preds(df_ref, probs, preds, id_col, out_path):
    out = pd.DataFrame({
        id_col: df_ref[id_col].values,
        "Probability": probs.flatten(),
        "Prediction": preds.flatten()
    })
    out.to_csv(out_path, index=False)

# =========================================
# MAIN FUNCTION
# =========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV for training")
    parser.add_argument("--test", type=str, help="Path to test CSV")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to save outputs")
    parser.add_argument("--model_type", type=str, default="attention", help="Choose model type")
    parser.add_argument("--iter", type=int, default=1, help="Iteration number")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    # Prepare output folder
    out_folder = args.output_folder
    os.makedirs(out_folder, exist_ok=True)

    LABEL_COL = "Label"
    ID_COL = "PUBCHEM_CID"
    DROP_COLS = [LABEL_COL, ID_COL]

    # Load data
    df = pd.read_csv(args.input)
    train_df, val_df = split_data(df, label_col=LABEL_COL)
    test_df = pd.read_csv(args.test)

    X_train, y_train = train_df.drop(columns=DROP_COLS).values, train_df[LABEL_COL].values
    X_val, y_val = val_df.drop(columns=DROP_COLS).values, val_df[LABEL_COL].values
    X_test, y_test = test_df.drop(columns=DROP_COLS).values, test_df[LABEL_COL].values

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    dump(scaler, os.path.join(out_folder, "scaler_model.joblib"))

    # Choose model
    fingerprint_length = X_train.shape[1]
    if args.model_type == "attention":
        X_train = np.expand_dims(X_train, axis=-1)
        X_val = np.expand_dims(X_val, axis=-1)
        X_test = np.expand_dims(X_test, axis=-1)
        model = attention_model(fingerprint_length)

    # Train model and record time
    start_train_time = time.time()
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=args.epochs, batch_size=args.batch_size)
    end_train_time = time.time()
    training_time = end_train_time - start_train_time
    model.save(os.path.join(out_folder, "meta_attention_model.keras"))
    
    pred_start_time = time.time()
    # === Evaluate and save predictions & metrics ===
    metrics_list = []

    for split_name, X, y, df_ref in zip(
        ["train", "val", "test"],
        [X_train, X_val, X_test],
        [y_train, y_val, y_test],
        [train_df, val_df, test_df]
    ):
        acc, auc, auprc, mcc, probs, preds = evaluate_model(model, X, y)
        save_preds(df_ref, probs, preds, ID_COL, os.path.join(out_folder, f"{split_name}_results.csv"))

        # Store metrics
        metrics_list.append({
            "Split": split_name,
            "ACC": acc,
            "AUROC": auc,
            "AUPRC": auprc,
            "MCC": mcc
        })

        print(f"{split_name.upper()} - ACC:{acc:.4f}, AUROC:{auc:.4f}, AUPRC:{auprc:.4f}, MCC:{mcc:.4f}")

    # Save metrics to CSV
    metrics_df = pd.DataFrame(metrics_list)
    metrics_path = os.path.join(out_folder, f"metrics_iter{args.iter}.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics saved to {metrics_path}")

    end_eval_time = time.time()
    evaluation_time = end_eval_time - pred_start_time

    # === Record Training Information ===
    total_time = training_time + evaluation_time
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Save training info
    train_info_path = os.path.join(args.output_folder, "training_info.csv")
    train_info_df = pd.DataFrame([{
        "Iteration": args.iter,
        "Model_Type": args.model_type,
        "Total training time(s)": [total_time],
        "Model training time(s)": [training_time],
        "Prediction time(s)": [evaluation_time],
        "Timestamp": [timestamp]
    }])
    if os.path.exists(train_info_path):
        existing = pd.read_csv(train_info_path)
        train_info_df = pd.concat([existing, train_info_df], ignore_index=True)
    train_info_df.to_csv(train_info_path, index=False)

    print(f"\nTraining info saved to {train_info_path}")
    print("\nDONE ✅")

if __name__ == "__main__":
    main()
