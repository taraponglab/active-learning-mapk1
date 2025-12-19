import os
import argparse
import pandas as pd
import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from joblib import dump
from datetime import datetime
import time
import warnings
warnings.filterwarnings("ignore")


# =========================================
# MODEL DEFINITION
# =========================================
def cnn_model(fingerprint_length):
    model = Sequential()
    model.add(Conv1D(32, kernel_size=3, input_shape=(fingerprint_length,1), activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=['accuracy']
    )
    return model


# =========================================
# HELPER FUNCTIONS
# =========================================
def split_data(df, label_col):
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


# =========================================
# MAIN
# =========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=str, required=True, help="Path to subset CSV for training")
    parser.add_argument("--test", type=str, required=True, help="Path to test CSV")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to save outputs")
    parser.add_argument("--iter", type=int, default=1, help="Iteration number")
    args = parser.parse_args()

    # Paths
    subset_path = args.subset
    test_path = args.test
    output_folder = args.output_folder

    os.makedirs(output_folder, exist_ok=True)
    print(f"Saving outputs to: {output_folder}\n")

    # =========================================
    # COLUMN NAMES (single place to modify)
    # =========================================
    LABEL_COL = "Label"
    ID_COL = "PUBCHEM_CID"
    DROP_COLS = [LABEL_COL, ID_COL]

    # =========================================
    # LOAD DATA
    # =========================================
    df = pd.read_csv(subset_path)
    test_df = pd.read_csv(test_path)

    train_df, val_df = split_data(df, label_col=LABEL_COL)

    X_train = train_df.drop(columns=DROP_COLS)
    y_train = train_df[LABEL_COL]

    X_val = val_df.drop(columns=DROP_COLS)
    y_val = val_df[LABEL_COL]

    X_test = test_df.drop(columns=DROP_COLS)
    y_test = test_df[LABEL_COL]

    # Convert to numpy arrays
    X_train_np = X_train.values
    X_val_np = X_val.values
    X_test_np = X_test.values

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_np)
    X_val_scaled = scaler.transform(X_val_np)
    X_test_scaled = scaler.transform(X_test_np)

    dump(scaler, os.path.join(output_folder, "scaler_model.joblib"))

    fingerprint_length = X_train_scaled.shape[1]

    # Add channel dimension for CNN
    X_train_scaled = np.expand_dims(X_train_scaled, axis=-1)
    X_val_scaled = np.expand_dims(X_val_scaled, axis=-1)
    X_test_scaled = np.expand_dims(X_test_scaled, axis=-1)

    # =========================================
    # TRAINING
    # =========================================
    start_train_time = time.time()

    model = cnn_model(fingerprint_length)
    model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=20, batch_size=32
    )

    model.save(os.path.join(output_folder, "cnn_model.keras"))

    end_train_time = time.time()
    training_time = end_train_time - start_train_time

    # =========================================
    # SAVE PREDICTIONS
    # =========================================
    pred_start_time = time.time()

    def save_preds(df_ref, preds, fname):
        out = pd.DataFrame({
            ID_COL: df_ref[ID_COL].values,
            "y_pred": preds.flatten()
        })
        out.to_csv(os.path.join(output_folder, fname), index=False)
    def save_probs(df_ref, probs, fname):
        out = pd.DataFrame({
            ID_COL: df_ref[ID_COL].values,
            "y_prob": probs.flatten()
        })
        out.to_csv(os.path.join(output_folder, fname), index=False)

    # Train
    acc, auc, auprc, mcc, prob, pred = evaluate_model(model, X_train_scaled, y_train)
    save_probs(train_df, prob, "train_prob.csv")
    save_preds(train_df, pred, "train_pred.csv")

    # Validation
    acc, auc, auprc, mcc, prob, pred = evaluate_model(model, X_val_scaled, y_val)
    save_probs(val_df, prob, "val_prob.csv")
    save_preds(val_df, pred, "val_pred.csv")

    # Test
    acc, auc, auprc, mcc, prob, pred = evaluate_model(model, X_test_scaled, y_test)
    save_probs(test_df, prob, "test_prob.csv")
    save_preds(test_df, pred, "test_pred.csv")

    end_eval_time = time.time()
    evaluation_time = end_eval_time - pred_start_time

    # === Record Training Information ===
    total_time = training_time + evaluation_time
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Summary metrics
    summary = pd.DataFrame([{
        "Iteration": args.iter,
        "Accuracy": acc,
        "AUROC": auc,
        "AUPRC": auprc,
        "MCC": mcc
    }])
    summary.to_csv(os.path.join(output_folder, "summary_metrics.csv"), index=False)

    # =========================================
    # APPEND TRAINING INFO
    # =========================================
    training_info = {
        "Iteration": [args.iter],
        "Total training time(s)": [total_time],
        "Model training time(s)": [training_time],
        "Prediction time(s)": [evaluation_time],
        "Timestamp": [timestamp]
    }
    training_info_df = pd.DataFrame(training_info)

    training_info_csv = os.path.join(args.output_folder, "training_info.csv")

    if os.path.exists(training_info_csv):
        existing = pd.read_csv(training_info_csv)
        combined = pd.concat([existing, training_info_df], ignore_index=True)
    else:
        combined = training_info_df

    combined.to_csv(training_info_csv, index=False)

    print(f"Saved training info to {training_info_csv}")
    print("\nDONE ✅")


if __name__ == "__main__":
    main()


#python run_cnn.py \
#  --subset entropy4/descriptor/x_subset.csv \
#  --test initial_al/descriptor/x_test.csv \
#  --output_folder results_cnn \
#  --iter 3