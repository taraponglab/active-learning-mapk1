import os
import pandas as pd
import argparse

def generate_meta_inputs(output_folder, model_names, train_csv, test_csv,
                         id_col="PUBCHEM_CID", label_col="Label"):
    """
    Generate meta-input CSVs for meta-models.
    - Combines train + val predictions for each baseline model
    - Aligns Label from master CSV
    - Produces meta-input for test set
    """
    # -----------------------------
    # Combine train + val predictions
    # -----------------------------
    trainval_meta = None
    for model_name in model_names:
        train_path = os.path.join(output_folder, model_name, "train_prob.csv")
        val_path   = os.path.join(output_folder, model_name, "val_prob.csv")

        if not os.path.exists(train_path) or not os.path.exists(val_path):
            print(f"⚠️ Missing train/val results for {model_name}, skipping.")
            continue

        train_df = pd.read_csv(train_path)
        val_df   = pd.read_csv(val_path)

        combined = pd.concat([train_df, val_df], ignore_index=True)
        combined = combined[[id_col, 'y_prob']]
        combined = combined.rename(columns={'y_prob': f"{model_name}_prob"})

        if trainval_meta is None:
            trainval_meta = combined
        else:
            trainval_meta = trainval_meta.merge(combined, on=id_col, how='inner')

    if trainval_meta is not None:
        # Merge Labels from master CSV
        master_df = pd.read_csv(train_csv)
        trainval_meta = trainval_meta.merge(master_df[[id_col, label_col]], on=id_col, how='left')

        trainval_path = os.path.join(output_folder, "meta_input_trainval.csv")
        trainval_meta.to_csv(trainval_path, index=False)
        print(f"✅ Meta trainval input saved: {trainval_path}")
    else:
        print("❌ No train+val predictions found.")


    # -----------------------------
    # Combine test predictions
    # -----------------------------
    test_meta = None
    for model_name in model_names:
        test_path = os.path.join(output_folder, model_name, "test_prob.csv")
        if not os.path.exists(test_path):
            print(f"⚠️ Missing test results for {model_name}, skipping.")
            continue

        test_df = pd.read_csv(test_path)
        test_df = test_df[[id_col, 'y_prob']]
        test_df = test_df.rename(columns={'y_prob': f"{model_name}_prob"})

        if test_meta is None:
            test_meta = test_df
        else:
            test_meta = test_meta.merge(test_df, on=id_col, how='inner')

    if test_meta is not None:
        # Add labels from master CSV (filter only for IDs in test set)
        master_df = pd.read_csv(test_csv)
        test_meta = test_meta.merge(master_df[[id_col, label_col]], on=id_col, how='left')

        test_path = os.path.join(output_folder, "meta_input_test.csv")
        test_meta.to_csv(test_path, index=False)
        print(f"✅ Meta test input saved: {test_path}")
    else:
        print("❌ No test predictions found.")

# =========================================
# CLI
# =========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_folder", type=str, required=True,
                        help="Folder where baseline model outputs are saved")
    parser.add_argument("--models", type=str, nargs='+', required=True,
                        help="List of baseline model names")
    parser.add_argument("--train_csv", type=str, required=True,
                        help="CSV with Label column")
    parser.add_argument("--test_csv", type=str, required=True,
                        help="CSV with Label column")
    parser.add_argument("--id_col", type=str, default="PUBCHEM_CID", help="ID column name")
    parser.add_argument("--label_col", type=str, default="Label", help="Label column name")
    args = parser.parse_args()

    generate_meta_inputs(args.output_folder, args.models, args.train_csv, args.test_csv,
                         args.id_col, args.label_col)
