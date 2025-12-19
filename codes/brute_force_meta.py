import os
import shutil

def main():
    # === User parameters ===
    base_output_dir = "brtute-force-42"  # All training folders under this
    data_folder_path = "data/random42"
    iteration = 1  # Single iteration for brute force
    
    # Make base directory
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Paths to scripts
    scripts = {
        "attention": "attention.py",
        "cnn": "cnn.py",
        "gcn": "gcn.py",
        "gmpnn_att": "gmpnn_att.py",
        "meta_input": "meta_input.py",
        "meta_cnn": "meta_cnn.py",
        "meta_attention": "meta_attention.py",
    }
    
    # Starting data
    current_train_des   = f"{data_folder_path}/x_subset_descriptor.csv"
    current_train_graph = f"{data_folder_path}/x_subset.csv"
    train_csv           = "data/x_train.csv"
    test_csv            = "data/x_test.csv"

    # === Brute force ===
    folder = os.path.join(base_output_dir)
    os.makedirs(folder, exist_ok=True)
    
    # Copy current training data
    train_des_copy = os.path.join(folder, f"train_des.csv")
    shutil.copy(current_train_des, train_des_copy)

    train_graph_copy = os.path.join(folder, f"train_graph.csv")
    shutil.copy(current_train_graph, train_graph_copy)

    # === Step 1: Run descriptor models ===
    for model_name in ["attention", "cnn"]:
        model_folder = os.path.join(folder, model_name)
        os.makedirs(model_folder, exist_ok=True)
        os.system(
            f"python {scripts[model_name]} --subset {train_des_copy} "
            f"--test {data_folder_path}/x_test_descriptor.csv --output_folder {model_folder} --iter {iteration}"
        )

    # === Step 1b: Run graph models ===
    for model_name in ["gcn", "gmpnn_att"]:
        model_folder = os.path.join(folder, model_name)
        os.makedirs(model_folder, exist_ok=True)
        os.system(
            f"python {scripts[model_name]} --subset {train_graph_copy} "
            f"--test {data_folder_path}/x_test.csv --output_folder {model_folder} --iter {iteration}"
            )

    # === Step 2: Meta Input ===
    print(f"=== Step 2: Running meta input for brute force {iteration} ===")
    folder = os.path.join(base_output_dir)  # Base folder for outputs
    os.makedirs(folder, exist_ok=True)

    # List of baseline model subfolders
    baseline_models = ["attention", "cnn", "gcn", "gmpnn_att"]
    models_str = " ".join(baseline_models)

    # Run meta_input.py
    os.system(
        f"python {scripts['meta_input']} "
        f"--output_folder {folder} "
        f"--models {models_str} "
        f" --train_csv {train_csv}"
        f" --test_csv {test_csv}"
    )

    # === Step 3: Meta CNN ===
    print(f"=== Step 3: Running meta CNN for brute force {iteration} ===")

    # Use combined train+val meta-input for training
    meta_input_file = os.path.join(folder, "meta_input_trainval.csv")
    meta_test_file  = os.path.join(folder, "meta_input_test.csv")

    meta_cnn_out = os.path.join(folder, "meta_cnn")
    os.makedirs(meta_cnn_out, exist_ok=True)

    os.system(
        f"python {scripts['meta_cnn']} --input {meta_input_file} --test {meta_test_file} --output_folder {meta_cnn_out} --model_type cnn --iter {iteration}"
    )


    # === Step 4: Meta Attention ===
    print(f"=== Step 4: Running meta Attention for brute force {iteration} ===")

    meta_attention_out = os.path.join(folder, "meta_attention")
    os.makedirs(meta_attention_out, exist_ok=True)

    os.system(
        f"python {scripts['meta_attention']} --input {meta_input_file} --test {meta_test_file} --output_folder {meta_attention_out} --model_type attention --iter {iteration}"
    )


if __name__ == "__main__":
    main()