import os
import shutil

def main():
    # === User parameters ===
    start_iter = 1
    num_iterations = 4
    base_output_dir = "iterations-42"  # All iteration folders under this
    data_folder_path = "data/random42"
    
    # Make base directory
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Paths to scripts
    scripts = {
        "attention": "attention.py",
        "cnn": "cnn.py",
        "gcn": "gcn.py",
        "gmpnn_att": "gmpnn_att.py",
        "meta_input": "meta_input.py",
        "meta_attention": "meta_attention.py",
        "pool_pred": "pool_pred.py",
        "margin_cal": "margin_cal.py"
    }
    
    # Starting data
    current_train_des   = f"{data_folder_path}/subset_merged_descriptors_iter0.csv"
    current_pool_des    = f"{data_folder_path}/pool_remaining_descriptors_iter0.csv"
    current_train_graph = f"{data_folder_path}/subset_merged_iter0.csv"
    current_pool_graph  = f"{data_folder_path}/pool_remaining_iter0.csv"
    train_csv           = "data/x_train.csv"
    test_csv            = "data/x_test.csv"
    
    # === Loop ===
    for iteration in range(start_iter, num_iterations + 1):
        print(f"\n=== Iteration {iteration} ===\n")
        
        iter_folder = os.path.join(base_output_dir, f"iteration_{iteration}")
        os.makedirs(iter_folder, exist_ok=True)
        
        # Copy current training/pool data
        train_des_copy = os.path.join(iter_folder, f"subset{iteration}_des.csv")
        pool_des_copy  = os.path.join(iter_folder, f"pool{iteration}_des.csv")
        shutil.copy(current_train_des, train_des_copy)
        shutil.copy(current_pool_des, pool_des_copy)
    
        train_graph_copy = os.path.join(iter_folder, f"subset{iteration}_graph.csv")
        pool_graph_copy  = os.path.join(iter_folder, f"pool{iteration}_graph.csv")
        shutil.copy(current_train_graph, train_graph_copy)
        shutil.copy(current_pool_graph, pool_graph_copy)

        # === Step 1: Run descriptor models ===
        for model_name in ["attention", "cnn"]:
            model_folder = os.path.join(iter_folder, model_name)
            os.makedirs(model_folder, exist_ok=True)
            os.system(
                f"python {scripts[model_name]} --subset {train_des_copy} "
                f"--test {data_folder_path}/x_test_descriptor.csv --output_folder {model_folder} --iter {iteration}"
            )

        # === Step 1b: Run graph models ===
        for model_name in ["gcn", "gmpnn_att"]:
            model_folder = os.path.join(iter_folder, model_name)
            os.makedirs(model_folder, exist_ok=True)
            os.system(
                f"python {scripts[model_name]} --subset {train_graph_copy} "
                f"--test {data_folder_path}/x_test.csv --output_folder {model_folder} --iter {iteration}"
                )

        # === Step 2: Meta Input ===
        print(f"=== Step 2: Running meta input for brute force {iteration} ===")
        folder = os.path.join(iter_folder)  # Base folder for outputs
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

        # === Step 3: Meta Attention ===
        print(f"=== Step 3: Running meta Attention for brute force {iteration} ===")

        # Use combined train+val meta-input for training
        meta_input_file = os.path.join(folder, "meta_input_trainval.csv")
        meta_test_file  = os.path.join(folder, "meta_input_test.csv")

        meta_cnn_out = os.path.join(folder, "meta_attention")
        os.makedirs(meta_cnn_out, exist_ok=True)

        os.system(
            f"python {scripts['meta_attention']} --input {meta_input_file} --test {meta_test_file} --output_folder {meta_cnn_out} --model_type attention --iter {iteration}"
        )

        # === Step 4: Pool prediction ===
        print(f"=== Step 4: Running pool prediction for iteration {iteration} ===")
        # Run pool prediction for descriptors and graphs
        pool_desc_file  = pool_des_copy      # copied pool descriptors
        pool_graph_file = pool_graph_copy    # copied pool graph

        os.system(
            f"python {scripts['pool_pred']} --input_folder {iter_folder} --pool_desc {pool_desc_file} --pool_graph {pool_graph_file}"
        )

        # === Step 5: Strategy ===
        os.system(
            f"python {scripts['margin_cal']} "
            f"--file_path {iter_folder} "
            f"--all_data_file {pool_graph_copy} "
            f"--previous_subset_file {train_graph_copy} "
            f"--previous_descriptors_file {current_train_des} "
            f"--all_data_descriptors_file {pool_des_copy} "
            f"--meta_prob_file {os.path.join(folder, 'pool_pred_meta.csv')} "
            f"--iteration {iteration} "
            f"--top_percent 0.05"
        )
    
        # === Step 6b: Update for next iteration (descriptors) ===
        current_train_des = os.path.join(iter_folder, f"subset_merged_descriptors_iter{iteration}.csv")
        current_pool_des  = os.path.join(iter_folder, f"pool_remaining_descriptors_iter{iteration}.csv")
    
        # === Step 6c: Update for next iteration (graphs) ===
        current_train_graph = os.path.join(iter_folder, f"subset_merged_iter{iteration}.csv")  # same IDs as descriptors
        current_pool_graph  = os.path.join(iter_folder, f"pool_remaining_iter{iteration}.csv")  # same IDs as descriptors
    
        print(f"=== Iteration {iteration} completed ===\n")

if __name__ == "__main__":
    main()