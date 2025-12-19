import pandas as pd
import numpy as np
import os
import argparse


# --------------------------------------------------------------
# STEP 1 — MARGIN-BASED UNCERTAINTY
# --------------------------------------------------------------

def calculate_margin(meta_prob_file):
    """
    Compute margin uncertainty from probability predictions.
    Margin = difference between highest and second-highest probability.
    Smaller margin = more uncertain.
    """
    df = pd.read_csv(meta_prob_file)
    if 'meta_prob' not in df.columns:
        raise ValueError("Column 'meta_prob' not found in meta probability file")
    # Compute class 0 probability
    df['meta_prob_0'] = 1 - df['meta_prob']

    # Check required columns
    required = {"PUBCHEM_CID", "meta_prob", "meta_prob_0"}
    if not required.issubset(df.columns):
        raise ValueError(f"❌ Missing required columns: {required - set(df.columns)}")

    # For binary classification with two probabilities: p0 and p1
    proba = df[['meta_prob_0', 'meta_prob']].values  # or however your columns are named

    # Top two probabilities using partition
    part = np.partition(-proba, 1, axis=1)
    max_p = -part[:, 0]       # largest probability
    second_p = -part[:, 1]    # second largest probability
    margins = max_p - second_p  # Margin = difference between top two (positive)
    df['y_prob_margin'] = margins
    
    # Sort ascending → smallest margin = most uncertain
    df = df.sort_values(by='y_prob_margin', ascending=True)

    return df


# --------------------------------------------------------------
# STEP 2 — SELECT LOWEST MARGINS
# --------------------------------------------------------------

def select_top_uncertain(uncertainty_df, top_percent, iteration, file_path):

    n_select = max(1, int(len(uncertainty_df) * top_percent))
    top_uncertain = uncertainty_df.head(n_select)

    out_file = os.path.join(file_path, f"top_margin_samples_iter{iteration}.csv")
    top_uncertain.to_csv(out_file, index=False)
    print(f"[INFO] Top {n_select} margin-uncertain compounds saved → {out_file}")

    return top_uncertain


# --------------------------------------------------------------
# STEP 3 — SPLIT DATASET INTO SELECTED + REMAINING
# --------------------------------------------------------------

def split_dataset(all_data_file, previous_subset_file, top_uncertain_df, iteration, file_path):

    all_data = pd.read_csv(all_data_file)

    merged = all_data.merge(
        top_uncertain_df[['PUBCHEM_CID']], 
        on='PUBCHEM_CID', 
        how='left', 
        indicator=True
    )

    top_subset = merged[merged['_merge'] == 'both'].drop(columns=['_merge'])
    remaining_pool = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])

    top_file = os.path.join(file_path, f"subset_top_iter{iteration}.csv")
    remaining_file = os.path.join(file_path, f"pool_remaining_iter{iteration}.csv")

    top_subset.to_csv(top_file, index=False)
    remaining_pool.to_csv(remaining_file, index=False)
    print(f"[INFO] Dataset split: {len(top_subset)} top, {len(remaining_pool)} remaining")

    # Merge with previous subset
    prev_df = pd.read_csv(previous_subset_file)
    combined_df = pd.concat([top_subset, prev_df], ignore_index=True)

    merged_output = os.path.join(file_path, f"subset_merged_iter{iteration}.csv")
    combined_df.to_csv(merged_output, index=False)

    return top_file, remaining_file


# --------------------------------------------------------------
# STEP 4 — SPLIT DESCRIPTORS FOR NEW + REMAINING
# --------------------------------------------------------------

def split_descriptors(all_descriptors_file, top_file, remaining_file, previous_descriptors_file, iteration, file_path):

    descriptors_df = pd.read_csv(all_descriptors_file)
    top_df = pd.read_csv(top_file)
    remaining_df = pd.read_csv(remaining_file)

    top_desc = descriptors_df[descriptors_df['PUBCHEM_CID'].isin(top_df['PUBCHEM_CID'])]
    remaining_desc = descriptors_df[descriptors_df['PUBCHEM_CID'].isin(remaining_df['PUBCHEM_CID'])]

    top_desc_file = os.path.join(file_path, f"subset_top_descriptors_iter{iteration}.csv")
    remaining_desc_file = os.path.join(file_path, f"pool_remaining_descriptors_iter{iteration}.csv")

    top_desc.to_csv(top_desc_file, index=False)
    remaining_desc.to_csv(remaining_desc_file, index=False)

    prev_desc = pd.read_csv(previous_descriptors_file)
    combined_desc = pd.concat([top_desc, prev_desc], ignore_index=True)

    merged_desc_file = os.path.join(file_path, f"subset_merged_descriptors_iter{iteration}.csv")
    combined_desc.to_csv(merged_desc_file, index=False)

    print(f"[INFO] Descriptor split done. Combined descriptor size: {len(combined_desc)}")


# --------------------------------------------------------------
# MAIN
# --------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--file_path", required=True)
    parser.add_argument("--all_data_file", required=True)
    parser.add_argument("--previous_subset_file", required=True)
    parser.add_argument("--previous_descriptors_file", required=True)
    parser.add_argument("--all_data_descriptors_file", required=True)
    parser.add_argument("--meta_prob_file", required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--top_percent", type=float, default=0.05)

    args = parser.parse_args()

    # 1. Margin uncertainty
    margin_df = calculate_margin(args.meta_prob_file)

    # 2. Select top uncertain
    top_uncertain_df = select_top_uncertain(
        margin_df, args.top_percent, args.iteration, args.file_path
    )

    # 3. Dataset split
    top_file, remaining_file = split_dataset(
        args.all_data_file,
        args.previous_subset_file,
        top_uncertain_df,
        args.iteration,
        args.file_path
    )

    # 4. Descriptor split
    split_descriptors(
        args.all_data_descriptors_file,
        top_file,
        remaining_file,
        args.previous_descriptors_file,
        args.iteration,
        args.file_path
    )
