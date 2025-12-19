import pandas as pd
import os
import argparse

# --------------------------------------------------------------
# RANDOM SELECTION
# --------------------------------------------------------------
def select_top_random(all_data_df, top_percent, iteration, file_path):
    """
    Randomly select a percentage of compounds from all_data_df.
    """
    n_select = max(1, int(len(all_data_df) * top_percent))
    top_random = all_data_df.sample(n=n_select, random_state=42)  # reproducible

    out_file = os.path.join(file_path, f"top_random_samples_iter{iteration}.csv")
    top_random.to_csv(out_file, index=False)
    print(f"[INFO] Top {n_select} randomly selected compounds saved → {out_file}")

    return top_random

# --------------------------------------------------------------
# SPLIT DATASET INTO SELECTED + REMAINING
# --------------------------------------------------------------
def split_dataset(all_data_file, previous_subset_file, top_df, iteration, file_path):
    all_data = pd.read_csv(all_data_file)

    merged = all_data.merge(
        top_df[['PUBCHEM_CID']],
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
# SPLIT DESCRIPTORS FOR NEW + REMAINING
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
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--top_percent", type=float, default=0.05)

    args = parser.parse_args()

    # Load all data and select randomly
    all_data_df = pd.read_csv(args.all_data_file)
    top_df = select_top_random(all_data_df, args.top_percent, args.iteration, args.file_path)

    # Dataset split
    top_file, remaining_file = split_dataset(
        args.all_data_file,
        args.previous_subset_file,
        top_df,
        args.iteration,
        args.file_path
    )

    # Descriptor split
    split_descriptors(
        args.all_data_descriptors_file,
        top_file,
        remaining_file,
        args.previous_descriptors_file,
        args.iteration,
        args.file_path
    )
