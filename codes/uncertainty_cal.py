import pandas as pd
import os
import argparse

def calculate_uncertainty(meta_prob_file):
    """Compute uncertainty based on meta predictions (multi-class compatible)."""
    df = pd.read_csv(meta_prob_file)
    if 'meta_prob' not in df.columns:
        raise ValueError("Column 'meta_prob' not found in meta probability file")
    
    # Compute class 0 probability
    df['meta_prob_0'] = 1 - df['meta_prob']
    
    # Take the maximum probability per row
    df['most_likely_prob'] = df[['meta_prob', 'meta_prob_0']].max(axis=1)
    
    # Uncertainty = 1 - most likely probability
    df['uncertain_prob'] = 1 - df['most_likely_prob']
    
    # Distance to 0.5 for sorting
    df['distance_to_0.5'] = abs(df['uncertain_prob'] - 0.5)
    
    # Drop helper column
    df = df.drop(columns=['most_likely_prob'])
    
    # Sort by uncertainty
    df = df.sort_values(by='distance_to_0.5')
    
    return df[['PUBCHEM_CID', 'meta_prob', 'meta_prob_0', 'uncertain_prob', 'distance_to_0.5']]

def select_top_uncertain(uncertainty_df, top_percent, iteration, file_path):
    """Select top uncertain compounds based on distance to 0.5."""
    n_select = max(1, int(len(uncertainty_df) * top_percent))
    top_uncertain = uncertainty_df.head(n_select)
    
    top_uncertain_file = os.path.join(file_path, f"top_uncertain_samples_iter{iteration}.csv")
    top_uncertain.to_csv(top_uncertain_file, index=False)
    print(f"[INFO] Top {n_select} uncertain compounds saved to {top_uncertain_file}")
    
    return top_uncertain

def split_dataset(all_data_file, previous_subset_df, top_uncertain_df, iteration, file_path):
    """Split full dataset into top-uncertain subset and remaining pool."""
    all_data = pd.read_csv(all_data_file)
    merged = all_data.merge(top_uncertain_df[['PUBCHEM_CID']], on='PUBCHEM_CID', how='left', indicator=True)
    top_subset = merged[merged['_merge'] == 'both'].drop(columns=['_merge'])
    remaining_pool = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])

    top_file = os.path.join(file_path, f"subset_top_iter{iteration}.csv")
    remaining_file = os.path.join(file_path, f"pool_remaining_iter{iteration}.csv")
    top_subset.to_csv(top_file, index=False)
    remaining_pool.to_csv(remaining_file, index=False)
    print(f"[INFO] Dataset split completed: {len(top_subset)} top, {len(remaining_pool)} remaining")

    top_df = pd.read_csv(top_file)
    top_list = all_data[all_data['PUBCHEM_CID'].isin(top_df['PUBCHEM_CID'])]

    prev_df = pd.read_csv(previous_subset_df)
    combined_df = pd.concat([top_list, prev_df], ignore_index=True)
    merged_df_file = os.path.join(file_path, f"subset_merged_iter{iteration}.csv")
    combined_df.to_csv(merged_df_file, index=False)

    return top_file, remaining_file

def split_descriptors(all_descriptors_file, top_file, remaining_file, previous_descriptors_file, iteration, file_path):
    """Split descriptors for top and remaining compounds, merge with previous subset descriptors."""
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

    print(f"[INFO] Descriptor splits saved. Combined descriptors total rows: {len(combined_desc)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", required=True, help="Folder containing pool prediction CSVs")
    parser.add_argument("--all_data_file", required=True, help="Current pool CSV file")
    parser.add_argument("--previous_subset_file", required=True, help="Previous subset CSV file")
    parser.add_argument("--previous_descriptors_file", required=True, help="Previous subset descriptors CSV file")
    parser.add_argument("--all_data_descriptors_file", required=True, help="Current pool descriptors CSV file")
    parser.add_argument("--meta_prob_file", required=True, help="CSV file with meta predictions")
    parser.add_argument("--iteration", type=int, required=True, help="Iteration number")
    parser.add_argument("--top_percent", type=float, default=0.05, help="Top % of uncertain samples to select")
    args = parser.parse_args()

    file_path = args.file_path
    iteration = args.iteration

    # Step 1: Compute uncertainty
    uncertainty_df = calculate_uncertainty(args.meta_prob_file)

    # Step 2: Select top uncertain compounds
    top_uncertain_df = select_top_uncertain(uncertainty_df, args.top_percent, iteration, file_path)

    # Step 3: Split dataset into top-uncertain and remaining pool
    top_file, remaining_file = split_dataset(args.all_data_file, args.previous_subset_file,  top_uncertain_df, iteration, file_path)

    # Step 4: Split descriptors accordingly and merge with previous subset descriptors
    split_descriptors(args.all_data_descriptors_file, top_file, remaining_file, args.previous_descriptors_file, iteration, file_path)