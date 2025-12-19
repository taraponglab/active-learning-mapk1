import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
import networkx as nx
from tqdm import tqdm

# ============================================
# Graph Feature Extraction
# ============================================

def graph_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # ---- Atom lists ----
    atomic_nums, degrees, formal_charges = [], [], []
    chiralities, hcounts, hybridizations, aromatics, masses = [], [], [], [], []

    for atom in mol.GetAtoms():
        atomic_nums.append(atom.GetAtomicNum())
        degrees.append(atom.GetDegree())
        formal_charges.append(atom.GetFormalCharge())
        chiralities.append(int(atom.GetChiralTag()))
        hcounts.append(atom.GetTotalNumHs())
        hybridizations.append(int(atom.GetHybridization()))
        aromatics.append(int(atom.GetIsAromatic()))
        masses.append(atom.GetMass())

    # ---- Bond lists ----
    bond_types, in_ring, stereo, conjugated = [], [], [], []
    for bond in mol.GetBonds():
        bond_types.append(float(bond.GetBondTypeAsDouble()))
        in_ring.append(int(bond.IsInRing()))
        stereo.append(int(bond.GetStereo()))
        conjugated.append(int(bond.GetIsConjugated()))

    # ---- Aggregate features ----
    def agg(x):
        if len(x) == 0:
            return 0, 0, 0, 0  # mean, std, max, min
        return np.mean(x), np.std(x), np.max(x), np.min(x)

    feats = {}

    # Atom aggregates
    feats["atomic_num_mean"], feats["atomic_num_std"], _, _ = agg(atomic_nums)
    feats["degree_mean"], _, feats["degree_max"], _ = agg(degrees)
    feats["formal_charge_sum"] = np.sum(formal_charges)
    feats["num_h_mean"], _, _, _ = agg(hcounts)
    feats["hybridization_mean"], _, _, _ = agg(hybridizations)
    feats["aromatic_atom_count"] = np.sum(aromatics)
    feats["aromatic_atom_fraction"] = np.mean(aromatics) if len(aromatics) > 0 else 0
    feats["mass_mean"], feats["mass_std"], _, _ = agg(masses)
    feats["chirality_mean"], _, feats["chirality_max"], _ = agg(chiralities)
    feats["num_atoms"] = len(atomic_nums)

    # Bond aggregates
    feats["bond_type_mean"], _, feats["bond_type_max"], _ = agg(bond_types)
    feats["ring_bond_fraction"] = np.mean(in_ring) if len(in_ring) > 0 else 0
    feats["stereo_mean"] = np.mean(stereo) if len(stereo) > 0 else 0
    feats["conjugated_fraction"] = np.mean(conjugated) if len(conjugated) > 0 else 0
    feats["num_bonds"] = len(bond_types)

    return feats



# ============================================
# Main Script
# ============================================

if __name__ == "__main__":

    # Load your descriptor dataset (edit the filename here)
    INPUT_FILE = "data/experiment_validation_cleaned.csv"   # <-- your input file
    SMILES_COL = "canonical_smiles"  # <-- change if needed
    OUTPUT_FILE = "data/experiment_validation_graph_rf.csv"

    print(f"📥 Loading {INPUT_FILE} ...")
    df = pd.read_csv(INPUT_FILE)

    # Check SMILES column
    if SMILES_COL not in df.columns:
        raise ValueError(f"❌ Column '{SMILES_COL}' not found in the input CSV.")

    print("🔍 Computing molecular graph descriptors...\n")

    graph_rows = []
    for smi in tqdm(df[SMILES_COL], desc="Processing molecules"):
        feats = graph_features(smi)
        if feats is None:
            feats = {
                # ---- Atom features ----
                "atomic_num_mean": np.nan,
                "atomic_num_std": np.nan,
                "degree_mean": np.nan,
                "degree_max": np.nan,
                "formal_charge_sum": np.nan,
                "num_h_mean": np.nan,
                "hybridization_mean": np.nan,
                "aromatic_atom_count": np.nan,
                "aromatic_atom_fraction": np.nan,
                "mass_mean": np.nan,
                "mass_std": np.nan,
                "chirality_mean": np.nan,
                "chirality_max": np.nan,
                "num_atoms": np.nan,

                # ---- Bond features ----
                "bond_type_mean": np.nan,
                "bond_type_max": np.nan,
                "ring_bond_fraction": np.nan,
                "stereo_mean": np.nan,
                "conjugated_fraction": np.nan,
                "num_bonds": np.nan,
            }
        graph_rows.append(feats)

    graph_df = pd.DataFrame(graph_rows)
    print("\n📊 Graph feature shape:", graph_df.shape)

    # Merge graph features with original descriptors
    merged = pd.concat([df, graph_df], axis=1)

    # Save output
    merged.to_csv(OUTPUT_FILE, index=False)

    print(f"\n✅ Saved new dataset with graph features to: {OUTPUT_FILE}")
    print("Done!")
