import pandas as pd
from rdkit.Chem import AllChem as Chem

def remove_missing_data(df, smiles):
    '''
    Remove missing data from specific dataframe and columns
    -------
    Parameters:
    df      : DataFrame containing chemical data, including a 'SMILES' column.
    smiles  : SMILES column
    -------
    Return:
    df_select   : DataFrame without missing data from specific columns.
    '''
    df_select = df.dropna(subset=[smiles])
    number_row_before = len(df)
    number_row_after  = len(df_select)
    print('Remove ', str(number_row_before - number_row_after), ' MISSING data. Only ', number_row_after, ' remaining')
    return df_select

def canonical_smiles(df, smiles_col):
    '''
    Turn your SMILES to canonical SMILES with isomeric
    ------
    Parameters:
    df      : DataFrame containing chemical data, including a 'SMILES' column.
    smiles  : SMILES column
    ------
    Return: 
    df_select   : DataFrame with canonical SMILES
    '''
    def to_canonical(smiles):
        mol= Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Chem.MolToSmiles(mol, isomericSmiles=True)
        else:
            return None
    df['canonical_smiles'] = df[smiles_col].apply(to_canonical)
    df_select = df.dropna(subset=['canonical_smiles'])
    df_select= df_select.drop(columns=[smiles_col], axis=1)
    
    number_row_before = len(df)
    number_row_after  = len(df_select)
    print('Remove ', str(number_row_before - number_row_after), ' INVALID data. Only ', number_row_after, ' remaining')
    print("Finish converting SMILES to canonical SMILES (isomeric)")
    
    return df_select

def has_carbon_atoms(smiles):
    '''
    Helper function.
    Check whether SMILES contain carbon atoms.
    if molecule contains carbon atoms, it will return TRUE, else FALSE
    ------
    Parameters
    smiles: SMILES STRING
    '''
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        # List of carbon atoms (atomic number 6)
        carbon_atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6]
        return len(carbon_atoms) > 0
    return False

def remove_inorganic(df, smiles_col):
    '''
    Remove inorganics (no carbon) using TRUE/FALSE
    Select only organics (has carbon atom == TRUE)
    ------
    Parameters:
    df: Dataframe
    smiles_col: SMILES column in dataframe (df)
    '''
    has_carbon = df[smiles_col].apply(has_carbon_atoms)
    df_select = df[has_carbon == True]
    number_row_before = len(df)
    number_row_after  = len(df_select)
    print('Remove ', str(number_row_before - number_row_after), ' INORGANICS data. Only ', number_row_after, ' remaining')
    return df_select

def remove_mixtures(df, smiles_col):
    '''
    Check if molecule is a mixture using '.' as separator and store in is_mixture.
    Drop mixture if SMILES contain . (is_mixture == FALSE)
    ------
    Parameters:
    df: Dataframe
    smiles_col: SMILES column in dataframe (df)
    '''
    is_mixture = df[smiles_col].apply(lambda x: '.' in x)
    df_select = df[is_mixture == False]
    number_row_before = len(df)
    number_row_after  = len(df_select)
    print('Remove ', str(number_row_before - number_row_after), ' MIXTURES data. Only ', number_row_after, ' remaining')
    return df_select

def process_duplicate(df, smiles_col, remove_duplicate=False):
    '''
    Get duplicate from SMILES
    Average acvalues_um from SMILED
    ------
    Parameters:
    df: Dataframe
    smiles_col: SMILES column in df
    '''
    duplicate_entries = df[df.duplicated(subset=smiles_col, keep = False)].sort_values(smiles_col)
    #save duplicate for inspection
    duplicate_entries.to_csv('duplicates.csv')
    #specify
    if remove_duplicate == True:
        df_no_duplicate = df.drop_duplicates(subset=[smiles_col], keep=False)
    else:
        df_no_duplicate = df.groupby(smiles_col).mean().reset_index()
    number_duplicate = len(duplicate_entries)
    number_row_after = len(df_no_duplicate)
    print('This dataframe contains ', number_duplicate, ' duplicate entries')
    print('After remove DUPLICATES data, this dataframe contain ', number_row_after, ' data')
    return df_no_duplicate

def main():
    print("This software is starting to remove missing data, inorganics, mixtures, select columns, average duplicates, computed pIC50 from uM ..")
    file_name = 'smiles-naturalproduct-np-mrd/smiles_NP0300001_NP0350000.csv'  #input("Type your CSV file name (including extension): ")
    df = pd.read_csv(file_name)
    print("#"*100)
    print("Preprocessing data")
    print("Total datapoint = ", len(df))
    df_select = remove_missing_data(df, 'SMILES')
    df_select = canonical_smiles(df_select, 'SMILES')
    df_select = remove_inorganic(df_select, 'canonical_smiles')
    df_select = remove_mixtures(df_select, 'canonical_smiles')
    df_select = process_duplicate(df_select, 'canonical_smiles',remove_duplicate=True)
    print("Save data")
    df_select.to_csv('smiles-naturalproduct-np-mrd/cleaned/smiles_NP0300001_NP0350000_cleaned.csv')
    print("Example of results")
    print(df_select)
    print("#"*100)
    print('Finished!')

if __name__ == "__main__":
    main()
