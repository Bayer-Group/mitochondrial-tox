from rdkit import Chem
from rdkit.Chem import PandasTools
from rdkit.Chem.MolStandardize.standardize import Standardizer
standardizer = Standardizer(max_tautomers=10)
include_stereoinfo = False

def standardizeMol(mol):
    """
    :param mol: rdkit molecule object
    :return: cleaned rdkit molecule object
    """
    if mol is None:
        return mol
    try:
        mol = standardizer.charge_parent(mol)
        mol = standardizer.isotope_parent(mol)
        if include_stereoinfo is False:
            mol = standardizer.stereo_parent(mol)
        mol = standardizer.tautomer_parent(mol)
        return standardizer.standardize(mol)
    except Chem.rdchem.AtomValenceException:
        return None
    
def prepare_structures(input_df, smiles_column='smiles'):
    '''
    Standardize compounds and get canonical smiles
    :param input_df: dataframe containing the structures to standardize
    :param smiles_column: name of the column (in input_df) containing the input SMILES
    :return: dataframe with column 'canonical_smiles' containing the standardized and canonicalized SMILES
    '''
    df = input_df.copy()
    PandasTools.AddMoleculeColumnToFrame(df, smiles_column,'molecule', includeFingerprints=False)
    len_1 = len(df)
    df.dropna(subset=['molecule'], axis=0, inplace=True)
    len_2 = len(df)
    if len_1-len_2 > 0:
        print(f'No. of missing molecules: {len_1-len_2}')

    # Standardize molecules and get canonical smiles
    for idx in df.index:
        try:
            stand_mol = standardizeMol(df.loc[idx, 'molecule'])
            df.loc[idx, 'canonical_smiles'] = Chem.MolToSmiles(stand_mol, canonical=True)
        except:
            df.loc[idx, 'canonical_smiles'] = None
    
    df = df.drop(['molecule', smiles_column], axis=1)
    df = df.dropna(subset=['canonical_smiles'])
    return df