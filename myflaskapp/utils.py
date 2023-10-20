import numpy as np
from rdkit.Chem import rdMolDescriptors, MolFromSmiles, rdmolfiles, rdmolops
from sklearn.model_selection import train_test_split
YOUR_INPUT_SHAPE = 50
YOUR_VOCAB_SIZE = 30
def fingerprint_features(smile_string, radius=2, size=2048):
    mol = MolFromSmiles(smile_string)
    new_order = rdmolfiles.CanonicalRankAtoms(mol)
    mol = rdmolops.RenumberAtoms(mol, new_order)
    return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius,
                                                          nBits=size,
                                                          useChirality=True,
                                                          useBondTypes=True,
                                                          useFeatures=False
                                                          )


def preprocess_smiles(smiles, input_shape=50, vocab_size=30):
    encoded_smiles = np.zeros((input_shape, vocab_size))
    for i, char in enumerate(smiles):
        if i >= input_shape:
            break
        char_index = ord(char) % vocab_size
        encoded_smiles[i, char_index] = 1
    return encoded_smiles
