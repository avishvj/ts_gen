from rdkit import Chem
import numpy as np

# constants
BATCH_SIZE = 1
elements = "HCNO"
num_elements = len(elements)

# this func builds edge and vertex features for each molecule as part of a batch
def prepare_batch(batch_mols, MAX_SIZE):

    # Initialization
    size = len(batch_mols) 
    V = np.zeros((size, MAX_SIZE, num_elements+1), dtype=np.float32) # vertices  [MAX_SIZE[5]]
    E = np.zeros((size, MAX_SIZE, MAX_SIZE, 3), dtype=np.float32) # leftmost number in tuple is outermost array
                                                                  # 3 because 3 features: aromatic, bonded, exp(avg dist)
                                                                  # batch index * max atoms * max atoms * 3 edge features
    sizes = np.zeros(size, dtype=np.int32) # populated later on, corresponds to each ts
    coordinates = np.zeros((size, MAX_SIZE, 3), dtype=np.float32) # number of mols in batch * max number of atoms * 3 i.e. (xyz) for each atom in mol

    # Build atom features
    for bx in range(size): # iterate through batch? yes, bx is batch index
        reactant, product = batch_mols[bx] 
        N_atoms = reactant.GetNumAtoms()
        sizes[bx] = int(N_atoms)  # cast to int [can it not be an int?], but basically have each size value as int of number atoms corresponding to reactant

        # topological distances matrix i.e. number of bonds between atoms in mol e.g. molecule v1-v2-v3-v4 will have tdm[1][4]=3
        # also symm matrix
        MAX_D = 10. # i.e. don't have more than 10 bonds between molecules
        D = (Chem.GetDistanceMatrix(reactant) + Chem.GetDistanceMatrix(product)) / 2
        D[D > MAX_D] = MAX_D 

        D_3D_rbf = np.exp( -( (Chem.Get3DDistanceMatrix(reactant) + Chem.Get3DDistanceMatrix(product) ) / 2) )  # lP: squared. AV: [is it?]
        # distance matrix between atoms aka topographic distance matrix aka geometric distance matrix
        # just averaging the distances corresponding to the same atom pairs in reactant and product

        for i in range(N_atoms):
            # Edge features
            for j in range(N_atoms):
                E[bx, i, j, 2] = D_3D_rbf[i][j]
                if D[i][j] == 1.:  # if stays bonded
                    if reactant.GetBondBetweenAtoms(i, j).GetIsAromatic():
                        E[bx, i, j, 0] = 1.
                    E[bx, i, j, 1] = 1. 
                    # so each reaction (reactant-product pair) has 3 features: whether aromatic, whether bond broken/formed, exp(avg dist)

            # Recover coordinates
            # for k, mol_typ in enumerate([reactant, ts, product]):
            pos = reactant.GetConformer().GetAtomPosition(i) 
            np.asarray([pos.x, pos.y, pos.z])
            coordinates[bx, i, :] = np.asarray([pos.x, pos.y, pos.z]) # bx is basically mol_id or rxn_id; each molecule has i atoms with (xyz)

            # Node features: whether HCNO present and then atomic number/10
            atom = reactant.GetAtomWithIdx(i) # get type of atom
            e_ix = elements.index(atom.GetSymbol()) # get chem symbol of atom and corresponding elements index
            V[bx, i, e_ix] = 1. # whether HCNO present
            V[bx, i, num_elements] = atom.GetAtomicNum() / 10. # atomic number/10

    batch_dict = {
        "nodes": V,
        "edges": E,
        "sizes": sizes,
        "coordinates": coordinates
    }
    return batch_dict, batch_mols

def sample_batch(test_data, BATCH_SIZE, MAX_SIZE):
    batches = (len(test_data) - 1) // BATCH_SIZE + 1 # number of batches = (842-1)//(1+1) = 420
    for i in range(batches):
        batch_mols = test_data[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        yield prepare_batch(batch_mols, MAX_SIZE)