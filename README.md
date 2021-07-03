# ts_gen

## Installation on Windows 10 [AV]
Bit fiddly because Tensorflow 1.14 can only be installed on Windows 10 if you have Python 3.5+. So we use Python 3.6 instead.
* Python (version=3.6)
* TensorFlow (version=1.14)
* RDKit (version=2018.09.3)

1. Git clone this repo: `git clone https://github.com/PattanaikL/ts_gen`
2. Install RDKit with conda: `conda create -c rdkit -n tsgen-rdkit-env rdkit` [implicitly uses Python 3.6]
3. Activate the conda environment: `conda activate tsgen-rdkit-env`
4. Update RDKit to version=2018.09.3: `conda install -c rdkit rdkit=2018.09.1` [you may be able to automatically do this in step 2]
5. Install TensorFlow: `conda install tensorflow==1.14`
6. Install pymol `conda install -c schrodinger pymol`
7. Train the model: `python train.py --r_file data/intra_rxns_reactants.sdf --p_file data/intra_rxns_products.sdf --ts_file data/intra_rxns_ts.sdf` -> there should be some niggly code issues to sort out with indentation, zips. If you get the most recent version of my [avishvj] code, you shouldn't have to worry about this once the conda environment is set up.
Once this main train command works, turn attention to getting .ipynb notebook to work. 
8. Install ipykernel: `conda install ipykernel`
9. Install py3Dmol **using pip**: `pip install py3Dmol`

## Some notes for me
- Sign out of Azure from the VSCode command palette.


# From original
Generate 3D transition state geometries with GNNs (Note: python3 pytorch version and integration into [ARC](https://github.com/ReactionMechanismGenerator/ARC) coming soon!)

## Installation (for Ubuntu... I'm assuming?)
Requirements:
* python (version=2.7)
* tensorflow (version=1.14)
* rdkit (version=2018.09.3)
`git clone https://github.com/PattanaikL/ts_gen`

## Usage
To train the model, call the `train.py` script with the following parameters defined. If training with your own data, ensure data is in sdf format and molecules between reactants, products, and transition states are all aligned.

`python train.py --r_file data/intra_rxns_reactants.sdf --p_file data/intra_rxns_products.sdf --ts_file data/intra_rxns_ts.sdf`

To evaluate the trained model, refer to `use_trained_model.ipynb`

## Data structures
Currently, we support sdf integration through rdkit, but all that's required is an rdkit mol. If you have data in xyz format, consider using the [code](https://github.com/jensengroup/xyz2mol) from the Jensen group to convert to rdkit.
