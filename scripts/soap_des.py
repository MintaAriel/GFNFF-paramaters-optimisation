from ase import Atoms
from ase.io import read
from ase.visualize import view
from dscribe.descriptors import SOAP
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from pathlib import Path

script_dir = Path(__file__).parent
project_root = script_dir.parent

experimental = read(project_root / 'data/experimental.cif')
# experimental_2 =  read('/home/vito/PythonProjects/ASEProject/CARLO/gnff/Fine_tun/exp_2_MUT_2.cif')
experimental_2 = read(project_root / 'tests/single_param_cal/CalcFold/out1.cif')

soap = SOAP(
    species=["C", "H", "O", "N"],  # Elements in your system
    r_cut=5.0,                       # Cutoff radius (Å)
    n_max=8,                         # Radial basis functions
    l_max=6,                         # Angular basis functions
    sigma=0.2,                      # Gaussian smearing width
    rbf="gto",                      # Radial basis function type
    periodic=True,                  # CRITICAL for crystals!
    average="inner",                  # 'off' for per-atom, 'inner' for global
    sparse=False
)

# Compute SOAP descriptors
# Returns: (n_atoms, n_features) matrix
s1 = soap.create(experimental)
s2 = soap.create(experimental_2)

print(f"SOAP shape: {s1.shape}")
print(f"Number of features per atom: {soap.get_number_of_features()}")
print(s1)

similarity = cosine_similarity(s1.reshape(1,-1), s2.reshape(1,-1))

print(similarity)

dot = np.dot(s1, s2)
k = dot**2 / np.sqrt((np.dot(s1, s1)**2)*(np.dot(s2, s2)**2))
print(k)