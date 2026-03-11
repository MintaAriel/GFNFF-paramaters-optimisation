from pygulp.molecule import fix_mol_gradient
from ase.ga.data import DataConnection
from ase.io import read, write
import matplotlib.pyplot as plt
from ase.io.trajectory import Trajectory
from ase.visualize import view
from ase.io import read
from pygulp.relaxation.relax import Gulp_relaxation_noadd
import numpy as np
from pygulp.io.read_gulp import read_results

from pathlib import Path
import os

da = DataConnection('/home/vito/uspex_matlab/theo_pyxtal/test_1/theophilline.db')
# da = DataConnection('/home/vito/PythonProjects/ASEProject/EA/data/theophylline/database/theophylline.db')
connection_dir = '/home/vito/PythonProjects/ASEProject/EA/data/theophylline/connections'
work_dir  = '/home/vito/PythonProjects/ASEProject/EA/test/struc-gen/Sym/small'

reaxff = Trajectory('/home/vito/PythonProjects/ASEProject/EA/test/struc-gen/Sym/all_relaxed_small.trajectory')



gulp_input = (f"opti    gradient conp conse qok c6 conp prop gfnff gwolf noauto\n"
                  f"gfnff_scale 0.8 1.343 0.727 1.0 2.859\n"
                  f"maths mrrr\n"
                  f"pressure 0 GPa"
                  )
options = (
    "output movie cif out1.cif\n"
    "maxcycle 300\n"
    "gtol 0.00001"
)

cal_dir = work_dir + '/single_param_cal'




for i in range(1,20):
    relax = Gulp_relaxation_noadd(path=cal_dir,
                                  library=None,
                                  gulp_keywords=gulp_input,
                                  gulp_options=options)

    molecule_ini = da.get_atoms(i+1)
    molecule_opt = reaxff[i-1]



    atom_ini = relax.use_gulp(molecule_ini)
    calc_ini = read_results(f"{cal_dir}/CalcFold/ginput1.got")

    atom_opt = relax.use_gulp(molecule_opt)
    calc_opt = read_results(f"{cal_dir}/CalcFold/ginput1.got")
    print(f'Structure {i} \n step | initial | reaxff optimised')
    print('step 0 :',calc_ini['energy'][0], calc_opt['energy'][0])
    print('step 300:',calc_ini['energy'][2], calc_opt['energy'][2])
    # print(molecule_ini.get_volume(), molecule_opt.get_volume())