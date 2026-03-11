from pygulp.molecule import fix_mol_gradient
from ase.ga.data import DataConnection
import matplotlib.pyplot as plt
from ase.io.trajectory import Trajectory
import os
from pygulp.relaxation.relax import Gulp_relaxation_noadd
from ase.visualize import view
from huggingface_hub import login

da = DataConnection('/home/vito/uspex_matlab/theo_pyxtal/test_1/theophilline.db')
# da = DataConnection('/home/vito/PythonProjects/ASEProject/EA/data/theophylline/database/theophylline.db')
connection_dir = '/home/vito/PythonProjects/ASEProject/EA/data/theophylline/connections'
work_dir  = '/home/vito/PythonProjects/ASEProject/EA/test/struc-gen/Sym'
# trajectory = Trajectory(os.path.join(work_dir, f'all_relaxed.trajectory'), 'w')
energies = {}

for i in range(2,21):
    atom = da.get_atoms(i)

    optimizer = fix_mol_gradient.GradientDescentGULP(atom, work_dir=work_dir, connections=connection_dir, library='reaxff')
    # optimizer = fix_mol_gradient.GradientDescentUMA(atom, work_dir=work_dir, connections=connection_dir,
    #                                                 login_key='insert yours',
    #                                                 device='cuda')
    # optimizer.eta_t = 20e-3
    # optimizer.eta_r = 5e-4
    try:
        optimizer.asu.cell *= 0.85
        new  = optimizer.run(steps =50,  traj=True)
        energies[i] = optimizer.energies
        steps = [i for i in range(len(optimizer.energies))]
        plt.plot(optimizer.energies)
        plt.scatter(steps, optimizer.energies, label=f'{i}')
        best = optimizer.best_structure
    except Exception as e:
        print(e)
        print(f'structure {i} not suitable')
        best = atom

    # trajectory.write(best)

# trajectory.close()
# view(atom)
# view(optimizer.best_structure)


# plt.ylabel('Energy, eV')
# plt.xlim(0,)
# plt.ylim(-480,0)
# plt.xlabel('cycle')
# plt.legend()
# plt.show()
