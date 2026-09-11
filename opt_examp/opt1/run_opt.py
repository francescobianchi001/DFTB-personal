#! /usr/bin/python3
# run from the repository root:  python3 opt_examp/run_opt.py <name>
# (Hamiltonian reads ATOMS_BS / ATOMS_POT / eig_neutral from the cwd)

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from optimisation import optimise, SD

NAMES = ('ethyne', 'benzene', 'butane', 'cyclohexane')


def run(name, method=None, alpha=0.3, ftol=1e-3, nmax=120):
    return optimise(geom=f'opt_examp/{name}.xyz', scale=None,
                    out=f'opt_examp/{name}_opt.xyz',
                    traj=f'opt_examp/{name}_traj.xyz',
                    method=method or SD(), alpha=alpha, ftol=ftol, nmax=nmax)


if __name__ == '__main__':
    run(sys.argv[1] if len(sys.argv) > 1 else 'ethyne')
