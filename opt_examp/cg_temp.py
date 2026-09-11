#! /usr/bin/python3
"""TEMPORARY standalone conjugate-gradient driver -- NOT part of optimisation.py.

optimisation.CG.direction() is still left unimplemented for you; this file exists only
to get results now and should be deleted once your own CG lands. It reuses
optimisation.write_xyz / update_geometry and Gradients_opt, and patches nothing.

Run from the repository root:  python3 opt_examp/cg_temp.py <name>
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Hamiltonian import H, Gradients_opt
from optimisation import write_xyz, update_geometry

ROOT = Path(__file__).resolve().parent


class CG:
    """Polak-Ribiere+ : beta = max(0, g.(g-g_old)/|g_old|^2), restart every 3N."""

    def __init__(self, restart=None):
        self.restart = restart
        self.reset()

    def reset(self):
        self.g_old = self.p_old = None
        self.n = 0

    def direction(self, g):
        if self.g_old is None or (self.restart and self.n >= self.restart):
            p, self.n = -g, 0
        else:
            den = float(self.g_old.ravel() @ self.g_old.ravel())
            beta = max(0.0, float(g.ravel() @ (g-self.g_old).ravel())/den) if den else 0.0
            p = -g + beta*self.p_old
            if float(p.ravel() @ g.ravel()) >= 0:      # not downhill -> restart
                p, self.n = -g, 0
        self.g_old, self.p_old = g.copy(), p.copy()
        self.n += 1
        return p


def line_min(E, E0, smax, tol=0.1, nmax=14):
    """Approximate exact line search on energies alone (a gradient costs 6N of these,
    so this is ~10% overhead and CG needs the accuracy)."""
    s1 = smax
    e1 = E(s1)
    if e1 < E0:                                        # expand while improving
        s2, e2 = 2*s1, E(2*s1)
        while e2 < e1 and s2 < 64*smax:
            s1, e1, s2 = s2, e2, 2*s2
            e2 = E(s2)
        a, fa, b, fb, cc, fc = 0.0, E0, s1, e1, s2, e2
    else:                                              # backtrack to a bracket
        a, fa, cc, fc = 0.0, E0, s1, e1
        b = s1
        fb = e1
        for _ in range(nmax):
            b *= 0.35
            fb = E(b)
            if fb < E0:
                break
        else:
            return None, None
        cc, fc = s1, e1
    for _ in range(nmax):                              # golden refine on [a, c]
        if cc-a < tol*max(b, 1e-8):
            break
        if b-a > cc-b:
            u = b - 0.382*(b-a)
        else:
            u = b + 0.382*(cc-b)
        fu = E(u)
        if fu < fb:
            if u < b:
                cc, fc = b, fb
            else:
                a, fa = b, fb
            b, fb = u, fu
        else:
            if u < b:
                a, fa = u, fu
            else:
                cc, fc = u, fu
    return (b, fb) if fb < E0 else (None, None)


def optimise_cg(name, alpha=0.3, ftol=1e-3, maxstep=0.20, nmax=200):
    geom = str(ROOT/f'{name}.xyz')
    out, traj = str(ROOT/f'{name}_opt.xyz'), str(ROOT/f'{name}_traj.xyz')

    mol = H(geom=geom, typor0=True)
    grad = Gradients_opt(mol, alpha=alpha)
    method = CG(restart=3*len(mol.atoms))

    coords = mol.coords[0].copy()
    E = grad.E(coords)
    g = grad.num()
    write_xyz(traj, mol.atoms, coords, f'step 0  E = {E:.8f} Ha')
    print(f'{"step":>4} {"E [Ha]":>16} {"dE [Ha]":>12} {"|g|max":>11} '
          f'{"step[a0]":>10} {"nE":>4}', flush=True)
    print(f'{0:>4} {E:>16.8f} {"":>12} {np.abs(g).max():>11.6f}', flush=True)

    it = nE = 0
    while it < nmax:
        if np.abs(g).max() < ftol:
            break
        p = method.direction(g)
        smax = maxstep/np.abs(p).max()

        cnt = [0]

        def f(s):
            cnt[0] += 1
            return grad.E(coords + s*p)

        s, Enew = line_min(f, E, smax)
        if s is None:
            if method.g_old is None or method.n <= 1:
                print('line search failed from steepest descent, stopping')
                break
            print('line search failed -> CG restart')
            method.reset()
            continue

        it += 1
        nE += cnt[0]
        gmax, Eprev = np.abs(g).max(), E
        coords = update_geometry(mol, coords, p, s, out, f'step {it}  E = {Enew:.8f} Ha')
        E = Enew
        g = grad.num()
        print(f'{it:>4} {E:>16.8f} {E-Eprev:>12.2e} {gmax:>11.6f} '
              f'{s*np.abs(p).max():>10.4f} {cnt[0]:>4}', flush=True)
        write_xyz(traj, mol.atoms, coords,
                  f'step {it}  E = {E:.8f} Ha  |g|max = {np.abs(g).max():.6f}', mode='a')

    gmax = np.abs(g).max()
    write_xyz(out, mol.atoms, coords,
              f'{name} optimised (CG)  E = {E:.8f} Ha  |g|max = {gmax:.6f}')
    print(f'\nconverged: {gmax < ftol}   steps: {it}   gradients: {it+1}   '
          f'line-search energies: {nE}   E = {E:.8f} Ha   |g|max = {gmax:.6f}')
    return coords, E, g


if __name__ == '__main__':
    optimise_cg(sys.argv[1] if len(sys.argv) > 1 else 'ethyne')
