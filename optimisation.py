#! /usr/bin/python3

import numpy as np
from pathlib import Path
from Hamiltonian import H, Gradients_opt
from read_xyz import ATOM_NAMES, BOHR_TO_ANGS


def write_xyz(path, Z, coords, comment='', mode='w'):
    # coords in bohr, file in Angstrom
    with open(path, mode) as f:
        f.write(f'{len(Z)}\n{comment}\n')
        for Zi, (x, y, z) in zip(Z, np.asarray(coords)*BOHR_TO_ANGS):
            f.write(f'{ATOM_NAMES[int(Zi)].capitalize():<2s}'
                    f'{x:18.10f}{y:18.10f}{z:18.10f}\n')


def prescale(Z, coords, target=1.54):
    # heavy-atom skeleton scaled about its centroid to a mean NN bond of `target` [A];
    # every H rides rigidly with its nearest heavy atom, so C-H stays put
    Z = np.asarray(Z)
    heavy = np.nonzero(Z != 1)[0]
    d = np.linalg.norm(coords[:, None]-coords[None, :], axis=-1)

    dh = d[np.ix_(heavy, heavy)].copy()
    np.fill_diagonal(dh, np.inf)
    nn = dh.min(axis=1).mean()
    f = (target/BOHR_TO_ANGS)/nn

    c0 = coords[heavy].mean(axis=0)
    new = coords.copy()
    new[heavy] = c0 + f*(coords[heavy]-c0)

    light = np.nonzero(Z == 1)[0]
    parent = heavy[np.argmin(d[np.ix_(light, heavy)], axis=1)]
    new[light] = coords[light] + (new[parent]-coords[parent])

    print(f'prescale: mean heavy NN {nn*BOHR_TO_ANGS:.3f} -> '
          f'{target:.3f} A  (factor {f:.4f})')
    return new


def kick(Z, coords, amp=0.1, seed=0):
    # break a planar ring: heavy atoms displaced along the best-fit plane normal,
    # H rides with its parent, so bond lengths survive to first order
    rng = np.random.default_rng(seed)
    Z = np.asarray(Z)
    heavy = np.nonzero(Z != 1)[0]

    p = coords[heavy]-coords[heavy].mean(axis=0)
    n = np.linalg.svd(p)[2][2]                       # normal of the best-fit plane
    dz = rng.normal(0.0, amp/BOHR_TO_ANGS, len(heavy))
    dz -= dz.mean()                                  # no net translation

    new = coords.copy()
    new[heavy] += dz[:, None]*n

    d = np.linalg.norm(coords[:, None]-coords[None, :], axis=-1)
    light = np.nonzero(Z == 1)[0]
    parent = heavy[np.argmin(d[np.ix_(light, heavy)], axis=1)]
    new[light] = coords[light] + (new[parent]-coords[parent])

    print(f'kick: out-of-plane rms {np.sqrt((dz**2).mean())*BOHR_TO_ANGS:.4f} A '
          f'(seed {seed})')
    return new


def update_geometry(mol, coords, p, step, path=None, comment=''):
    # old coords + step along p -> new coords, stored on mol and on disk
    new = coords + step*p
    mol.coords = new[None]
    if path is not None:
        write_xyz(path, mol.atoms, new, comment)
    return new


# ---------------------------------------------------------------- step method

class SD:
    # baseline so the script runs end to end; p.g < 0 always
    def reset(self):
        pass

    def direction(self, g, x):
        return -g


class CG:
    """Nonlinear conjugate gradient.

    direction(g, x) -> search direction p, with g = dE/dR [Ha/bohr] at x.
    Keep whatever state you need (g_old, p_old, counter) on self; reset()
    is called at start and whenever the line search fails, and must drop it.

    Notes for the implementation:
      - Polak-Ribiere with beta = max(0, beta_PR) restarts itself; plain
        Fletcher-Reeves stalls here.
      - force a restart every 3*natoms directions regardless.
      - guard p.g < 0; fall back to -g if the direction is not downhill.
    """

    def __init__(self, restart=None):
        self.restart = restart
        self.reset()

    def reset(self):
        self.g_old = self.p_old = None
        self.n = 0

    def direction(self, g, x):
        raise NotImplementedError('CG.direction')


# ---------------------------------------------------------------- line search

def line_search(grad, coords, p, E0, g0, smax, c1=1e-4, rho=0.5, nmax=12):
    # backtracking Armijo along p; energies are ~90x cheaper than a gradient
    # here, so a cubic/Brent search would be nearly free -- swap this out.
    slope = float(g0.ravel() @ p.ravel())
    if slope >= 0:
        return None, None, None
    s = smax
    for _ in range(nmax):
        E = grad.E(coords + s*p)
        if E <= E0 + c1*s*slope:
            return s, E, coords + s*p
        s *= rho
    return None, None, None


# ---------------------------------------------------------------------- driver

def optimise(geom='H5C10.xyz', out=None, traj=None, method=None, alpha=0.05,
             scale=1.54, amp=None, seed=0, ftol=1e-3, maxstep=0.2, nmax=200,
             typor0=True):

    stem = Path(geom).stem
    out = Path(out or f'{stem}_opt.xyz')
    traj = Path(traj or f'{stem}_traj.xyz')
    method = SD() if method is None else method

    mol = H(geom=geom, typor0=typor0)
    grad = Gradients_opt(mol, alpha=alpha)

    coords = mol.coords[0].copy()
    if scale:
        coords = prescale(mol.atoms, coords, scale)
    if amp:
        coords = kick(mol.atoms, coords, amp, seed)
    mol.coords = coords[None]

    if getattr(method, 'restart', None) is None and hasattr(method, 'restart'):
        method.restart = 3*len(mol.atoms)
    method.reset()

    E = grad.E(coords)
    g = grad.num()                                   # dE/dR, NOT the force
    write_xyz(traj, mol.atoms, coords, f'step 0  E = {E:.8f} Ha')

    print(f'{"step":>4} {"E [Ha]":>16} {"dE [Ha]":>12} '
          f'{"|g|max [Ha/a0]":>15} {"step [a0]":>10}', flush=True)
    print(f'{0:>4} {E:>16.8f} {"":>12} {np.abs(g).max():>15.6f} {"":>10}', flush=True)

    it = 0
    while it < nmax:
        gmax = np.abs(g).max()
        if gmax < ftol:
            break

        p = method.direction(g, coords)
        smax = maxstep/np.abs(p).max()               # trust radius on displacement
        s, Enew, new = line_search(grad, coords, p, E, g, smax)

        if s is None:                                # no downhill point: restart
            if isinstance(method, SD) or method.g_old is None:
                print('line search failed from a steepest-descent direction, stopping')
                break
            print('line search failed -> restart')
            method.reset()
            continue

        it += 1
        update_geometry(mol, coords, p, s, out,
                        f'step {it}  E = {Enew:.8f} Ha')
        print(f'{it:>4} {Enew:>16.8f} {Enew-E:>12.2e} '
              f'{gmax:>15.6f} {s*np.abs(p).max():>10.4f}', flush=True)

        coords, E = new, Enew
        g = grad.num()
        write_xyz(traj, mol.atoms, coords,
                  f'step {it}  E = {E:.8f} Ha  |g|max = {np.abs(g).max():.6f}',
                  mode='a')

    gmax = np.abs(g).max()
    write_xyz(out, mol.atoms, coords,
              f'{stem} optimised  E = {E:.8f} Ha  |g|max = {gmax:.6f} Ha/a0')
    print(f'\nconverged: {gmax < ftol}   steps: {it}   '
          f'E = {E:.8f} Ha   |g|max = {gmax:.6f} Ha/a0')
    print(f'geometry -> {out}\ntrajectory -> {traj}')
    return mol, coords, E, g


if __name__ == '__main__':
    optimise(method=SD())
