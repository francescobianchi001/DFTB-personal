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

# Both searches return (s, E, new_coords, g_new); g_new is None when the search
# never evaluated a gradient, and the driver then has to compute one itself.

def line_search(grad, coords, p, E0, g0, smax, c1=1e-4, rho=0.5, nmax=12):
    # backtracking Armijo along p. An energy is only ~3.3x cheaper than an
    # analytic gradient (0.57 vs 1.9 s, butane), so probes are NOT free and
    # `wolfe` below -- which pays one fused E+gradient per probe -- wins.
    slope = float(g0.ravel() @ p.ravel())
    if slope >= 0:
        return None, None, None, None
    s = smax
    for _ in range(nmax):
        E = grad.E(coords + s*p)
        if E <= E0 + c1*s*slope:
            return s, E, coords + s*p, None
        s *= rho
    return None, None, None, None


def wolfe(fg, E0, d0, smax, c1=1e-4, c2=0.1, nmax=12):
    """Strong Wolfe line search (Nocedal-Wright alg. 3.5/3.6).

    fg(s) -> (phi, dphi, g) with phi = E(x+s p), dphi = g(x+s p).p, g the full
    gradient. One fused E_and_grad per trial, and the gradient at the accepted
    step is returned so the caller does not pay a separate per-step gradient.
    c2 < 0.5 is what keeps a Polak-Ribiere direction descending.
    """
    def interp(a, fa, da, b, fb, db):
        # cubic through (a,fa,da),(b,fb,db); bisect if it lands outside (a,b)
        t = da + db - 3*(fa-fb)/(a-b)
        r = t*t - da*db
        lo, hi = (a, b) if a < b else (b, a)
        if r <= 0:
            return 0.5*(a+b)
        q = np.sqrt(r)*(1 if b > a else -1)
        den = db - da + 2*q
        if den == 0:
            return 0.5*(a+b)
        s = b - (b-a)*(db+q-t)/den
        pad = 1e-3*(hi-lo)
        return s if lo+pad < s < hi-pad else 0.5*(a+b)

    def zoom(lo, flo, dlo, hi, fhi, dhi, budget):
        for _ in range(budget):
            s = interp(lo, flo, dlo, hi, fhi, dhi)
            f, d, g = fg(s)
            if f > E0 + c1*s*d0 or f >= flo:
                hi, fhi, dhi = s, f, d
            else:
                if abs(d) <= -c2*d0:               # strong Wolfe satisfied
                    return s, f, g
                if d*(hi-lo) >= 0:
                    hi, fhi, dhi = lo, flo, dlo
                lo, flo, dlo = s, f, d
        return None, None, None

    if d0 >= 0:                                    # not a descent direction
        return None, None, None
    s_prev, f_prev, d_prev = 0.0, E0, d0
    s = smax
    for i in range(nmax):
        f, d, g = fg(s)
        if f > E0 + c1*s*d0 or (i and f >= f_prev):
            return zoom(s_prev, f_prev, d_prev, s, f, d, nmax-i)
        if abs(d) <= -c2*d0:
            return s, f, g
        if d >= 0:                                 # passed the minimum
            return zoom(s, f, d, s_prev, f_prev, d_prev, nmax-i)
        s_prev, f_prev, d_prev = s, f, d
        s = min(2*s, 64*smax)
    return None, None, None


def wolfe_search(grad, coords, p, E0, g0, smax, **kw):
    # driver-facing wrapper: same 4-tuple convention as line_search
    def fg(s):
        Es, gs = grad.E_and_grad(coords + s*p)
        return Es, float(gs.ravel() @ p.ravel()), gs

    s, E, g = wolfe(fg, E0, float(g0.ravel() @ p.ravel()), smax, **kw)
    return (None, None, None, None) if s is None else (s, E, coords + s*p, g)


# ---------------------------------------------------------------------- driver

def optimise(geom='H5C10.xyz', out=None, traj=None, method=None, alpha=0.05,
             scale=1.54, amp=None, seed=0, ftol=1e-3, maxstep=0.2, nmax=200,
             typor0=True, search='wolfe'):

    search = {'wolfe': wolfe_search, 'armijo': line_search}[search]
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

    E, g = grad.E_and_grad(coords)                   # dE/dR, NOT the force
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
        s, Enew, new, gnew = search(grad, coords, p, E, g, smax)

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
        g = gnew if gnew is not None else grad.grad(coords)   # wolfe already has it
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
