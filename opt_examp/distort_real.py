#! /usr/bin/python3
"""Realistic starting geometries: perturb the carbon skeleton, then REBUILD every
hydrogen on it with ideal C-H. Bond lengths/angles/torsions come out wrong (which is
the point) but nothing is chemically absurd -- no stray or overlapping H.

Run from the repository root:  python3 opt_examp/distort_real.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from read_xyz import get_coords, BOHR_TO_ANGS
from optimisation import write_xyz

ROOT = Path(__file__).resolve().parent
IDEAL = ROOT/'opt1'
CH_SP3, CH_SP2, CH_SP = 1.090, 1.087, 1.061
CC_BOND = 1.95          # A, C-C bonded cutoff


def unit(v):
    return np.asarray(v, float)/np.linalg.norm(v)


def rot(axis, deg):
    a = unit(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    t = np.radians(deg)
    return np.eye(3)+np.sin(t)*K+(1-np.cos(t))*K@K


def build_H(c, i, nbC, nH):
    """Ideal hydrogens on carbon i given its bonded carbons nbC."""
    ci = c[i]
    if nH == 1 and len(nbC) == 1:                      # ethyne: along the C-C axis
        return [ci + CH_SP*unit(ci-c[nbC[0]])]
    if nH == 1 and len(nbC) == 2:                      # aromatic CH, in-plane, outward
        u = unit(unit(c[nbC[0]]-ci) + unit(c[nbC[1]]-ci))
        return [ci - CH_SP2*u]
    if nH == 2:                                        # CH2
        u = unit(unit(c[nbC[0]]-ci) + unit(c[nbC[1]]-ci))
        w = unit(np.cross(c[nbC[0]]-ci, c[nbC[1]]-ci))
        h = np.radians(107.0)
        return [ci + CH_SP3*(-u*np.cos(h/2) + w*np.sin(h/2)),
                ci + CH_SP3*(-u*np.cos(h/2) - w*np.sin(h/2))]
    if nH == 3:                                        # CH3
        n = unit(c[nbC[0]]-ci)
        ref = np.array([0., 0., 1.]) if abs(n[2]) < 0.9 else np.array([1., 0., 0.])
        x = unit(np.cross(n, ref))
        y = np.cross(n, x)
        a = np.radians(109.47)
        return [ci + CH_SP3*(n*np.cos(a) + (x*np.cos(t)+y*np.sin(t))*np.sin(a))
                for t in np.radians([0, 120, 240])]
    raise ValueError(f'atom {i}: nH={nH}, {len(nbC)} C neighbours')


def distort(name, amp=0.13, stretch=0.0, twist=None, seed=0):
    Z, c = get_coords(str(IDEAL/f'{name}.xyz'))
    c = c[0].copy()*BOHR_TO_ANGS                       # work in Angstrom
    rng = np.random.default_rng(seed)
    C = np.where(Z == 6)[0]
    Hh = np.where(Z == 1)[0]

    d0 = np.linalg.norm(c[:, None]-c[None, :], axis=-1)
    parent = C[np.argmin(d0[np.ix_(Hh, C)], axis=1)]
    nH = {int(a): int((parent == a).sum()) for a in C}
    bonded = {int(a): [int(b) for b in C if b != a and d0[a, b] < CC_BOND] for a in C}

    if twist is not None:                              # rotate the tail about C2-C3
        R = rot(c[C[2]]-c[C[1]], twist)
        c[C[3]] = c[C[2]] + R@(c[C[3]]-c[C[2]])

    if stretch:                                        # uniform expansion of the skeleton
        ctr = c[C].mean(0)
        c[C] = ctr + (1.0+stretch)*(c[C]-ctr)
    c[C] += rng.normal(0, amp, c[C].shape)             # perturb the skeleton only

    for a in C:                                        # rebuild every H from scratch
        hs = build_H(c, int(a), bonded[int(a)], nH[int(a)])
        for h, pos in zip(Hh[parent == a], hs):
            c[h] = pos

    d = np.linalg.norm(c[:, None]-c[None, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    dh = d[np.ix_(C, C)].copy()
    np.fill_diagonal(dh, np.inf)
    ch = d[np.ix_(Hh, C)].min(axis=1)
    hh = d[np.ix_(Hh, Hh)]
    print('%-12s minpair %.3f  C-C nn %.3f-%.3f  C-H %.3f-%.3f  min H-H %.3f' % (
        name, d.min(), dh.min(axis=1).min(), dh.min(axis=1).max(),
        ch.min(), ch.max(), hh.min()))
    write_xyz(str(ROOT/f'{name}.xyz'), Z, c/BOHR_TO_ANGS,
              f'{name}, realistic distortion: {amp} A skeleton noise'
              + (f' + {stretch:+.0%} stretch' if stretch else '')
              + (f' + {twist} deg twist' if twist else '') + f' (seed {seed})')


if __name__ == '__main__':
    distort('ethyne',      amp=0.05, stretch=0.09, seed=1)
    distort('benzene',     amp=0.13, stretch=0.05, seed=2)
    # butane starts with one C-C at 2.157 A -- past VMD's covalent cutoff, so load
    # the movie through show.tcl, which takes bonds from the converged last frame
    distort('butane',      amp=0.13, stretch=0.05, twist=45, seed=3)
    distort('cyclohexane', amp=0.13, stretch=0.05, seed=4)
