#!/usr/bin/python3
"""H(R) and S(R) for a diatomic on a grid of bond lengths.

Rebuilds the SAME H object at every R (atom data loaded once) and stores the
matrices in an npz plus a readable text dump.
"""
import argparse
import numpy as np
from Hamiltonian import H

HA = 27.21138
R_DEFAULT = np.linspace(1.0, 12.0, 50)


def read_R(spec):
    """R grid from a comma/space separated list, or from a file of numbers."""
    from pathlib import Path
    text = Path(spec).read_text() if Path(spec).is_file() else spec
    return np.array([float(x) for x in text.replace(',', ' ').split()])


def ao_labels(mol):
    sh = 'spdfg'
    names = [mol.names[mol.Z2elem[Z]] for Z in mol.atoms]
    return [f"{names[ao.atom]}{ao.atom + 1} {ao.n}{sh[ao.l]}{ao.m:+d}" for ao in mol.basis]


def scan(geom, R, **kw):
    mol = H(geom=geom, **kw)
    if len(mol.atoms) != 2:
        raise ValueError(f"{geom}: expected 2 atoms, got {len(mol.atoms)}")
    axis = mol.coords[0][1] - mol.coords[0][0]
    axis /= np.linalg.norm(axis)
    R0 = mol.coords[0][0].copy()

    Hs, Ss, Eb, Ev, labels = [], [], [], [], None
    for d in R:
        mol.coords[0][1] = R0 + d * axis          # move atom 2 along the bond
        mol.H_matrix()
        if labels is None:
            labels = ao_labels(mol)
        try:                                      # R=0 duplicates the basis -> S singular
            E, C = mol.diag()
            eband = mol.Eband
        except np.linalg.LinAlgError as e:
            print(f"  R = {d:7.4f} bohr   diag failed ({e}) -- H, S still stored",
                  flush=True)
            E, eband = np.full(mol.N, np.nan), np.nan
        else:
            print(f"  R = {d:7.4f} bohr   E_band = {eband:12.6f} Ha "
                  f"= {eband*HA:10.3f} eV   S_max_offdiag = "
                  f"{np.abs(mol.S - np.diag(np.diag(mol.S))).max():.5f}", flush=True)
        Hs.append(mol.H.copy()), Ss.append(mol.S.copy())
        Eb.append(eband), Ev.append(E.copy())
    return mol, np.asarray(R), np.asarray(Hs), np.asarray(Ss), \
           np.asarray(Eb), np.asarray(Ev), labels


def dump(path, R, Hs, Ss, labels):
    w = max(10, max(len(s) for s in labels) + 1)
    with open(path, 'w') as f:
        for k, d in enumerate(R):
            for name, M in (('H', Hs[k]), ('S', Ss[k])):
                f.write(f"\n{name}(R = {d:.6f} bohr)\n")
                f.write(' ' * (w + 2) + ''.join(f"{s:>{w}}" for s in labels) + '\n')
                for i, s in enumerate(labels):
                    f.write(f"{s:>{w}}  " + ''.join(f"{M[i, j]:>{w}.6f}"
                                                    for j in range(M.shape[1])) + '\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('geom', nargs='?', default='DIALECT/CH.xyz')
    ap.add_argument('--out', default='CH_scan_HS')
    ap.add_argument('--R', default=None,
                    help='bond lengths in bohr: comma/space separated list, or a file '
                         'of numbers (default: 50 points from 1 to 12)')
    ap.add_argument('--full', action='store_true', help='full basis (default: minimal valence)')
    ap.add_argument('--diffradi', nargs='?', const=True, default=None, dest='typor0',
                    metavar='FILE', help='per-element confinement radii (default radi.txt)')
    ap.add_argument('--VO', type=int, default=None, dest='vo')
    ap.add_argument('--r0', type=float, default=None)
    ap.add_argument('--r0-VO', type=float, default=None, dest='r0_vo')
    ap.add_argument('--lb94', action='store_true', default=None)
    ap.add_argument('--no-lb94', action='store_false', dest='lb94')
    args = ap.parse_args()

    mol, R, Hs, Ss, Eb, Ev, labels = scan(
        args.geom, read_R(args.R) if args.R else R_DEFAULT, frozen_core=not args.full, typor0=args.typor0,
        vo=args.vo, r0=args.r0, r0_vo=args.r0_vo, lb94=args.lb94)

    np.savez(args.out + '.npz', R=R, H=Hs, S=Ss, Eband=Eb, eigenvalues=Ev,
             labels=np.array(labels), occupations=mol.f, nelec=mol.nelec)
    dump(args.out + '.txt', R, Hs, Ss, labels)
    print(f"\nwrote {args.out}.npz  (H, S shape {Hs.shape}) and {args.out}.txt")
