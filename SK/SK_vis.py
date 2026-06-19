#!/usr/bin/python3
"""Slater-Koster overlaps S(R): plot per couple and/or dump as readable tables.

    ./SK_vis.py                      # tables -> ./SK_tables + plot every couple
    ./SK_vis.py --pairs C-H C-C      # only plot/tabulate those couples
    ./SK_vis.py --no-tables          # just the plots
"""

import argparse
from pathlib import Path

import numpy as np

from SlaterKonster import slaterkonster as SK

GREEK = {0: 'sigma', 1: 'pi', 2: 'delta'}   # bond type o, ascii for files
SYMBOL = {0: 'σ', 1: 'π', 2: 'δ'}           # ... unicode for plots
SPDF = 'spdf'

PERIODIC = {
    1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O',
    9: 'F', 10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P',
    16: 'S', 17: 'Cl', 18: 'Ar',
}


def orb(n, l):
    return f"{n + 1}{SPDF[l]}"


def channel_label(key, greek=GREEK):
    nA, lA, nB, lB, o = key
    return f"{orb(nA, lA)}-{orb(nB, lB)} {greek[o]}"


def elem(init, a):
    return PERIODIC.get(init.Znum[a], init.names[a])


def write_table(path, d, channels, curves, init, i, j):
    """One couple's S(R): R in column 1, one channel per column. loadtxt-friendly."""
    labels = [channel_label(k) for k in channels]
    widths = [max(len(lab), 14) for lab in labels]

    header_cols = f"{'R[Bohr]':>14}" + "".join(
        f"{lab:>{w + 2}}" for lab, w in zip(labels, widths))

    lines = [
        f"# Slater-Koster two-center overlaps  S(R)",
        f"# couple: {elem(init, i)} (Z={init.Znum[i]})  -  {elem(init, j)} (Z={init.Znum[j]})"
        f"   [atom indices {i}, {j}]",
        f"# R in Bohr; {len(channels)} channel column(s); rows are distances",
        f"#" + header_cols[1:],
    ]
    for k, R in enumerate(d):
        row = f"{R:>14.6f}" + "".join(
            f"{curves[c, k]:>{w + 2}.6e}" for c, w in enumerate(widths))
        lines.append(row)

    path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(
        description="Plot and/or tabulate Slater-Koster overlaps S(R).")
    ap.add_argument('-o', '--outdir', default='SK_tables', type=Path,
                    help="directory for the text tables (default: SK_tables)")
    ap.add_argument('--no-tables', dest='tables', action='store_false',
                    help="skip writing the readable S(R) tables")
    ap.add_argument('--no-plot', dest='plot', action='store_false',
                    help="skip the on-screen figures")
    ap.add_argument('--pairs', nargs='+', metavar='A-B',
                    help="only handle these couples, e.g. C-H C-C (default: all)")
    args = ap.parse_args()

    init = SK()

    # S_d: list over distances of {(i, j): {(nA, lA, nB, lB, o): S}}
    S_d, d = init.distance_gradient()
    d = np.asarray(d)
    atom_pairs = list(S_d[0].keys())

    if args.pairs:
        want = {frozenset(p.upper().split('-')) for p in args.pairs}
        atom_pairs = [(i, j) for (i, j) in atom_pairs
                      if frozenset({elem(init, i).upper(), elem(init, j).upper()}) in want]
        if not atom_pairs:
            ap.error(f"no couples match {args.pairs}; "
                     f"available: {sorted({elem(init, a) for a in range(len(init.names))})}")

    if args.tables:
        args.outdir.mkdir(parents=True, exist_ok=True)

    plt = None
    if args.plot:
        import matplotlib.pyplot as plt

    for (i, j) in atom_pairs:
        channels = list(S_d[0][(i, j)].keys())
        curves = np.array([[S_d[k][(i, j)][key] for k in range(len(d))]
                           for key in channels])

        tag = f"{elem(init, i)}-{elem(init, j)}"

        if args.tables:
            write_table(args.outdir / f"S_{tag}_{i}-{j}.dat",
                        d, channels, curves, init, i, j)

        if args.plot:
            fig, ax = plt.subplots(figsize=(7.5, 5))
            for key, curve in zip(channels, curves):
                ax.plot(d, curve, label=channel_label(key, SYMBOL))

            ax.axhline(0, color='k', lw=0.5)
            ax.set_xlabel('internuclear distance  R  (Bohr)')
            ax.set_ylabel('overlap  S(R)')
            ax.set_title(f'Slater-Koster overlaps:  {elem(init, i)} - {elem(init, j)}')
            ax.legend(fontsize=8, ncol=2)

            sig = np.where(np.any(np.abs(curves) > 1e-3, axis=0))[0]
            ax.set_xlim(0, d[sig[-1]] * 1.15 if len(sig) else d[-1])
            fig.tight_layout()

    if args.tables:
        print(f"wrote {len(atom_pairs)} table(s) to {args.outdir}/")
    if args.plot:
        plt.show()


if __name__ == '__main__':
    main()
