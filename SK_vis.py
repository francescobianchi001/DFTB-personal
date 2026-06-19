#!/usr/bin/python3
"""Plot Slater-Koster two-center overlaps S(R).

One figure per (i, j) atom couple; each figure shows every orbital channel as a
function of internuclear distance R, labelled in SK notation, e.g. "2p-2p pi".
Saves one PNG per atom pair.
"""

import numpy as np
import matplotlib.pyplot as plt
from SlaterKonster import slaterkonster as SK

GREEK = {0: 'σ', 1: 'π', 2: 'δ'}      # bond type o -> Slater-Koster symbol
SPDF = 'spdf'                          # angular momentum l -> letter

init = SK()

# S_d : list of dicts, one per distance, each  {(i, j): {(nA,lA,nB,lB,o): S}}
# d   : the internuclear distances
S_d, d = init.distance_gradient()
d = np.asarray(d)


def elem(a):
    """Cheap element label from the basis content (order-independent)."""
    bs = init.basisets[a]
    if any(len(shell) > 1 for shell in bs):     # has a shell with an l=1 (p) entry
        return 'C'
    if sum(len(s) for s in bs) == 1:            # a single 1s function
        return 'H'
    return f'atom{a}'


def orb(n, l):
    """Orbital name, e.g. (n=1,l=0)->'2s'.  Principal number = shell index + 1."""
    return f"{n + 1}{SPDF[l]}"


def channel_label(key):
    nA, lA, nB, lB, o = key
    return f"{orb(nA, lA)}-{orb(nB, lB)} {GREEK[o]}"


atom_pairs = list(S_d[0].keys())

for (i, j) in atom_pairs:
    channels = list(S_d[0][(i, j)].keys())

    # collect every channel curve: shape (n_channels, n_distances)
    curves = np.array([[S_d[k][(i, j)][key] for k in range(len(d))]
                       for key in channels])

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for key, curve in zip(channels, curves):
        ax.plot(d, curve, label=channel_label(key))

    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('internuclear distance  R  (Bohr)')
    ax.set_ylabel('overlap  S(R)')
    ax.set_title(f'Slater-Koster overlaps:  {elem(i)} - {elem(j)}')
    ax.legend(fontsize=8, ncol=2)

    # auto-zoom to where there is still appreciable overlap
    sig = np.where(np.any(np.abs(curves) > 1e-3, axis=0))[0]
    ax.set_xlim(0, d[sig[-1]] * 1.15 if len(sig) else d[-1])

    fig.tight_layout()

plt.show()      # display every figure on screen
