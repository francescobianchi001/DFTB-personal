#!/usr/bin/env python3
"""Decompose DFTB-vs-reference MO-energy error.

  on-site rigid shift  -> level placement (Gap B: LB94 / confinement / XC)
  bonding/shape residual (after removing the shift) -> relative spacings (Gap A: r0, integrals)
  deep-vs-frontier split of the residual -> a deep-heavy residual flags the -Z/r
  core-singularity (integration) error, worst for the s-derived deep levels.
"""
import numpy as np
import re

HARTREE = 27.211386


def parse_dialect(path):
    E = [float(e) for _, e in
         re.findall(r'MO:\s*\d+\s+([\d.]+)\s+(-?\d+\.\d+)', open(path).read())]
    return np.sort(np.array(E, float))


def decompose(mine, ref, nocc, deg_tol=1e-3):
    mine = np.sort(np.asarray(mine, float))
    ref = np.sort(np.asarray(ref, float))
    d = mine[:nocc] - ref[:nocc]
    shift = d.mean()
    resid = d - shift
    k = max(1, nocc // 3)
    deg = lambda E: int(np.sum(np.diff(E[:nocc]) < deg_tol))
    return dict(
        nocc=nocc,
        shift=shift * HARTREE,
        resid_rms=np.sqrt((resid ** 2).mean()) * HARTREE,
        resid_max=np.abs(resid).max() * HARTREE,
        deep=np.abs(resid[:k]).mean() * HARTREE,
        frontier=np.abs(resid[-k:]).mean() * HARTREE,
        homo_err=(mine[nocc - 1] - ref[nocc - 1]) * HARTREE,
        gap_err=(((mine[nocc] - mine[nocc - 1]) - (ref[nocc] - ref[nocc - 1])) * HARTREE
                 if len(mine) > nocc and len(ref) > nocc else np.nan),
        deg_mine=deg(mine),
        deg_ref=deg(ref),
    )


def report(mine, ref, nocc, name="", deg_tol=1e-3):
    r = decompose(mine, ref, nocc, deg_tol)
    src = "integration/core-heavy" if r["deep"] > 1.5 * r["frontier"] else "spread = bonding"
    print(f"--- error decomposition {name} (vs reference, eV; {nocc} occ) ---")
    print(f"  on-site rigid shift     : {r['shift']:+.2f}")
    print(f"  bonding/shape residual  : rms {r['resid_rms']:.2f}   max {r['resid_max']:.2f}")
    print(f"    deep / frontier       : {r['deep']:.2f} / {r['frontier']:.2f}   ({src})")
    print(f"  HOMO error              : {r['homo_err']:+.2f}")
    print(f"  HOMO-LUMO gap error     : {r['gap_err']:+.2f}")
    print(f"  degeneracies (<{deg_tol:g}) : mine {r['deg_mine']}  ref {r['deg_ref']}"
          f"   {'match' if r['deg_mine'] == r['deg_ref'] else 'DIFFER'}")
    return r
