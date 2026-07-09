#!/usr/bin/env python3
"""Initialize DFTB atomic data by running the DFT/allplotter.py solver per atom.

For every atom in ATOMS two solves are run:

  * confined pseudo-atom  -> ATOMS_BS   (basis WFs + confined eigenvalues)
                          -> ATOMS_POT  (converged Veff/Vconf)   [off-diagonal]
  * free neutral atom     -> eig_neutral (physical on-site levels) [diagonal]

--VO N adds N virtual shells. It is passed to BOTH solves on purpose: the DFTB
Hamiltonian reads the on-site level eigN[atom][n][l] for every basis orbital,
so the confined basis (ATOMS_BS) and the free-atom diagonal (eig_neutral) must
carry the same (n,l) set or the indexing desyncs. Free-atom virtuals only bind
with the LB94 -1/r tail, so --VO turns LB94 on for the free solve unless
--no-lb94 is given. The confined solve stays plain LDA (off-diagonal recipe).
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# atom label -> atomic number Z. Edit to change the system (default: He dimer).
ATOMS = {
    "C": 6,
    "O": 8,
}

ROOT = Path(__file__).resolve().parent
PLOTTER = ROOT.parent / "DFT" / "allplotter.py"
WFDIR = ROOT / "ATOMS_BS"
POTDIR = ROOT / "ATOMS_POT"
EIGDIR = ROOT / "eig_neutral"


def run(extra):
    cmd = [sys.executable, str(PLOTTER), *extra]
    print("  $", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--VO", type=int, default=None, metavar="N",
                    help="add N virtual shells to both solves (DFTB polarization basis)")
    ap.add_argument("--lb94", dest="lb94", action="store_true", default=None,
                    help="force LB94 on the free-atom (diagonal) solve")
    ap.add_argument("--no-lb94", dest="lb94", action="store_false",
                    help="disable LB94 even when --VO is set")
    args = ap.parse_args()

    # LB94 defaults on for the free solve whenever VO is requested (virtuals need
    # the -1/r tail to bind); off otherwise. --lb94/--no-lb94 override explicitly.
    lb94 = args.lb94 if args.lb94 is not None else (args.VO is not None)

    if not PLOTTER.exists():
        sys.exit(f"solver not found: {PLOTTER}")

    os.environ["MPLBACKEND"] = "Agg"   # headless: plt.show() must not block

    for d in (WFDIR, POTDIR, EIGDIR):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir()

    vo = ["--VO", str(args.VO)] if args.VO is not None else []
    print(f"VO={args.VO}  LB94(free)={lb94}  atoms={list(ATOMS)}")

    for atom, Z in ATOMS.items():
        print(f"[{atom}] Z={Z}")
        # Confined pseudo-atom: basis shapes + confined eigenvalues + Veff/Vconf.
        run([str(Z), "--pseudoatom", "--exp-grid", *vo,
             "--save", str(WFDIR / atom),
             "--save_V", str(POTDIR / atom)])
        # Free neutral atom: physical (negative) on-site levels for the diagonal.
        free = [str(Z), "--exp-grid", *vo, "--save", str(EIGDIR / atom)]
        if lb94:
            free.append("--LB94")
        run(free)

    print("Done.")


if __name__ == "__main__":
    main()
