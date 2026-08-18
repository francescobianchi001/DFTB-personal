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
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# atom label -> atomic number Z. Edit to change the system (default: He dimer).
ATOMS = {
    "C": 6,
    "H": 1,
}

ROOT = Path(__file__).resolve().parent
PLOTTER = ROOT.parent / "DFT" / "allplotter.py"
WFDIR = ROOT / "ATOMS_BS"
POTDIR = ROOT / "ATOMS_POT"
EIGDIR = ROOT / "eig_neutral"
# Records WHICH settings the stored atoms were solved with, so a later run can
# tell "add a new element" (same level of theory -> keep the rest) apart from
# "the method changed" (-> everything on disk is stale and must be redone).
MANIFEST = ROOT / "atoms_provenance.json"


def resolve_lb94(lb94, vo):
    """LB94 defaults on for the free solve whenever VO is requested (virtuals
    need the -1/r tail to bind); off otherwise. Explicit flags override.
    prepare_atoms in Hamiltonian.py duplicates this rule -- keep them in step."""
    return bool(lb94) if lb94 is not None else (vo is not None)


def read_manifest():
    if not MANIFEST.exists():
        return None
    try:
        return json.loads(MANIFEST.read_text())
    except (ValueError, OSError):
        return None


def run(extra):
    cmd = [sys.executable, str(PLOTTER), *extra]
    print("  $", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--VO", type=int, default=None, metavar="N",
                    help="add N virtual shells to both solves (DFTB polarization basis)")
    ap.add_argument("--r0", dest="r0", type=float, default=None, metavar="R0",
                    help="override the valence confinement radius (bohr); default 2*r_cov "
                         "(tunes the DFTB off-diagonal / bonding)")
    ap.add_argument("--r0-VO", dest="r0_VO", type=float, default=None, metavar="R0",
                    help="weaker confinement radius (bohr) for the --VO virtual shells; "
                         "triggers the split-confinement solve (writes vo_shells/Vconf_VO)")
    ap.add_argument("--lb94", dest="lb94", action="store_true", default=None,
                    help="force LB94 on the free-atom (diagonal) solve")
    ap.add_argument("--no-lb94", dest="lb94", action="store_false",
                    help="disable LB94 even when --VO is set")
    ap.add_argument("--fresh", action="store_true",
                    help="wipe ATOMS_BS/ATOMS_POT/eig_neutral and redo every element "
                         "(implied when the requested settings differ from the stored "
                         "ones -- a change of method invalidates all of them)")
    ap.add_argument("--typor0", type=json.loads, default=None, metavar="JSON",
                    help="per-element confinement radii (bohr) as JSON, e.g. "
                         "'{\"H\": 1.084, \"C\": 2.657}'; overrides --r0 per element")
    args = ap.parse_args()

    lb94 = resolve_lb94(args.lb94, args.VO)
    settings = {"VO": args.VO, "r0": args.r0, "r0_VO": args.r0_VO, "lb94": lb94}
    if args.typor0:
        settings["typor0"] = args.typor0

    if not PLOTTER.exists():
        sys.exit(f"solver not found: {PLOTTER}")

    os.environ["MPLBACKEND"] = "Agg"   # headless: plt.show() must not block

    prov = read_manifest()
    stale = prov is not None and prov.get("settings") != settings
    if stale:
        print(f"settings changed {prov.get('settings')} -> {settings}; "
              f"rebuilding every element")
    fresh = args.fresh or stale

    if fresh:
        for d in (WFDIR, POTDIR, EIGDIR):
            shutil.rmtree(d, ignore_errors=True)
    for d in (WFDIR, POTDIR, EIGDIR):
        d.mkdir(exist_ok=True)

    vo = ["--VO", str(args.VO)] if args.VO is not None else []
    r0vo = ["--r0-VO", str(args.r0_VO)] if args.r0_VO is not None else []
    typor0 = args.typor0 or {}
    todo = {a: Z for a, Z in ATOMS.items()
            if not (WFDIR / f"{a}.npz").exists()}
    keep = [a for a in ATOMS if a not in todo]
    print(f"VO={args.VO}  r0={typor0 or args.r0}  r0_VO={args.r0_VO}  LB94(free)={lb94}")
    print(f"  solve: {list(todo) or '(nothing)'}"
          + (f"   keep (already at these settings): {keep}" if keep else ""))

    for atom, Z in todo.items():
        print(f"[{atom}] Z={Z}")
        # Per-element radius wins over --r0; neither set keeps the solver default.
        r0_atom = typor0.get(atom, args.r0)
        r0 = ["--r0", str(r0_atom)] if r0_atom is not None else []
        # Confined pseudo-atom: basis shapes + confined eigenvalues + Veff/Vconf.
        run([str(Z), "--pseudoatom", "--exp-grid", *vo, *r0, *r0vo,
             "--save", str(WFDIR / atom),
             "--save_V", str(POTDIR / atom)])
        # Free neutral atom: physical (negative) on-site levels for the diagonal.
        free = [str(Z), "--exp-grid", *vo, "--save", str(EIGDIR / atom)]
        if lb94:
            free.append("--LB94")
        run(free)

    # Record the settings plus everything now on disk, so the next run can tell
    # "new element at the same level" from "method changed, all of it is stale".
    MANIFEST.write_text(json.dumps(
        {"settings": settings,
         "elements": sorted(p.stem for p in WFDIR.glob("*.npz"))}, indent=2))
    print("Done.")


if __name__ == "__main__":
    main()
