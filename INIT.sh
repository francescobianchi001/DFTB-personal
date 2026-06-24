#!/bin/bash
set -e

# Run matplotlib headless so plt.show() doesn't block the loop with a window.
export MPLBACKEND=Agg

declare -A list
list=( [carbon]=6 [hydrogen]=1)

# Output directories: confined WFs (also carry confined eigenvalues), confined
# potentials, and free-atom (neutral) eigenvalues for the H on-site diagonal.
WFDIR='ATOMS_BS'
POTDIR='ATOMS_POT'
EIGDIR='eig_neutral'
ROOT="$(cd "$(dirname "$0")" && pwd)"
PLOTTER="$ROOT/../DFT/allplotter.py"

for d in "$WFDIR" "$POTDIR" "$EIGDIR"; do
	rm -rf "$ROOT/$d"
	mkdir "$ROOT/$d"
done

for atom in "${!list[@]}"; do
	number="${list[$atom]}"
	# Confined pseudo-atom SCF solve: WF (incl. confined eigenvalues) -> ATOMS_BS,
	# converged Veff/Vconf -> ATOMS_POT. The compression shapes the DFTB basis.
	"$PLOTTER" "$number" --pseudoatom --exp-grid \
		--save   "$ROOT/$WFDIR/$atom" \
		--save_V "$ROOT/$POTDIR/$atom"
	# Free (neutral, unconfined) atom: same solver without --pseudoatom. Its
	# eigenvalues are the physical (negative) on-site levels -> eig_neutral.
	"$PLOTTER" "$number" --exp-grid \
		--save   "$ROOT/$EIGDIR/$atom"
done




