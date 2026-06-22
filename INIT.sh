#!/bin/bash
set -e

# Run matplotlib headless so plt.show() doesn't block the loop with a window.
export MPLBACKEND=Agg

declare -A list
list=( [carbon]=6 [hydrogen]=1)

# Output directories: wavefunctions (also carry eigenvalues) and potentials.
WFDIR='ATOMS_BS'
POTDIR='ATOMS_POT'
ROOT="$(cd "$(dirname "$0")" && pwd)"
PLOTTER="$ROOT/../DFT/allplotter.py"

for d in "$WFDIR" "$POTDIR"; do
	rm -rf "$ROOT/$d"
	mkdir "$ROOT/$d"
done

for atom in "${!list[@]}"; do
	number="${list[$atom]}"
	# One confined pseudo-atom SCF solve; write WF (incl. eigenvalues) and
	# potentials into their respective directories.
	"$PLOTTER" "$number" --pseudoatom --exp-grid \
		--save   "$ROOT/$WFDIR/$atom" \
		--save_V "$ROOT/$POTDIR/$atom"
done




