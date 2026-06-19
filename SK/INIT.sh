#!/bin/bash
set -e

# Run matplotlib headless so plt.show() doesn't block the loop with a window.
export MPLBACKEND=Agg

declare -A list
list=( [carbon]=6 [hydrogen]=1 [nytogen]=7 [boron]=5 [oxygen]=8 [fluorine]=9 [Litium]=3 [alluminium]=13)

ATOMSDIR='ATOMS_BS'
ROOT="$(cd "$(dirname "$0")" && pwd)"
PLOTTER="$ROOT/../DFT/allplotter.py"

rm -rf "$ROOT/$ATOMSDIR"
mkdir "$ROOT/$ATOMSDIR"

for atom in "${!list[@]}"; do
	number="${list[$atom]}"
	( cd "$ROOT/$ATOMSDIR" && "$PLOTTER" "$number" --pseudoatom --exp-grid --save "$atom" )
done




