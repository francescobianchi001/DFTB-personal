# Vpot — repulsive potential V_rep

Everything pulled from hotbit (github.com/pekkosk/hotbit, shallow clone of master,
2026-08-28) that the V_rep fit needs. Nothing here is ours yet.

## data/ — fitting targets

The reference (DFT) energy curves the CH repulsion was fitted to, from
`examples/CH_parametrization/`. Same four systems the paper lists for C–H
(r_cut = 3.40 a0, lambda = 35, sigma_i = 1): CH-, ethyne, methane, benzene.

`*.traj` are the originals (ASE ULM v2, needs ase to read). Each was converted
here with ase 3.29 into two dependency-free copies:

- `*.xyz`  multi-frame plain XYZ, energy on the comment line
- `*.npz`  symbols, Z, R (nframes,nat,3), E (nframes,), F (nframes,nat,3)

Units are the ASE ones as stored: **Angstrom, eV, eV/Angstrom** — not converted.

| system  | frames | atoms |
|---------|--------|-------|
| CH-     | 14     | 2     |
| ethyne  |  7     | 4     |
| methane | 20     | 5     |
| benzene |  6     | 12    |
| H       |  2     | 1     |

Forces are present in every frame. `H.traj` is the isolated atom (unused by CH.py).

## ref/ — hotbit's implementation, for reference

- `fitting.py`   `RepulsiveFitting`: append_dimer / append_energy_curve /
                 append_force_curve, the smoothing-spline fit, write_par
- `repulsion.py` how the fitted V_rep is evaluated in the calculator
- `CH.py`        the driver; phase 3 is the repulsion fit

## par/ — targets to reproduce

- `C_H_no_repulsion.par` / `C_H_repulsion.par` — the example's before/after.
  Fit used: r_cut = 1.6000 Ang, s = 100, k = 3; systems CH dimer | CH- | ethyne |
  methane | benzene. The `repulsion=` block is (r [Bohr], V_rep [Hartree]).
- `CH_repulsion.pdf` — its plot.
- `C.elm`, `H.elm` — on-site energies, U, FWHM used by that fit.
- `EurPhysJD_67_38_2013/` — the published set we are reproducing.
  C_H fit there: r_cut = 1.5500 Ang, s = 40, k = 3.
