#!/usr/bin/env python3
"""Visualise the molecular orbitals coming out of the DFTB Hamiltonian.

For each MO k the wavefunction is rebuilt in real space from the eigenvectors,
    psi_k(r) = sum_mu  C[mu, k] * R_{n l}(r_mu) * Y_{l m}(theta_mu, phi_mu),
each AO centred on its atom, and drawn as a signed amplitude in a 2D plane.
The radial part R = u/r comes from the basis; the angular part Y is taken
straight from Y_real.harmonics. Run directly to pop the full MO gallery.

The geometry (any number of atoms) is read from an xyz file -- pass it on the
command line, e.g.  ./MO.py N2.xyz.  All AOs are evaluated with their true 3D
cartesian offsets, so the molecule does not have to lie on any particular axis.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from Hamiltonian import H
from Y_real import Y_real

HA = 27.21138


class MOViz:
    def __init__(self, geom='geometry.xyz', frozen_core=True, **kw):
        self.mol = H(geom=geom, frozen_core=frozen_core, **kw)
        self.mol.H_matrix()
        self.E, self.C = self.mol.diag()
        self.basis = self.mol.basis
        self.geom = geom
        self.nocc = int(round(self.mol.nelec)) // 2
        # true 3D nuclear positions in bohr, (natoms, 3), first frame of the xyz.
        self.pos = np.asarray(self.mol.coords[0], dtype=float)
        self.symbols = [self.mol.names[self.mol.Z2elem[Z]] for Z in self.mol.atoms]
        self.label = '-'.join(self.symbols) if len(self.symbols) <= 6 else \
                     f"{len(self.symbols)} atoms"
        self._frame()

    def _frame(self):
        """Molecular frame: e_long = principal axis, plus two transverse axes.

        The transverse pair is taken from the global x/y/z axes least aligned
        with the long axis, so a diatomic on z keeps the familiar xz / yz slices
        (needed to see pi orbitals, which vanish in one of the two planes).
        """
        R = self.pos - self.pos.mean(axis=0)
        if len(R) < 2 or np.allclose(R, 0.0):
            self.e_long = np.array([0.0, 0.0, 1.0])
        else:
            _, _, Vt = np.linalg.svd(R, full_matrices=False)
            self.e_long = Vt[0]
        order = np.argsort(np.abs(self.e_long))          # least-aligned first
        self.e_trans = [np.eye(3)[order[0]], np.eye(3)[order[1]]]
        self.center = self.pos.mean(axis=0)
        self.span = float(np.ptp(R @ self.e_long)) / 2.0  # half-length along the axis

    def _psi_at(self, k, pts):
        """Evaluate MO k at cartesian points `pts` of shape (..., 3)."""
        C = self.C
        psi = np.zeros(pts.shape[:-1])
        for mu, ao in enumerate(self.basis):
            c = C[mu, k]
            if abs(c) < 1e-9:
                continue
            d = pts - self.pos[ao.atom]                  # offset from this AO's centre
            r = np.sqrt((d**2).sum(-1))
            rs = np.where(r < 1e-9, 1e-9, r)
            theta = np.arccos(np.clip(d[..., 2] / rs, -1.0, 1.0))
            phi = np.arctan2(d[..., 1], d[..., 0])
            R = np.interp(r.ravel(), ao.grid, ao.u / ao.grid,
                          right=0.0).reshape(r.shape)
            psi += c * R * Y_real.harmonics[(ao.l, ao.m)](theta, phi)
        return psi

    def eval_mo(self, k, plane='auto', npts=240, half=4.0):
        """Sample MO k on a 2D plane containing the molecular (long) axis."""
        if plane == 'auto':                         # pi_y vanishes in xz -> use yz
            mdom = self.basis[int(np.argmax(np.abs(self.C[:, k])))].m
            plane = 1 if mdom < 0 else 0
        et = self.e_trans[plane]
        s = np.linspace(-self.span - half, self.span + half, npts)   # along the axis
        t = np.linspace(-half, half, npts)                           # transverse
        Ss, Tt = np.meshgrid(s, t, indexing='ij')
        pts = (self.center
               + Ss[..., None] * self.e_long
               + Tt[..., None] * et)
        nuc = ((self.pos - self.center) @ self.e_long,
               (self.pos - self.center) @ et)
        return Ss, Tt, self._psi_at(k, pts), plane, nuc

    def plot(self, which='all', npts=240, half=4.0):
        ks = list(range(self.mol.N)) if which == 'all' else list(which)
        ncols = min(4, len(ks))
        nrows = (len(ks) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.4 * nrows),
                                 squeeze=False)
        for idx, k in enumerate(ks):
            ax = axes[idx // ncols][idx % ncols]
            Ss, Tt, psi, plane, nuc = self.eval_mo(k, npts=npts, half=half)
            vmax = np.max(np.abs(psi)) or 1.0
            lv = np.linspace(-vmax, vmax, 41)
            ax.contourf(Ss, Tt, psi, levels=lv, cmap='RdBu_r', extend='both')
            ax.contour(Ss, Tt, psi, levels=[0.0], colors='k', linewidths=0.4)
            ax.plot(nuc[0], nuc[1], 'ko', ms=4)              # nuclei
            tag = 'occ' if k < self.nocc else 'virt'
            ax.set_title(f"MO{k+1}   {self.E[k]*HA:.2f} eV  ({tag})\n"
                         f"[plane {plane}]", fontsize=9)
            ax.set_aspect('equal')
            ax.set_xticks([]); ax.set_yticks([])
        for idx in range(len(ks), nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)
        fig.suptitle(f"{self.label}  molecular orbitals  ({self.geom})", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        return fig

    def eval_mo_3d(self, k, npts=70, half=4.0):
        """Sample MO k on a full 3D (x,y,z) grid boxing the molecule."""
        lo, hi = self.pos.min(axis=0) - half, self.pos.max(axis=0) + half
        axes = [np.linspace(lo[i], hi[i], npts) for i in range(3)]
        X, Y, Z = np.meshgrid(*axes, indexing='ij')
        pts = np.stack([X, Y, Z], axis=-1)
        return axes, self._psi_at(k, pts)

    def plot3d(self, which='all', npts=70, half=4.0, iso_frac=0.2):
        from skimage import measure
        ks = list(range(self.mol.N)) if which == 'all' else list(which)
        ncols = min(4, len(ks))
        nrows = (len(ks) + ncols - 1) // ncols
        fig = plt.figure(figsize=(3.6 * ncols, 3.6 * nrows))
        for idx, k in enumerate(ks):
            (x, y, z), psi = self.eval_mo_3d(k, npts=npts, half=half)
            ax = fig.add_subplot(nrows, ncols, idx + 1, projection='3d')
            level = iso_frac * np.max(np.abs(psi))
            spacing = (x[1] - x[0], y[1] - y[0], z[1] - z[0])
            origin = np.array([x[0], y[0], z[0]])
            for sign, color in ((+1, '#d62728'), (-1, '#1f77b4')):   # +/- phase lobes
                field = sign * psi
                if field.max() <= level:
                    continue
                try:
                    verts, faces, _, _ = measure.marching_cubes(
                        field, level=level, spacing=spacing)
                    verts = verts + origin
                    ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2],
                                    color=color, alpha=0.55, linewidth=0,
                                    antialiased=True)
                except (ValueError, RuntimeError):
                    pass
            ax.scatter(self.pos[:, 0], self.pos[:, 1], self.pos[:, 2],
                       color='k', s=25)                              # nuclei
            tag = 'occ' if k < self.nocc else 'virt'
            ax.set_title(f"MO{k+1}  {self.E[k]*HA:.2f} eV ({tag})", fontsize=9)
            ax.set_axis_off()
            try:
                ax.set_box_aspect((1, 1, 1.3))
            except Exception:
                pass
        fig.suptitle(f"{self.label}  molecular orbitals  ({self.geom})"
                     f"  —  red/blue = +/- phase", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        return fig

    def print_levels(self):
        """Print the MO eigenvalues with occupation, HOMO/LUMO and gap."""
        print(f"\n{self.label}  MO levels  ({self.geom})")
        print(f"{'MO':>4} {'occ':>4} {'E (Ha)':>12} {'E (eV)':>10}")
        print('-' * 34)
        for k in range(self.mol.N):
            occ = 2 if k < self.nocc else 0
            mark = '  <- HOMO' if k == self.nocc - 1 else ('  <- LUMO' if k == self.nocc else '')
            print(f"{k+1:>4} {occ:>4} {self.E[k]:>12.5f} {self.E[k]*HA:>10.2f}{mark}")
        if self.nocc < self.mol.N:
            gap = (self.E[self.nocc] - self.E[self.nocc - 1]) * HA
            print(f"HOMO-LUMO gap = {gap:.2f} eV\n")
        else:
            print("(all orbitals occupied — no virtual/LUMO)\n")

    def print_charges(self):
        Dq = self.mol.Mulliken_charge()
        print(f"\n{self.label}  Mulliken charges  ({self.geom})")
        print(f"{'atom':>6} {'q_net':>10}")
        print('-' * 18)
        for a, name in enumerate(self.symbols):
            print(f"{name:>6} {Dq[a]:>+10.5f}")
        print(f"{'sum':>6} {sum(Dq):>+10.2e}\n")

    def plot3d_pyvista(self, which='all', npts=80, half=4.0, iso_frac=0.18):
        """Smooth, GPU (VTK) isosurfaces — one subplot per MO, linked cameras."""
        import pyvista as pv
        ks = list(range(self.mol.N)) if which == 'all' else list(which)
        ncols = min(4, len(ks))
        nrows = (len(ks) + ncols - 1) // ncols
        pl = pv.Plotter(shape=(nrows, ncols), title=f"{self.label} MOs")
        for idx, k in enumerate(ks):
            (x, y, z), psi = self.eval_mo_3d(k, npts=npts, half=half)
            level = iso_frac * float(np.max(np.abs(psi)))
            grid = pv.ImageData(dimensions=psi.shape,
                                spacing=(x[1]-x[0], y[1]-y[0], z[1]-z[0]),
                                origin=(x[0], y[0], z[0]))
            grid.point_data['psi'] = psi.ravel(order='F')
            pl.subplot(idx // ncols, idx % ncols)
            try:
                contours = grid.contour([-level, level], scalars='psi')
                pl.add_mesh(contours, cmap='coolwarm', clim=[-level, level],
                            smooth_shading=True, show_scalar_bar=False)
            except Exception:
                pass
            for p in self.pos:                                     # nuclei
                pl.add_mesh(pv.Sphere(radius=0.16, center=tuple(p)), color='black')
            tag = 'occ' if k < self.nocc else 'virt'
            pl.add_text(f"MO{k+1}  {self.E[k]*HA:.2f} eV ({tag})", font_size=8)
        pl.link_views()                                            # rotate all together
        pl.background_color = 'white'
        print('WINDOW_READY', flush=True)
        pl.show()


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(
        description='Visualise the DFTB molecular orbitals of a geometry.')
    ap.add_argument('geom', nargs='?', default='geometry.xyz',
                    help='xyz geometry file (default geometry.xyz)')
    ap.add_argument('--full', action='store_true', help='full basis (default: minimal valence)')
    ap.add_argument('--mpl', action='store_true', help='matplotlib 3D instead of pyvista')
    ap.add_argument('--2d', dest='twod', action='store_true', help='2D slices instead of 3D')
    ap.add_argument('--mulliken', action='store_true', help='print Mulliken charges and exit')
    ap.add_argument('--levels', action='store_true', help='print the levels and exit (no plot)')
    ap.add_argument('--half', type=float, default=4.0,
                    help='padding (bohr) added around the molecule in the plots')
    # basis-generation knobs, forwarded to INIT.py (only used if the .npz are missing)
    ap.add_argument('--VO', type=int, default=None, dest='vo',
                    help='number of virtual (polarization) shells to add')
    ap.add_argument('--lb94', action='store_true', default=None, help='force the LB94 tail')
    ap.add_argument('--no-lb94', action='store_false', dest='lb94', help='disable the LB94 tail')
    ap.add_argument('--r0-VO', type=float, default=None, dest='r0_vo',
                    help='confinement radius (bohr) for the virtual shells')
    ap.add_argument('--r0', type=float, default=None,
                    help='confinement radius (bohr) for the valence shells')
    args = ap.parse_args()

    viz = MOViz(geom=args.geom, frozen_core=not args.full,
                vo=args.vo, lb94=args.lb94, r0_vo=args.r0_vo, r0=args.r0)
    viz.print_levels()
    if args.mulliken:
        viz.print_charges()
        sys.exit(0)
    if args.levels:
        sys.exit(0)
    if args.twod:
        plt.switch_backend('TkAgg'); viz.plot(which='all', half=args.half)
        print('WINDOW_READY', flush=True); plt.show()
    elif args.mpl:
        plt.switch_backend('TkAgg'); viz.plot3d(which='all', half=args.half)
        print('WINDOW_READY', flush=True); plt.show()
    else:
        viz.plot3d_pyvista(which='all', half=args.half)
