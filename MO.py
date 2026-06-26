#!/usr/bin/python3
"""Visualise the molecular orbitals coming out of the DFTB Hamiltonian.

For each MO k the wavefunction is rebuilt in real space from the eigenvectors,
    psi_k(r) = sum_mu  C[mu, k] * R_{n l}(r_mu) * Y_{l m}(theta_mu, phi_mu),
each AO centred on its atom, and drawn as a signed amplitude in a 2D plane.
The radial part R = u/r comes from the basis; the angular part Y is taken
straight from Y_real.harmonics. Run directly to pop the full MO gallery.
"""
import numpy as np
import matplotlib.pyplot as plt
from Hamiltonian import H
from Y_real import Y_real

HA = 27.21138


class MOViz:
    def __init__(self, distance, minimal_BS=True):
        self.mol = H(distance, minimal_BS=minimal_BS)
        self.mol.H_matrix()
        self.E, self.C = self.mol.diag()
        self.basis = self.mol.basis
        self.distance = distance
        self.nocc = int(round(self.mol.nelec)) // 2
        # diatomic geometry along z: atom 0 at the origin, atom 1 at d.
        self.pos_z = [0.0, distance]

    def eval_mo(self, k, plane='auto', npts=240, half=4.0):
        """Sample MO k on a 2D plane through the molecular (z) axis."""
        C = self.C
        if plane == 'auto':                         # pi_y vanishes in xz -> use yz
            mdom = self.basis[int(np.argmax(np.abs(C[:, k])))].m
            plane = 'yz' if mdom < 0 else 'xz'
        h = np.linspace(-half, half, npts)          # transverse coordinate
        z = np.linspace(-half, self.distance + half, npts)
        Hh, Zz = np.meshgrid(h, z, indexing='ij')
        psi = np.zeros_like(Hh)
        for mu, ao in enumerate(self.basis):
            c = C[mu, k]
            if abs(c) < 1e-9:
                continue
            dz = Zz - self.pos_z[ao.atom]
            r = np.sqrt(Hh**2 + dz**2)
            rs = np.where(r < 1e-9, 1e-9, r)
            theta = np.arccos(np.clip(dz / rs, -1.0, 1.0))
            if plane == 'xz':
                phi = np.where(Hh >= 0, 0.0, np.pi)
            else:                                    # yz plane
                phi = np.where(Hh >= 0, np.pi / 2, -np.pi / 2)
            R = np.interp(r.ravel(), ao.grid, ao.u / ao.grid,
                          right=0.0).reshape(r.shape)
            Y = Y_real.harmonics[(ao.l, ao.m)](theta, phi)
            psi += c * R * Y
        return Zz, Hh, psi, plane

    def plot(self, which='all', npts=240, half=4.0):
        ks = list(range(self.mol.N)) if which == 'all' else list(which)
        ncols = min(4, len(ks))
        nrows = (len(ks) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.4 * nrows),
                                 squeeze=False)
        for idx, k in enumerate(ks):
            ax = axes[idx // ncols][idx % ncols]
            Zz, Hh, psi, plane = self.eval_mo(k, npts=npts, half=half)
            vmax = np.max(np.abs(psi)) or 1.0
            lv = np.linspace(-vmax, vmax, 41)
            ax.contourf(Zz, Hh, psi, levels=lv, cmap='RdBu_r', extend='both')
            ax.contour(Zz, Hh, psi, levels=[0.0], colors='k', linewidths=0.4)
            ax.plot(self.pos_z, [0, 0], 'ko', ms=4)          # nuclei
            tag = 'occ' if k < self.nocc else 'virt'
            ax.set_title(f"MO{k+1}   {self.E[k]*HA:.2f} eV  ({tag})\n[{plane}]",
                         fontsize=9)
            ax.set_aspect('equal')
            ax.set_xticks([]); ax.set_yticks([])
        for idx in range(len(ks), nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)
        fig.suptitle(f"{'-'.join(self.mol.names)}  molecular orbitals "
                     f"(d = {self.distance:.3f} bohr)", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        return fig

    def eval_mo_3d(self, k, npts=70, half=4.0):
        """Sample MO k on a full 3D (x,y,z) grid."""
        C = self.C
        x = np.linspace(-half, half, npts)
        y = np.linspace(-half, half, npts)
        z = np.linspace(-half, self.distance + half, npts)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        psi = np.zeros_like(X)
        for mu, ao in enumerate(self.basis):
            c = C[mu, k]
            if abs(c) < 1e-9:
                continue
            dz = Z - self.pos_z[ao.atom]
            r = np.sqrt(X**2 + Y**2 + dz**2)
            rs = np.where(r < 1e-9, 1e-9, r)
            theta = np.arccos(np.clip(dz / rs, -1.0, 1.0))
            phi = np.arctan2(Y, X)                       # full azimuth -> all m correct
            R = np.interp(r.ravel(), ao.grid, ao.u / ao.grid,
                          right=0.0).reshape(r.shape)
            psi += c * R * Y_real.harmonics[(ao.l, ao.m)](theta, phi)
        return (x, y, z), psi

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
            ax.scatter([0, 0], [0, 0], self.pos_z, color='k', s=25)  # nuclei
            tag = 'occ' if k < self.nocc else 'virt'
            ax.set_title(f"MO{k+1}  {self.E[k]*HA:.2f} eV ({tag})", fontsize=9)
            ax.set_axis_off()
            try:
                ax.set_box_aspect((1, 1, 1.3))
            except Exception:
                pass
        fig.suptitle(f"{'-'.join(self.mol.names)}  molecular orbitals "
                     f"(d = {self.distance:.3f} bohr)  —  red/blue = +/- phase",
                     fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        return fig

    def print_levels(self):
        """Print the MO eigenvalues with occupation, HOMO/LUMO and gap."""
        print(f"\n{'-'.join(self.mol.names)}  MO levels  (d = {self.distance:.3f} bohr)")
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

    def plot3d_pyvista(self, which='all', npts=80, half=4.0, iso_frac=0.18):
        """Smooth, GPU (VTK) isosurfaces — one subplot per MO, linked cameras."""
        import pyvista as pv
        ks = list(range(self.mol.N)) if which == 'all' else list(which)
        ncols = min(4, len(ks))
        nrows = (len(ks) + ncols - 1) // ncols
        pl = pv.Plotter(shape=(nrows, ncols),
                        title=f"{'-'.join(self.mol.names)} MOs")
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
            for zi in self.pos_z:                                  # nuclei
                pl.add_mesh(pv.Sphere(radius=0.16, center=(0, 0, zi)), color='black')
            tag = 'occ' if k < self.nocc else 'virt'
            pl.add_text(f"MO{k+1}  {self.E[k]*HA:.2f} eV ({tag})", font_size=8)
        pl.link_views()                                            # rotate all together
        pl.background_color = 'white'
        print('WINDOW_READY', flush=True)
        pl.show()


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(
        description='Visualise the DFTB molecular orbitals of whatever pair is in ATOMS_BS.')
    ap.add_argument('d', type=float, nargs='?', default=2.28,
                    help='bond length in bohr (default 2.28)')
    ap.add_argument('--ang', action='store_true', help='interpret d as Angstrom instead of bohr')
    ap.add_argument('--full', action='store_true', help='full basis (default: minimal valence)')
    ap.add_argument('--mpl', action='store_true', help='matplotlib 3D instead of pyvista')
    ap.add_argument('--2d', dest='twod', action='store_true', help='2D slices instead of 3D')
    args = ap.parse_args()

    d = args.d * 1.8897259886 if args.ang else args.d     # Angstrom -> bohr
    viz = MOViz(distance=d, minimal_BS=not args.full)
    viz.print_levels()
    if args.twod:
        plt.switch_backend('TkAgg'); viz.plot(which='all')
        print('WINDOW_READY', flush=True); plt.show()
    elif args.mpl:
        plt.switch_backend('TkAgg'); viz.plot3d(which='all')
        print('WINDOW_READY', flush=True); plt.show()
    else:
        viz.plot3d_pyvista(which='all')
