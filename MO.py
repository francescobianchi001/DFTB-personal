#!/usr/bin/env python3
"""Visualise the molecular orbitals coming out of the DFTB Hamiltonian.

For each MO k the wavefunction is rebuilt in real space from the eigenvectors,
    psi_k(r) = sum_mu  C[mu, k] * R_{n l}(r_mu) * Y_{l m}(theta_mu, phi_mu),
each AO centred on its atom. The radial part R = u/r comes from the basis; the
angular part Y is taken straight from Y_real.harmonics.

The geometry (any number of atoms) is read from an xyz file -- pass it on the
command line, e.g.  ./MO.py N2.xyz.  All AOs are evaluated with their true 3D
cartesian offsets, so the molecule does not have to lie on any particular axis.

By default this opens ONE orbital at a time (starting at the HOMO) and the
arrow keys step through the rest -- the gmolden/GaussView habit. The AO values
on the grid do not depend on which MO is drawn, so they are evaluated once and
cached per AO on its own sub-box; each orbital after the first is then a
handful of slice-adds. On progesterone (54 atoms, 132 MOs) that is ~7 s once
and ~0.1 s per orbital, against ~4.6 s EVERY orbital before -- and at a finer,
molecule-size-independent resolution. --gallery restores the old wall of
subplots, --2d the plane slices, --mpl the matplotlib isosurfaces.
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
        self._ao_key = self._ao_val = None      # AO-on-grid cache (see _ao_stack)
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

    # Cap on the AO cache. Past this it is cheaper to pay the per-MO cost than
    # to hold the stack; the uncached path stays available as a fallback.
    AO_CACHE_MAX_BYTES = 2_000_000_000

    def _ao_at(self, ao, pts):
        """One AO evaluated at cartesian points `pts` of shape (..., 3)."""
        d = pts - self.pos[ao.atom]                      # offset from its centre
        r = np.sqrt((d**2).sum(-1))
        rs = np.where(r < 1e-9, 1e-9, r)
        theta = np.arccos(np.clip(d[..., 2] / rs, -1.0, 1.0))
        phi = np.arctan2(d[..., 1], d[..., 0])
        R = np.interp(r.ravel(), ao.grid, ao.u / ao.grid,
                      right=0.0).reshape(r.shape)
        return R * Y_real.harmonics[(ao.l, ao.m)](theta, phi)

    def _ao_stack(self, pts, key):
        """Every AO sampled on `pts`, cached under `key`. Returns None if too big.

        The AO values on a grid do not depend on WHICH MO is being drawn -- only
        the coefficients do. Evaluating them once turns each subsequent MO into
        one tensordot. Used for the 2D slices, where the point set is small and
        need not be axis-aligned; the 3D path uses _ao_boxes instead.
        """
        nbytes = len(self.basis) * int(np.prod(pts.shape[:-1])) * 4
        if nbytes > self.AO_CACHE_MAX_BYTES:
            return None
        if self._ao_key != key:
            self._ao_val = None                          # free the old one first
            AO = np.empty((len(self.basis),) + pts.shape[:-1], dtype=np.float32)
            for mu, ao in enumerate(self.basis):
                AO[mu] = self._ao_at(ao, pts)
            self._ao_key, self._ao_val = key, AO
        return self._ao_val

    @staticmethod
    def _rcut(ao, tol=1e-6):
        """Radius past which this AO contributes nothing worth drawing.

        u = r*R is tabulated, so beyond the last point where |u| > tol*max|u|
        the radial function is dead. Conservative for R = u/r too, since that
        cutoff always lands at r > 1 where R < u.
        """
        a = np.abs(ao.u)
        nz = np.nonzero(a > tol * a.max())[0]
        return float(ao.grid[nz[-1]]) if len(nz) else 0.0

    def _ao_boxes(self, axes, key):
        """Each AO evaluated only on the sub-box it actually reaches.

        AOs are local: on a 54-atom box a typical one is non-negligible over
        ~20% of the volume, so storing full-grid copies wastes most of the
        memory. Keeping (slice, values) per AO cuts both the footprint and the
        one-off build by ~4-5x, which is what makes a resolution fine enough
        for big molecules affordable. Returns a list of (slices, values|None).
        """
        if self._ao_key == key:
            return self._ao_val
        self._ao_val = None                              # free the old one first
        boxes = []
        for ao in self.basis:
            rc, c = self._rcut(ao), self.pos[ao.atom]
            sl, sub = [], []
            for i, a in enumerate(axes):
                j0 = int(np.searchsorted(a, c[i] - rc, 'left'))
                j1 = int(np.searchsorted(a, c[i] + rc, 'right'))
                sl.append(slice(j0, j1))
                sub.append(a[j0:j1])
            if any(len(s) == 0 for s in sub):            # AO misses the box
                boxes.append((tuple(sl), None))
                continue
            P = np.stack(np.meshgrid(*sub, indexing='ij'), axis=-1)
            boxes.append((tuple(sl), self._ao_at(ao, P).astype(np.float32)))
        mb = sum(v.nbytes for _, v in boxes if v is not None) / 1e6
        if mb > 200:                                     # worth knowing about
            print(f'  [MO cache {mb:,.0f} MB for a '
                  f'{"x".join(str(len(a)) for a in axes)} grid; '
                  f'use --spacing to trade detail for memory]')
        self._ao_key, self._ao_val = key, boxes
        return boxes

    def _psi_grid(self, k, axes, key):
        """MO k on the axis-aligned grid `axes`, accumulated from the sub-boxes."""
        boxes = self._ao_boxes(axes, key)
        psi = np.zeros(tuple(len(a) for a in axes), dtype=np.float32)
        for mu, (sl, val) in enumerate(boxes):
            c = self.C[mu, k]
            if val is None or abs(c) < 1e-9:
                continue
            psi[sl] += np.float32(c) * val
        return psi

    def _psi_at(self, k, pts, key=None):
        """Evaluate MO k at cartesian points `pts` of shape (..., 3).

        With `key`, go through the AO cache (fast when several MOs share one
        grid); without, evaluate the AOs inline for this MO only.
        """
        if key is not None:
            AO = self._ao_stack(pts, key)
            if AO is not None:
                return np.tensordot(self.C[:, k].astype(np.float32), AO, axes=(0, 0))
        psi = np.zeros(pts.shape[:-1])
        for mu, ao in enumerate(self.basis):
            c = self.C[mu, k]
            if abs(c) < 1e-9:
                continue
            psi += c * self._ao_at(ao, pts)
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
        key = ('2d', plane, npts, half)
        return Ss, Tt, self._psi_at(k, pts, key), plane, nuc

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

    # Isosurface resolution in bohr. Fixing the SPACING rather than the point
    # count is what keeps a 54-atom molecule as sharp as a 6-atom one -- with a
    # fixed 80^3 the big box came out at 0.35 bohr and visibly under-resolved
    # (progesterone's HOMO surface had 654 points against cyclohexane's ~10k).
    # 0.17 matches the FINEST axis the old fixed grid happened to give, so no
    # molecule renders coarser than before and the big ones render far better.
    GRID_SPACING = 0.17
    GRID_MAX_POINTS = 6_000_000            # guard for very large molecules

    def _grid_axes(self, half=4.0, npts=None, spacing=None):
        """Per-axis nodes boxing the molecule at (near-)isotropic resolution."""
        lo, hi = self.pos.min(axis=0) - half, self.pos.max(axis=0) + half
        ext = hi - lo
        if npts is not None:                             # npts = points on the LONG axis
            n = np.maximum(2, np.rint(npts * ext / ext.max()).astype(int))
        else:
            sp = spacing or self.GRID_SPACING
            n = np.maximum(2, np.rint(ext / sp).astype(int) + 1)
            if int(np.prod(n)) > self.GRID_MAX_POINTS:   # coarsen uniformly to fit
                n = np.maximum(2, (n * (self.GRID_MAX_POINTS
                                        / np.prod(n)) ** (1 / 3)).astype(int))
        return [np.linspace(lo[i], hi[i], int(n[i])) for i in range(3)]

    def eval_mo_3d(self, k, npts=None, half=4.0, spacing=None):
        """Sample MO k on a full 3D (x,y,z) grid boxing the molecule."""
        axes = self._grid_axes(half=half, npts=npts, spacing=spacing)
        key = ('3d', half, tuple(len(a) for a in axes))
        return axes, self._psi_grid(k, axes, key)

    def plot3d(self, which='all', npts=None, half=4.0, iso_frac=0.2, spacing=None):
        from skimage import measure
        ks = list(range(self.mol.N)) if which == 'all' else list(which)
        ncols = min(4, len(ks))
        nrows = (len(ks) + ncols - 1) // ncols
        fig = plt.figure(figsize=(3.6 * ncols, 3.6 * nrows))
        for idx, k in enumerate(ks):
            (x, y, z), psi = self.eval_mo_3d(k, npts=npts, half=half, spacing=spacing)
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

    def view3d_pyvista(self, start=None, npts=None, half=4.0, iso_frac=0.18,
                       spacing=None):
        """One MO at a time in a single window; arrow keys step through them.

        Same isosurfaces as plot3d_pyvista, just one per window instead of a
        wall of subplots -- which is both easier to read and much faster, since
        only the orbital on screen is drawn. The AO cache makes each step a
        tensordot, so navigation is immediate even for a 50-atom molecule.

        Keys:  Right / Up / n  next      Left / Down / p  previous
               h  HOMO      l  LUMO      Home  first      End  last
        """
        import pyvista as pv
        ks = list(range(self.mol.N))
        cur = [self.nocc - 1 if start is None else int(start)]
        cur[0] = min(max(cur[0], 0), len(ks) - 1)

        pl = pv.Plotter(title=f"{self.label} MOs — {self.geom}")
        pl.background_color = 'white'
        for p in self.pos:                                     # nuclei (static)
            pl.add_mesh(pv.Sphere(radius=0.16, center=tuple(p)), color='black')
        state = {'actor': None}

        def draw():
            k = ks[cur[0]]
            (x, y, z), psi = self.eval_mo_3d(k, npts=npts, half=half, spacing=spacing)
            level = iso_frac * float(np.max(np.abs(psi)))
            if state['actor'] is not None:
                pl.remove_actor(state['actor'], reset_camera=False)
                state['actor'] = None
            if level > 0:
                grid = pv.ImageData(dimensions=psi.shape,
                                    spacing=(x[1]-x[0], y[1]-y[0], z[1]-z[0]),
                                    origin=(x[0], y[0], z[0]))
                grid.point_data['psi'] = psi.ravel(order='F')
                try:
                    contours = grid.contour([-level, level], scalars='psi')
                    state['actor'] = pl.add_mesh(
                        contours, cmap='coolwarm', clim=[-level, level],
                        smooth_shading=True, show_scalar_bar=False,
                        reset_camera=False)
                except Exception:
                    pass
            tag = 'occ' if k < self.nocc else 'virt'
            extra = ('  <- HOMO' if k == self.nocc - 1 else
                     '  <- LUMO' if k == self.nocc else '')
            pl.add_text(f"MO{k+1}/{len(ks)}   {self.E[k]*HA:.2f} eV ({tag}){extra}\n"
                        f"arrows: step   h/l: HOMO/LUMO",
                        font_size=9, name='molabel', color='black')
            pl.render()

        def step(n):
            def go():
                cur[0] = min(max(cur[0] + n, 0), len(ks) - 1)
                draw()
            return go

        def goto(i):
            def go():
                cur[0] = min(max(i, 0), len(ks) - 1)
                draw()
            return go

        for key, fn in (('Right', step(+1)), ('Up', step(+1)), ('n', step(+1)),
                        ('Left', step(-1)), ('Down', step(-1)), ('p', step(-1)),
                        ('h', goto(self.nocc - 1)), ('l', goto(self.nocc)),
                        ('Home', goto(0)), ('End', goto(len(ks) - 1))):
            pl.add_key_event(key, fn)

        draw()
        print('WINDOW_READY', flush=True)
        pl.show()

    def plot3d_pyvista(self, which='all', npts=None, half=4.0, iso_frac=0.18,
                       spacing=None):
        """Smooth, GPU (VTK) isosurfaces — one subplot per MO, linked cameras."""
        import pyvista as pv
        ks = list(range(self.mol.N)) if which == 'all' else list(which)
        ncols = min(4, len(ks))
        nrows = (len(ks) + ncols - 1) // ncols
        pl = pv.Plotter(shape=(nrows, ncols), title=f"{self.label} MOs")
        for idx, k in enumerate(ks):
            (x, y, z), psi = self.eval_mo_3d(k, npts=npts, half=half, spacing=spacing)
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
    ap.add_argument('--gallery', action='store_true',
                    help='all MOs as a wall of subplots (default: one at a time, '
                         'arrow keys to step)')
    ap.add_argument('--mo', type=int, default=None, metavar='K',
                    help='MO to open on, 1-based (default: HOMO)')
    ap.add_argument('--npts', type=int, default=None,
                    help='isosurface grid points along the LONGEST axis; overrides '
                         '--spacing')
    ap.add_argument('--spacing', type=float, default=None,
                    help='isosurface resolution in bohr (default 0.22, kept constant '
                         'so big molecules stay as sharp as small ones)')
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
    elif args.gallery:
        viz.plot3d_pyvista(which='all', half=args.half, npts=args.npts,
                           spacing=args.spacing)
    else:
        viz.view3d_pyvista(start=None if args.mo is None else args.mo - 1,
                           half=args.half, npts=args.npts, spacing=args.spacing)
