#!/usr/bin/python3

import numpy as np
import re
import sys
from pathlib import Path
import matplotlib.pyplot as pl
from scipy.integrate import simpson
import subprocess as sub
from collections import namedtuple
from read_xyz import get_coords, ATOM_NAMES

# The real spherical harmonics, the SK angular weights and the SK rotation table
# (bond direction cosines -> combination of V_{l l' m}) are all generated once
# into Y_real.py. Regenerate only if it is missing or predates one of those
# blocks -- the expressions themselves never change -- before anything imports
# it. Doing this at module load keeps every run quiet.
def ensure_Y_real(path=Path("Y_real.py")):
    if path.exists() and 'rotations' in path.read_text():
        return
    sub.run(["SK/./Spherical_Harmonics.py"], check=True)

ensure_Y_real()

def prepare_atoms(geom, vo=None, lb94=None, r0_vo=None, r0=None):
    """Make sure the per-atom .npz data for every element in `geom` exists.

    Reads the geometry, maps each atomic number to its element symbol (via
    ATOM_NAMES), writes those elements into INIT.py's ATOMS dict, and runs
    INIT.py to generate the basis/potential/eig files -- but ONLY if any of
    the expected ATOMS_BS/<symbol>.npz files are missing. If they are all
    already present nothing is recomputed. Returns {symbol: Z}.

    Generation options are forwarded to INIT.py's CLI:
      vo   : int or None -- add this many virtual (polarization) shells (--VO N).
      lb94 : True  -> force the LB94 -1/r tail on the free-atom solve (--lb94)
             False -> disable it (--no-lb94)
             None  -> let INIT decide (LB94 defaults on iff vo is set).
      r0_vo: float or None -- weaker confinement radius (bohr) for the virtual
             shells (--r0-VO). Triggers the split-confinement solve that writes
             vo_shells/Vconf_VO, activating the Rayleigh on-site + VO-wall
             off-diagonal treatment. Needs vo to be set to have any effect.
    """
    atno, coords = get_coords(geom, maxlen=100)
    # unique elements in the geometry, Z -> capitalized symbol ("cl" -> "Cl")
    atoms = {ATOM_NAMES[int(Z)].capitalize(): int(Z) for Z in atno}

    p_bs = Path.cwd() / 'ATOMS_BS'
    missing = [sym for sym in atoms if not (p_bs / f'{sym}.npz').exists()]
    if not missing:
        return atoms

    print(f"prepare_atoms: missing basis data for {missing} -> running INIT.py")

    # Rewrite the ATOMS = { ... } dict in INIT.py so its solve targets exactly
    # the elements this geometry needs, then run it (INIT wipes and rebuilds
    # ATOMS_BS / ATOMS_POT / eig_neutral for every element in that dict).
    init_path = Path.cwd() / 'INIT.py'
    text = init_path.read_text()
    block = 'ATOMS = {\n' + ''.join(
        f'    "{sym}": {Z},\n' for sym, Z in atoms.items()) + '}'
    text = re.sub(r'ATOMS = \{.*?\}', block, text, count=1, flags=re.DOTALL)
    init_path.write_text(text)

    # Forward the chosen generation options to INIT.py's argparse CLI.
    extra = []
    if vo is not None:
        extra += ['--VO', str(vo)]
    if lb94 is True:
        extra.append('--lb94')
    elif lb94 is False:
        extra.append('--no-lb94')
    if r0_vo is not None:
        extra += ['--r0-VO', str(r0_vo)]
    if r0 is not None:
        extra += ['--r0', str(r0)]

    sub.run([sys.executable, str(init_path), *extra], check=True)
    return atoms

# One basis function (atomic orbital). Its position in the flat basis list is
# its row/column index in H and S, so no separate index map is needed.
AO = namedtuple('AO', 'atom elem n l m u grid d R')


class H:

    def __init__(self,distance=None,frozen_core=True,grid=None,geom='geometry.xyz',
                 vo=None,lb94=None,r0_vo=None,r0=None):

        # Make sure every element in the geometry has its .npz data on disk
        # (runs INIT.py only if something is missing), then read the geometry.
        # vo / lb94 / r0_vo / r0 choose how those files are generated (see prepare_atoms).
        prepare_atoms(geom, vo=vo, lb94=lb94, r0_vo=r0_vo, r0=r0)
        atom,coords = get_coords(geom, maxlen=100)

        p_bs = Path.cwd()/'ATOMS_BS'
        p_pot = Path.cwd()/'ATOMS_POT'

        atoms = []
        self.names = []          # element label per atom (file stem)
        self.Znum = []           # atomic number (self.Z is taken by the z-grid)


        # Sort all three dirs by stem so ATOMS_BS / ATOMS_POT / eig_neutral load in
        # the SAME element order -> a single element index `e` addresses all of them.
        for entry in sorted(p_bs.iterdir(), key=lambda p: p.stem):
            if entry.is_file():
                data = np.load(entry,allow_pickle=True)   # don't clobber `atom`
                atoms.append(data), self.names.append(entry.stem)
                self.Znum.append(int(data['Z']))

        V = []

        for entry in sorted(p_pot.iterdir(), key=lambda p: p.stem):
            if entry.is_file():
                V_entry = np.load(entry,allow_pickle=True)
                V.append(V_entry)

        self.r = [data['grid'] for data in atoms]
        self.occupied = [data['occupied'].tolist() for data in atoms]
        self.basisets =  [data['wavefunctions'].tolist() for data in atoms]
        self.eigenvalues = [data['eigenvalues'].tolist() for data in atoms]
        self.Veff = [data['Veff'] for data in V]
        self.Vconf = [data['Vconf']  for data in V]
        # split confinement: virtual shells listed in vo_shells use Vconf_VO
        # instead of Vconf. Absent (single-wall runs) -> None / empty set.
        self.Vconf_VO = [data['Vconf_VO'] if 'Vconf_VO' in data.files else None
                         for data in V]
        self.vo_shells = [set(map(tuple, data['vo_shells'].tolist()))
                          if 'vo_shells' in data.files else set() for data in V]
        
        self.atoms, self.coords = atom,coords         # atom = atomic numbers (natoms,)
        # map atomic number -> index into the per-element loaded arrays above, so
        # a physical atom `a` reaches its basis/potential via self.Z2elem[self.atoms[a]].
        self.Z2elem = {Z: e for e, Z in enumerate(self.Znum)}

        pth = Path.cwd()/'eig_neutral'

        eig_neutral = []
        for entry in sorted(pth.iterdir(), key=lambda p: p.stem):
            if entry.is_file():
                eig_neutral.append(np.load(entry,allow_pickle=True))
        self.eigN = [data['eigenvalues'].tolist() for data in eig_neutral]
        self.Vneutral = [data['Veff'] for data in eig_neutral]

        if grid!=None:
            self.r = self.r[:grid]
            self.occupied = self.occupied
            self.basisets = [[[self.basisets[i][j][k][:grid]
                               for k in range(len(self.basisets[i][j]))]
                              for j in range(len(self.basisets[i]))]
                             for i in range(len(self.basisets))]

        self.p = Path.cwd()
        self._gkey = self._gval = None     # last pair grid built (see _grid)
        self.frozen_core = frozen_core
        self.distance = distance          # vestigial: bond lengths now come from the
                                          # geometry (self.dist). Kept for the legacy
                                          # SK_int_parametre / constructor contract.

    def SK_int_parametre(self, distance):

        pth = self.p / 'ATOMS_BS'
        import sys
        sk_dir = self.p / 'SK'
        if str(sk_dir) not in sys.path:
            sys.path.insert(0, str(sk_dir))
        from SlaterKonster import slaterkonster
        SK = slaterkonster()
        S = SK.Space_definment(self.distance)

        return S

    def build_grid(self,d,N=400):
        """Quadrature grid for ONE pair at separation d. Returns the meshes and
        the 1D node arrays Simpson needs, without touching instance state.

        The z range is tied to d (not to the largest distance in the molecule):
        at fixed N a bigger span means a coarser dz, and while the overlap is
        smooth enough not to care, the H integrand's -Z/r cusps are resolved
        badly enough that stretching the span from a bond length to 30 bohr
        costs ~0.1 -> ~0.5 eV on individual two-center potential integrals.
        Gridding per pair keeps every bond at diatomic resolution.

        z stays symmetric about the bond midpoint d/2, which makes the
        reflection z -> d-z map the node set onto itself. NB the A<->B exchange
        identity does NOT rely on that: it holds pointwise on the integrand
        (see cross_terms), so it survives any choice of grid.
        """
        rmax = self.r[0][-1]
        r0 = self.r[0][0]
        x = np.linspace(r0,rmax,N)
        z = np.linspace(-rmax, d + rmax, N)
        X,Z = np.meshgrid(x,z,indexing='ij')
        return X,Z,x,z

    def _grid(self,d,N=400):
        """build_grid with the last result remembered.

        fill_cross asks for min(l1,l2)+1 channels per shell pair and they all
        share a d, so a single remembered grid is hit every time after the
        first. One slot, not a dict: each grid is ~2.6 MB, and a molecule has
        far too many pairs to keep them all.
        """
        if self._gkey != (d, N):
            self._gkey, self._gval = (d, N), self.build_grid(d, N)
        return self._gval

    def build_basis(self,dAB):
        LMAX = 2                                              # SK integrals stop at d
        self.basis = []
        self.nelec = 0
        for a in range(len(self.atoms)):
            e = self.Z2elem[self.atoms[a]]                    # element-data index
            if self.frozen_core:
                val = max(n for n in range(len(self.basisets[e]))
                          if any(self.occupied[e][n]))
                shells = range(val, len(self.basisets[e]))
            else:
                shells = range(len(self.basisets[e]))
            R = self.coords[0][a]                             # this atom's xyz
            for n in shells:
                for l in range(len(self.basisets[e][n])):
                    if l > LMAX:
                        continue
                    u = np.asarray(self.basisets[e][n][l])
                    if not np.any(u):                    # zeroed (VO-keep) / absent shell
                        continue
                    self.nelec += self.occupied[e][n][l]
                    for m in range(-l, l + 1):
                        self.basis.append(AO(a, e, n, l, m, u, self.r[e], dAB[a], R))
        self.N = len(self.basis)
        return self.N

    def space(self):
        R = self.coords[0]                               
        self.m = R.mean(axis=0)                         
        self.dist = np.sqrt(((R[:, None, :] - R[None, :, :])**2).sum(-1))
        self.build_basis(self.dist)
        return self.m
     
    def cross_terms(self,A,B,o=None):
        from Y_real import Y_real
        if o is None:
            o = abs(A.m)                                            # sigma/pi/delta channel
        d = A.d[B.atom]
        X, Zg, xg, zg = self._grid(d)          # this pair's own grid

        wall_A = self.Vconf[A.elem]
        if (A.n, A.l) in self.vo_shells[A.elem] and self.Vconf_VO[A.elem] is not None:
            wall_A = self.Vconf_VO[A.elem]
        wall_B = self.Vconf[B.elem]
        if (B.n, B.l) in self.vo_shells[B.elem] and self.Vconf_VO[B.elem] is not None:
            wall_B = self.Vconf_VO[B.elem]
        VphysA = self.Veff[A.elem] - self.Vconf[A.elem]
        VphysB = self.Veff[B.elem] - self.Vconf[B.elem]
        Vn_A = self.Vneutral[A.elem]
        Vn_B = self.Vneutral[B.elem]
        AW = Y_real.angular_weights[(A.l, B.l, o)]

        # S_SK and Vint integrands, sampled at bond-axis coordinate Zc.
        def integrand(Zc):
            rA = np.sqrt(X**2 + Zc**2)
            rB = np.sqrt(X**2 + (Zc - d)**2)
            RA = np.interp(rA.ravel(), A.grid, A.u / A.grid, right=0.0).reshape(rA.shape)
            RB = np.interp(rB.ravel(), B.grid, B.u / B.grid, right=0.0).reshape(rB.shape)
            VJ = (np.interp(rA.ravel(), A.grid, Vn_A, right=0.0).reshape(rA.shape)
                  + np.interp(rB.ravel(), B.grid, Vn_B, right=0.0).reshape(rB.shape)
                  - 0.5 * (np.interp(rA.ravel(), A.grid, VphysA, right=0.0).reshape(rA.shape)
                           + np.interp(rB.ravel(), B.grid, VphysB, right=0.0).reshape(rB.shape)
                           + np.interp(rA.ravel(), A.grid, wall_A, right=0.0).reshape(rA.shape)
                           + np.interp(rB.ravel(), B.grid, wall_B, right=0.0).reshape(rB.shape)))
            b = RA * RB * AW(X, Zc, d) * X # X = rho Jacobian
            return b, b * VJ

        # Reflection-average over z -> d-z. Analytically the integral is invariant
        # (z -> d-z is a change of variable, exact for A != B as much as for A == B);
        # numerically it symmetrizes the quadrature. What it really buys is the
        # A<->B EXCHANGE identity: pointwise at every node the integrand obeys
        #   f_AB(d-z) = (-1)^(l1+l2) f_BA(z)
        # (the radial factors swap, and the angular weight flips because
        # cos t1|_(d-z) = -cos t2|_z with P_l^m(-u) = (-1)^(l+m) P_l^m(u)), so the
        # AVERAGED integrand is exactly (-1)^(l1+l2) times the swapped one and
        #   V_{l1 l2 m}(A->B) = (-1)^(l1+l2) V_{l2 l1 m}(B->A)
        # holds to the last bit. H_matrix only ever evaluates ONE ordering per pair
        # (fill_cross gets A = the later atom, since the loop runs nu <= mu), so
        # without this the answer would depend on the order the atoms happen to be
        # listed in the xyz file -- 0.14 eV on the CO MO energies when tested.
        bS1, bV1 = integrand(Zg)
        bS2, bV2 = integrand(d - Zg)
        baseS = 0.5 * (bS1 + bS2)
        baseV = 0.5 * (bV1 + bV2)
        S_o = simpson(simpson(baseS, x=xg, axis=0), x=zg)
        V_o = simpson(simpson(baseV, x=xg, axis=0), x=zg)
        return S_o, V_o

    def fill_diag(self, A, mu):
        norm = simpson(A.u**2, x=A.grid)             # <phi|phi> ~ 1
        if abs(1 - norm) >= 1e-3:
            raise ValueError(
                f"AO {mu} (atom {A.atom}, n{A.n} l{A.l} m{A.m}) "
                f"not normalized: <phi|phi>={norm}")
        if (A.n, A.l) in self.vo_shells[A.elem]:
            # VO on-site: Rayleigh quotient of the confined orbital against the
            # free hamiltonian. Sits between eigN (variational min -> too deep)
            # and the confined eps (wall energy -> too high). 1D radial integral.
            r, u = A.grid, A.u
            Vphys = self.Veff[A.elem] - self.Vconf[A.elem]
            up = np.gradient(u, r)
            centrifugal = A.l * (A.l + 1) / (2.0 * r**2)
            E_ray = simpson(0.5 * up**2 + (centrifugal + Vphys) * u**2, x=r)
            # bracket check: neutral level (too deep) < Rayleigh < confined eps (too high)
            eig_neutral = self.eigN[A.elem][A.n][A.l]
            eps_conf    = self.eigenvalues[A.elem][A.n][A.l]
            if A.m == -A.l:
                print(f"[VO bracket] atom {A.atom} ({self.names[A.elem]}) "
                      f"n{A.n} l{A.l}: eigN={eig_neutral:+.6f} < "
                      f"E_ray={E_ray:+.6f} < eps_conf={eps_conf:+.6f}")
            if not (eig_neutral <= E_ray <= eps_conf):
                print(f"WARNING: VO Rayleigh out of bracket for atom {A.atom} "
                      f"n{A.n} l{A.l}: eigN={eig_neutral:+.6f}, E_ray={E_ray:+.6f}, "
                      f"eps_conf={eps_conf:+.6f}")
            self.S[mu, mu] = 1.0
            self.H[mu, mu] = E_ray
            return
        self.S[mu, mu] = 1.0
        self.H[mu, mu] = self.eigN[A.elem][A.n][A.l]   # neutral on-site level

    def fill_cross(self, A, B, mu, nu):
        from Y_real import Y_real
        l1, l2 = A.l, B.l
        d = A.d[B.atom]
        Lc, Mc, Nc = (B.R - A.R) / d                    # bond direction cosines A -> B

        # Bond-frame reduced integrals V_{l1 l2 |m|}: depend only on the two
        # shells and their separation, so compute once per shell pair and reuse
        # for every (m1, m2).
        key = (A.atom, A.n, l1, B.atom, B.n, l2)
        if key not in self._vcache:
            self._vcache[key] = {o: self.cross_terms(A, B, o)
                                 for o in range(min(l1, l2) + 1)}
        chan = self._vcache[key]

        # E_{m1 m2} = sum_m rotations[(l1,m1,l2,m2,m)](L,M,N) * V_{l1 l2 m}.
        # The rotation table decides which pairs vanish (missing key -> 0).
        S_rot = V_rot = 0.0
        for o, (S_o, V_o) in chan.items():
            c = Y_real.rotations.get((l1, A.m, l2, B.m, o))
            if c is None:                               # channel absent -> 0
                continue
            w = c(Lc, Mc, Nc)
            S_rot += w * S_o
            V_rot += w * V_o

        eps_a = self.eigenvalues[A.elem][A.n][l1]       # confined eps for the eps*S trick
        eps_b = self.eigenvalues[B.elem][B.n][l2]
        H_rot = 0.5 * (eps_a + eps_b) * S_rot + V_rot
        self.S[mu, nu] = self.S[nu, mu] = S_rot
        self.H[mu, nu] = self.H[nu, mu] = H_rot

    def H_matrix(self):
        ensure_Y_real()                                # cache: generate only once
        self.space()                                   # basis + distance matrix + center
        self.H = np.zeros((self.N, self.N))
        self.S = np.zeros((self.N, self.N))
        self._vcache = {}                              # bond-frame V_{ll'm} per shell pair
        self._gkey = self._gval = None                 # last grid built (see _grid)

        for mu in range(self.N):
            for nu in range(mu + 1):       # lower triangle + diagonal
                A, B = self.basis[mu], self.basis[nu]
                if A.atom == B.atom:
                    if mu == nu:
                        self.fill_diag(A, mu)
                    # same-atom off-diagonal = 0, leave it
                else:
                    self.fill_cross(A, B, mu, nu)
        return self.H

    def diag(self, thresh=1e-6):
        H = 0.5 * (self.H + self.H.T)
        S = 0.5 * (self.S + self.S.T)
        self.S = S

        s, U = np.linalg.eigh(S)
        keep = s > thresh
        ndrop = int(np.sum(~keep))
        if ndrop:
            print(f"diag: dropping {ndrop} of {len(s)} basis modes "
                  f"(min S eig {s.min():.2e} <= {thresh:.0e}) — basis near-singular/over-complete")
        U, s = U[:, keep], s[keep]

        X = U / np.sqrt(s)
        E, C_ = np.linalg.eigh(X.T @ H @ X)
        C = X @ C_

        assert np.allclose(C.T @ S @ C, np.eye(C.shape[1]), atol=1e-8)
        if ndrop == 0:
            assert np.allclose(H @ C, S @ C @ np.diag(E), atol=1e-6)

        return E, C

    def band_energy(self):
        E, C = self.diag()
        nocc = int(round(self.nelec)) // 2
        return 2.0 * np.sum(E[:nocc])

    def density_matrix(self):
        E, C = self.diag()
        nocc = int(round(self.nelec)) // 2
        Cocc = C[:, :nocc]
        P = 2.0 * Cocc @ Cocc.T
        return P

    def Mulliken_charge(self):
        P = self.density_matrix()
        gross = np.diag(P @ self.S)
        natoms = len(self.atoms)
        q = np.zeros(natoms)
        for mu, ao in enumerate(self.basis):
            q[ao.atom] += gross[mu]
        Z = np.zeros(natoms)
        for atom, elem, n, l in {(ao.atom, ao.elem, ao.n, ao.l) for ao in self.basis}:
            Z[atom] += self.occupied[elem][n][l]
        return [Z[a] - q[a] for a in range(natoms)]

            


        




