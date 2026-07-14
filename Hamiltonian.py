#!/usr/bin/python3

import numpy as np
from pathlib import Path
import matplotlib.pyplot as pl
from scipy.integrate import simpson
import subprocess as sub
from collections import namedtuple

# The real spherical harmonics / SK angular weights are generated once into
# Y_real.py. Regenerate only if it is missing (the expressions never change),
# before anything imports it. Doing this at module load keeps every run quiet.
if not Path("Y_real.py").exists():
    sub.run(["SK/./Spherical_Harmonics.py"])

# One basis function (atomic orbital). Its position in the flat basis list is
# its row/column index in H and S, so no separate index map is needed.
AO = namedtuple('AO', 'atom n l m u grid')


class H:

    def __init__(self,distance,frozen_core=True,grid=None):

        p_bs = Path.cwd()/'ATOMS_BS'
        p_pot = Path.cwd()/'ATOMS_POT'

        atoms = []
        self.names = []          # element label per atom (file stem)
        self.Znum = []           # atomic number (self.Z is taken by the z-grid)

        for entry in p_bs.iterdir():
            if entry.is_file():
                atom = np.load(entry,allow_pickle=True)
                atoms.append(atom)
                self.names.append(entry.stem)
                self.Znum.append(int(atom['Z']))

        V = []

        for entry in p_pot.iterdir():
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

        pth = Path.cwd()/'eig_neutral'

        eig_neutral = []
        for entry in pth.iterdir():
            if entry.is_file():
                eig_neutral.append(np.load(entry,allow_pickle=True))
        self.eigN = [data['eigenvalues'].tolist() for data in eig_neutral]

        if grid!=None:
            self.r = self.r[:grid]
            self.occupied = self.occupied
            self.basisets = [[[self.basisets[i][j][k][:grid]
                               for k in range(len(self.basisets[i][j]))]
                              for j in range(len(self.basisets[i]))]
                             for i in range(len(self.basisets))]

        self.p = Path.cwd()
        self.distance = distance
        self.frozen_core = frozen_core

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

    def build_grid(self,d_AB,N=400):
        rmax = self.r[0][-1]
        r0 = self.r[0][0]
        self.x = np.linspace(r0,rmax,N)
        # z symmetric about the bond midpoint d/2 so the reflection z -> d-z maps
        # the node set onto itself (needed for the Vint reflection-averaging in
        # fill_element to be exact).
        self.z = np.linspace(-rmax, d_AB + rmax, N)
        X,Z = np.meshgrid(self.x,self.z,indexing='ij')
        return X,Z

    def build_basis(self):
        LMAX = 2                                              # SK integrals stop at d
        self.basis = []
        self.nelec = 0
        for a in range(len(self.basisets)):
            if self.frozen_core:
                val = max(n for n in range(len(self.basisets[a]))
                          if any(self.occupied[a][n]))
                shells = range(val, len(self.basisets[a]))
            else:
                shells = range(len(self.basisets[a]))
            for n in shells:
                for l in range(len(self.basisets[a][n])):
                    if l > LMAX:
                        continue
                    u = np.asarray(self.basisets[a][n][l])
                    if not np.any(u):                    # zeroed (VO-keep) / absent shell
                        continue
                    self.nelec += self.occupied[a][n][l]
                    for m in range(-l, l + 1):
                        self.basis.append(AO(a, n, l, m, u, self.r[a]))
        self.N = len(self.basis)
        return self.N

    def fill_element(self, mu, nu):
        from Y_real import Y_real
        A, B = self.basis[mu], self.basis[nu]

        if A.atom == B.atom:
            if mu == nu:
                norm = simpson(A.u**2, x=A.grid)             # <phi|phi> ~ 1
                if abs(1 - norm) >= 1e-3:
                    raise ValueError(
                        f"AO {mu} (atom {A.atom}, n{A.n} l{A.l} m{A.m}) "
                        f"not normalized: <phi|phi>={norm}")
                if (A.n, A.l) in self.vo_shells[A.atom]:
                    # VO on-site (any s/p/d virtual, NOT the valence): Rayleigh
                    # quotient of the CONFINED orbital against the FREE hamiltonian.
                    # Sits between eigN (variational min -> too deep) and the
                    # confined eps (wall energy -> too high). 1D radial integral,
                    # <u|u>=1; kinetic by parts (u=0 at both ends kills the
                    # boundary term). Veff (saved) = physical + valence Vconf, so
                    # physical = Veff - Vconf regardless of this orbital's own wall.
                    r, u = A.grid, A.u
                    Vphys = self.Veff[A.atom] - self.Vconf[A.atom]
                    up = np.gradient(u, r)
                    centrifugal = A.l * (A.l + 1) / (2.0 * r**2)
                    E_ray = simpson(0.5 * up**2 + (centrifugal + Vphys) * u**2, x=r)
                    # bracket check: neutral level (too deep) < Rayleigh < confined eps (too high)
                    eig_neutral = self.eigN[A.atom][A.n][A.l]
                    eps_conf    = self.eigenvalues[A.atom][A.n][A.l]
                    if A.m == -A.l:
                        print(f"[VO bracket] atom {A.atom} ({self.names[A.atom]}) "
                              f"n{A.n} l{A.l}: eigN={eig_neutral:+.6f} < "
                              f"E_ray={E_ray:+.6f} < eps_conf={eps_conf:+.6f}")
                    if not (eig_neutral <= E_ray <= eps_conf):
                        print(f"WARNING: VO Rayleigh out of bracket for atom {A.atom} "
                              f"n{A.n} l{A.l}: eigN={eig_neutral:+.6f}, E_ray={E_ray:+.6f}, "
                              f"eps_conf={eps_conf:+.6f}")
                    self.Sij[mu, mu] = 1.0
                    self.H[mu, mu]   = E_ray
                    return

                self.Sij[mu, mu] = 1.0
                self.H[mu, mu]   = self.eigN[A.atom][A.n][A.l]   # neutral on-site level
            return                                              # off-diag same-atom = 0

        if A.m != B.m:
            return                                              # -> 0 by symmetry
        o = abs(A.m)                                            # sigma/pi/delta channel
        d = self.distance
        X = self.X

        # Symmetric two-center H. The eps*S trick eliminates the kinetic energy
        # via EITHER atom's confined eigen-equation, giving two exact but
        # numerically different forms. Using only A's breaks the A<->B
        # permutation symmetry (eps_a*S != eps_b*S when the two orbitals differ,
        # worst for the diffuse d whose VO-confined eps is far from the rest),
        # leaking spurious Mulliken charge onto homonuclear atoms. Average both:
        #   H_ab = 1/2 (eps_a+eps_b) S + 1/2 <a| Va_phys+Vb_phys - Wa - Wb |b>
        # W_X = wall X was solved in (VO for a re-confined virtual, valence
        # otherwise); Vx_phys = Veff_X - Vconf_X (valence) is the physical potential.
        wall_A = self.Vconf[A.atom]
        if (A.n, A.l) in self.vo_shells[A.atom] and self.Vconf_VO[A.atom] is not None:
            wall_A = self.Vconf_VO[A.atom]
        wall_B = self.Vconf[B.atom]
        if (B.n, B.l) in self.vo_shells[B.atom] and self.Vconf_VO[B.atom] is not None:
            wall_B = self.Vconf_VO[B.atom]
        VphysA = self.Veff[A.atom] - self.Vconf[A.atom]
        VphysB = self.Veff[B.atom] - self.Vconf[B.atom]
        AW = Y_real.angular_weights[(A.l, B.l, o)]

        # S_SK and Vint integrands, sampled at bond-axis coordinate Zc.
        def integrand(Zc):
            rA = np.sqrt(X**2 + Zc**2)
            rB = np.sqrt(X**2 + (Zc - d)**2)
            RA = np.interp(rA.ravel(), A.grid, A.u / A.grid, right=0.0).reshape(rA.shape)
            RB = np.interp(rB.ravel(), B.grid, B.u / B.grid, right=0.0).reshape(rB.shape)
            VJ = 0.5 * (np.interp(rA.ravel(), A.grid, VphysA, right=0.0).reshape(rA.shape)
                        + np.interp(rB.ravel(), B.grid, VphysB, right=0.0).reshape(rB.shape)
                        - np.interp(rA.ravel(), A.grid, wall_A, right=0.0).reshape(rA.shape)
                        - np.interp(rB.ravel(), B.grid, wall_B, right=0.0).reshape(rB.shape))
            b = RA * RB * AW(X, Zc, d) * X                     # X = rho Jacobian
            return b, b * VJ

        # Reflection-average over z -> d-z. Analytically the integral is invariant
        # (the substitution swaps A<->B); numerically it symmetrizes the quadrature
        # so the potential-weighted Vint is A<->B symmetric, not just S.
        bS1, bV1 = integrand(self.Z)
        bS2, bV2 = integrand(d - self.Z)
        baseS = 0.5 * (bS1 + bS2)
        baseV = 0.5 * (bV1 + bV2)
        S_SK = simpson(simpson(baseS, x=self.x, axis=0), x=self.z)
        Vint = simpson(simpson(baseV, x=self.x, axis=0), x=self.z)

        eps_a = self.eigenvalues[A.atom][A.n][A.l]             # confined eps for the eps*S trick
        eps_b = self.eigenvalues[B.atom][B.n][B.l]
        self.Sij[mu, nu] = self.Sij[nu, mu] = S_SK
        self.H[mu, nu]   = self.H[nu, mu]   = 0.5 * (eps_a + eps_b) * S_SK + Vint

    def H_matrix(self):
        if not Path("Y_real.py").exists():            # cache: generate only once
            sub.run(["SK/./Spherical_Harmonics.py"])
        self.X,self.Z = self.build_grid(self.distance)

        self.build_basis()
        self.H = np.zeros((self.N,self.N))
        self.Sij = np.zeros((self.N,self.N))

        for mu in range(self.N):
            for nu in range(mu + 1):       # lower triangle + diagonal
                self.fill_element(mu, nu)
        return self.H

    def diag(self, thresh=1e-6):
        H = 0.5 * (self.H + self.H.T)
        S = 0.5 * (self.Sij + self.Sij.T)

        s, U = np.linalg.eigh(S)
        keep = s > thresh
        if not np.all(keep):
            print(f"diag: dropping {np.sum(~keep)} of {len(s)} basis modes "
                  f"(min S eig {s.min():.2e} < {thresh:.0e}) — basis near-singular")
        U, s = U[:, keep], s[keep]

        X = U * (1.0 / np.sqrt(s))
        H_ = X.T @ H @ X

        E, C_ = np.linalg.eigh(H_)
        C = X @ C_

        assert np.allclose(C.T @ S @ C, np.eye(C.shape[1]), atol=1e-8)
        assert np.allclose(H @ C, S @ C @ np.diag(E), atol=1e-6)
        
        self.S= S

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
        natoms = len(self.occupied)
        q = np.zeros(natoms)
        for mu, ao in enumerate(self.basis):
            q[ao.atom] += gross[mu]
        Z = np.zeros(natoms)
        for atom, n, l in {(ao.atom, ao.n, ao.l) for ao in self.basis}:
            Z[atom] += self.occupied[atom][n][l]
        return [Z[a] - q[a] for a in range(natoms)]


