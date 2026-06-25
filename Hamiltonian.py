#!/usr/bin/python3

import numpy as np
from pathlib import Path
import matplotlib.pyplot as pl
from scipy.integrate import simpson
import subprocess as sub
from collections import namedtuple

# One basis function (atomic orbital). Its position in the flat basis list is
# its row/column index in H and S, so no separate index map is needed.
AO = namedtuple('AO', 'atom n l m u grid')


class H:

    def __init__(self,distance,grid=None):

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
        zmin = r0- rmax
        zmax= r0 + rmax + d_AB
        self.x = np.linspace(r0,rmax,N)
        self.z = np.linspace(zmin,zmax,N)
        X,Z = np.meshgrid(self.x,self.z,indexing='ij')
        return X,Z

    def build_basis(self):
        # Flat list of atomic orbitals, one entry per (atom, n, l, m). Its index
        # is the H/S row/column, so a separate index map is no longer needed.
        self.basis = []
        for a in range(len(self.basisets)):
            for n in range(len(self.basisets[a])):
                for l in range(len(self.basisets[a][n])):
                    u = np.asarray(self.basisets[a][n][l])   # confined radial (basis shape)
                    for m in range(-l, l + 1):               # l=0 -> just m=0
                        self.basis.append(AO(a, n, l, m, u, self.r[a]))
        self.N = len(self.basis)
        return self.N

    def fill_element(self, mu, nu):
        from Y_real import Y_real
        A, B = self.basis[mu], self.basis[nu]

        # --- same atom: on-site block is DIAGONAL only --------------------
        if A.atom == B.atom:
            if mu == nu:
                norm = simpson(A.u**2, x=A.grid)             # <phi|phi> ~ 1
                if abs(1 - norm) >= 1e-3:
                    raise ValueError(
                        f"AO {mu} (atom {A.atom}, n{A.n} l{A.l} m{A.m}) "
                        f"not normalized: <phi|phi>={norm}")
                self.Sij[mu, mu] = 1.0
                self.H[mu, mu]   = self.eigN[A.atom][A.n][A.l]   # neutral on-site level
            return                                              # off-diag same-atom = 0

        # --- different atoms: two-center, selection rule m_A == m_B -------
        if A.m != B.m:
            return                                              # -> 0 by symmetry
        o = abs(A.m)                                            # sigma/pi/delta channel
        d = self.distance
        X, Z = self.X, self.Z
        rA = np.sqrt(X**2 + Z**2)
        rB = np.sqrt(X**2 + (Z - d)**2)

        Veff_B  = np.interp(rB.ravel(), B.grid, self.Veff[B.atom],  right=0.0).reshape(rB.shape)
        Vconf_B = np.interp(rB.ravel(), B.grid, self.Vconf[B.atom], right=0.0).reshape(rB.shape)
        Vconf_A = np.interp(rA.ravel(), A.grid, self.Vconf[A.atom], right=0.0).reshape(rA.shape)
        VJ = Veff_B - Vconf_B - Vconf_A

        RA = np.interp(rA.ravel(), A.grid, A.u / A.grid, right=0.0).reshape(rA.shape)
        RB = np.interp(rB.ravel(), B.grid, B.u / B.grid, right=0.0).reshape(rB.shape)
        AW = Y_real.angular_weights[(A.l, B.l, o)]

        base = RA * RB * AW(X, Z, d) * self.X                  # self.X = rho Jacobian
        S_SK = simpson(simpson(base,      x=self.x, axis=0), x=self.z)
        Vint = simpson(simpson(base * VJ, x=self.x, axis=0), x=self.z)

        eps_a = self.eigenvalues[A.atom][A.n][A.l]             # confined eps for the eps*S trick
        self.Sij[mu, nu] = self.Sij[nu, mu] = S_SK
        self.H[mu, nu]   = self.H[nu, mu]   = eps_a * S_SK + Vint

    def H_matrix(self):
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

        return E, C
