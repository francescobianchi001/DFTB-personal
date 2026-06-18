#! /usr/bin/env  python3

import numpy as np
from pathlib import Path
import matplotlib.pyplot as pl
from scipy.integrate import simpson
from Y_real import Y_real

class slaterkonster:

    def __init__(self,grid=None):

        p = Path.cwd()/'ATOMS_BS'

        atoms = []

        for entry in p.iterdir():
            if entry.is_file():
                atom = np.load(entry,allow_pickle=True)
                atoms.append(atom)

        self.r = [data['grid'] for data in atoms]
        self.occupied = [data['occupied'].tolist() for data in atoms]
        self.basisets =  [data['wavefunctions'].tolist() for data in atoms]

        if grid!=None:
            self.r = self.r[:grid]
            self.occupied = self.occupied
            self.basisets = [[[self.basisets[i][j][k][:grid]
                               for k in range(len(self.basisets[i][j]))]
                              for j in range(len(self.basisets[i]))]
                             for i in range(len(self.basisets))]

    def SK(self, BS_1, BS_2, d_AB):
        """Two-center overlaps between basis sets BS_1 (atom A) and BS_2 (atom B),
        separated by d_AB along z.

        Data layout (confirmed): BS[shell][l] -> radial u(r) = r*R(r) on self.r.
        So the angular momentum IS the inner index l.  For a bond there is a single
        shared magnetic number (the phi selection rule m_A = m_B), so we loop one
        bond channel o = 0..min(lA,lB):  0=sigma, 1=pi, 2=delta.
        """
        S = {}

        # distances from each centre to every grid point (atom A at z=0, B at z=d_AB)
        rA = np.sqrt(self.X**2 + self.Z**2)
        rB = np.sqrt(self.X**2 + (self.Z - d_AB)**2)
        grid_A, grid_B = self.r[self.i], self.r[self.j]

        for nA in range(len(BS_1)):                  # principal shell, atom A
            for lA in range(len(BS_1[nA])):          # angular momentum, atom A
                for nB in range(len(BS_2)):          # principal shell, atom B
                    for lB in range(len(BS_2[nB])):  # angular momentum, atom B
                        for o in range(min(lA, lB) + 1):     # bond: sigma/pi/delta
                            AW = Y_real.angular_weights[(lA, lB, o)]   # callable W(x,z,R)
                            uA = np.asarray(BS_1[nA][lA])    # u_A = r*R_A on grid_A
                            uB = np.asarray(BS_2[nB][lB])    # u_B = r*R_B on grid_B

                            # --- Wire A: R = u/r interpolated onto r_A, r_B (0 past grid)
                            R_A = np.interp(rA.ravel(), grid_A, uA/grid_A,
                                            right=0.0).reshape(rA.shape)
                            R_B = np.interp(rB.ravel(), grid_B, uB/grid_B,
                                            right=0.0).reshape(rB.shape)

                            # --- Wire B: integrand R_A R_B W(x,z,R) * x , then 2D Simpson
                            integrand = R_A * R_B * AW(self.X, self.Z, d_AB) * self.X
                            S_x = simpson(integrand, x=self.x, axis=0)   # over x (rho)
                            value = simpson(S_x, x=self.z)               # over z

                            S[(nA, lA, nB, lB, o)] = value
        return S



    def build_grid(self, d_AB, N=400):
        ### cylindrical integration grid: x = rho (>=0), z along the bond axis ###
        rmax = self.r[0][-1]                           # outer radial extent (~15 Bohr)
        self.d_AB = d_AB
        self.x = np.linspace(1e-4, rmax, N)            # cylindrical radius (>= 0)
        self.z = np.linspace(-rmax, d_AB + rmax, N)    # axial coord spanning both atoms
        self.X, self.Z = np.meshgrid(self.x, self.z, indexing='ij')

    def Space_definment(self):
        # placeholder internuclear distance along z (we'll optimise R later)
        R_tot = self.r[0][-1] + self.r[1]
        d_AB = R_tot[-1]
        self.build_grid(d_AB)

        S = {}
        for i in range(len(self.basisets)):
            for j in range(i + 1):
                self.i = i
                self.j = j
                S[(i, j)] = self.SK(self.basisets[i], self.basisets[j], d_AB)
        return S


if __name__ == '__main__':
    sk = slaterkonster()

    # carbon = the atom with the most basis functions (1s,2s,2p); H has only 1s
    carbon = max(range(len(sk.basisets)),
                 key=lambda a: sum(len(s) for s in sk.basisets[a]))

    # --- sanity check at d = 0: self-overlaps must reproduce the normalization ---
    sk.i = sk.j = carbon
    sk.build_grid(0.0)
    S0 = sk.SK(sk.basisets[carbon], sk.basisets[carbon], 0.0)
    print('sanity @ d = 0 (carbon with itself):')
    print(f'  1s-1s sigma = {S0[(0, 0, 0, 0, 0)]:.4f}   (expect ~1, normalization)')
    print(f'  2s-2s sigma = {S0[(1, 0, 1, 0, 0)]:.4f}   (expect ~1, normalization)')
    print(f'  1s-2s sigma = {S0[(0, 0, 1, 0, 0)]:.4f}   (expect ~0, orthogonality)')

    # --- full run at the placeholder distance ---
    S = sk.Space_definment()
    print(f'\nfull run @ d_AB = {sk.d_AB:.2f} Bohr -- {len(S)} block(s)')
    cc = S[(carbon, carbon)]
    print(f'  1s-1s sigma (carbon-carbon) = {cc[(0, 0, 0, 0, 0)]:.3e}')
