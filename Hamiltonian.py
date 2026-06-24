#!/usr/bin/python3

import numpy as np
from pathlib import Path
import matplotlib.pyplot as pl
from scipy.integrate import simpson
import subprocess as sub

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

    def getH_ij(self,BS_A,BS_B,X,Z):
        from Y_real import Y_real
        d_AB= self.distance
        rA = np.sqrt(X**2+Z**2)
        rB = np.sqrt(X**2+(Z-d_AB)**2)
        grid_A, grid_B = self.r[self.i], self.r[self.j]
        Veff_j  = np.interp(rB.ravel(), grid_B, self.Veff[self.j],
                            right=0.0).reshape(rB.shape)
        Vconf_j = np.interp(rB.ravel(), grid_B, self.Vconf[self.j],
                            right=0.0).reshape(rB.shape)
        Vconf_i = np.interp(rA.ravel(), grid_A, self.Vconf[self.i],
                            right=0.0).reshape(rA.shape)
        
        VJ_eff = Veff_j - Vconf_j - Vconf_i
        
        if self.j==self.i:
            # on-site (non-SCC DFTB): diagonal = atomic orbital eigenvalue, no integral
            for n in range(len(BS_A)):
                for l in range(len(BS_A[n])):
                    g = self.idx[self.i][n][l]
                    eps = self.eigenvalues[self.i][n][l]
                    u = np.asarray(BS_A[n][l])
                    norm = simpson(u**2, x=grid_A)     # <phi|phi>, should be ~1
                    if abs(1 - norm) < 1e-3:
                        self.Sij[g,g] = 1.0
                    else:
                        raise ValueError(
                            f"orbital ({self.i},{n},{l}) not normalized: <phi|phi>={norm}")
                    self.H[g,g] = eps
            return
         
        for na in range(len(BS_A)):
            for nb in range(len(BS_B)):
                for la in range(len(BS_A[na])):
                    for lb in range(len(BS_B[nb])):
                        gA = self.idx[self.i][na][la]
                        gB = self.idx[self.j][nb][lb]
                        eps_a = self.eigenvalues[self.i][na][la]
                        for o in range(min(la, lb) + 1):
                            AW = Y_real.angular_weights[(la, lb, o)]
                            uA = np.asarray(BS_A[na][la])
                            uB = np.asarray(BS_B[nb][lb])

                            R_A = np.interp(rA.ravel(), grid_A, uA/grid_A,
                                            right=0.0).reshape(rA.shape)
                            R_B = np.interp(rB.ravel(), grid_B, uB/grid_B,
                                            right=0.0).reshape(rB.shape)

                            integrand_SK = R_A * R_B * AW(X,Z,d_AB) * self.X
                            integrand = integrand_SK * VJ_eff
                            S_SK_x = simpson(integrand_SK,x=self.x,axis=0)
                            S_SK = simpson(S_SK_x, x=self.z)
                            S_x = simpson(integrand,x=self.x,axis=0)
                            S = simpson(S_x,x=self.z)
                            self.Sij[gA,gB] += S_SK
                            self.H[gA,gB] += eps_a * S_SK + S
                        self.Sij[gB,gA] = self.Sij[gA,gB]
                        self.H[gB,gA] = self.H[gA,gB]
                            
    def build_index_map(self):
        # Assign every (atom, shell n, orbital l) channel a unique global
        # row/column index in H, walking the atoms in order.
        self.idx = []
        counter = 0
        for atom in self.basisets:
            atom_map = []
            for shell in atom:
                shell_map = []
                for l in range(len(shell)):
                    shell_map.append(counter)
                    counter += 1
                atom_map.append(shell_map)
            self.idx.append(atom_map)
        self.N = counter
        return self.N

    def H_matrix(self):
        sub.run(["SK/./Spherical_Harmonics.py"])
        self.X,self.Z = self.build_grid(self.distance)

        self.build_index_map()
        self.H = np.zeros((self.N,self.N))
        self.Sij = np.zeros((self.N,self.N))

        for i in range(len(self.names)):
            for j in range(i+1):
                self.i = i
                self.j = j
                self.getH_ij(self.basisets[i], self.basisets[j],self.X,self.Z)
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




        


        




















