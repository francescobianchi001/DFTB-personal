#! /usr/bin/python3

import numpy as np
from Hamiltonian import H, Vrep_fit
from pathlib import Path

p = Path.cwd() / 'DIALECT'
p = p / 'CH.xyz'
init = H(geom=p, typor0=True)

H0 = init.H_matrix()

Ec,E,C = init.E_tot()

P = init.P

#print(C[:,:1])

#print(P,C, init.Sij,P@init.Sij)
print('eigenvalues:',E)
print('P:',P)
print('C:', C)

##mulliken charges:
Q = init.Mulliken_charge()
print('MkC:',Q)

fit = Vrep_fit(typor0=True, Rcut=3.0236)
Vrep = fit.integration()

d = init.dist
m = ((init.atoms[:,None]-init.atoms[None,:] != 0)
     & np.triu(np.ones_like(d,bool),1) & (d < fit.Rcut))
Erep = float(Vrep(d[m]).sum())

print('E_rep:',Erep)
print('E_tot:',Ec+Erep)
