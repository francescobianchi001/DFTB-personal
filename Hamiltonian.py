#!/usr/bin/python3

import json
import numpy as np
import re
import sys
from pathlib import Path
import matplotlib.pyplot as pl
from scipy.integrate import simpson
from scipy import special
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

Ubbard = {'C':0.376, 'H': 0.395,          # Koskinen sec.V D (adjusted)
          'O':0.4468, 'N': 0.530}         # O = IE-EA; N unbound anion -> Hotbit value

def read_radii(path):
    radii = {}
    for lineno, raw in enumerate(Path(path).read_text().splitlines(), 1):
        fields = raw.split('#', 1)[0].split()
        if not fields:
            continue
        if len(fields) != 2:
            raise ValueError(f"{path}:{lineno}: expected '<symbol> <r0>', got {raw!r}")
        radii[fields[0].capitalize()] = float(fields[1])
    return radii


def prepare_atoms(geom, vo=None, lb94=None, r0_vo=None, r0=None, typor0=None):
    atno, coords = get_coords(geom, maxlen=100)
    atoms = {ATOM_NAMES[int(Z)].capitalize(): int(Z) for Z in atno}

    p_bs = Path.cwd() / 'ATOMS_BS'
    manifest = Path.cwd() / 'atoms_provenance.json'

    radii = None
    if typor0:
        radii = read_radii(Path.cwd() / 'radi.txt' if typor0 is True else typor0)

    want = {'VO': vo, 'r0': r0, 'r0_VO': r0_vo,
            'lb94': bool(lb94) if lb94 is not None else (vo is not None)}
    if radii:
        want['typor0'] = radii

    prov = None
    if manifest.exists():
        try:
            prov = json.loads(manifest.read_text()).get('settings')
        except (ValueError, OSError):
            prov = None

    on_disk = {p.stem for p in p_bs.glob('*.npz')} if p_bs.is_dir() else set()
    missing = [sym for sym in atoms if sym not in on_disk]
    stale = prov is not None and prov != want

    if prov is None and on_disk:
        print(f"prepare_atoms: {sorted(on_disk)} on disk with no provenance record "
              f"-- recording them as {want}. Delete atoms_provenance.json / the "
              f"ATOMS_* dirs if that is wrong.")
        manifest.write_text(json.dumps(
            {'settings': want, 'elements': sorted(on_disk)}, indent=2))
    if not stale and not missing:
        return atoms
    if stale:
        print(f"prepare_atoms: stored atoms were solved with {prov}, this run wants "
              f"{want} -> ALL elements will be recomputed")
    elif missing:
        print(f"prepare_atoms: missing basis data for {missing} -> solving just those "
              f"(keeping {sorted(on_disk)})")

    keep = {sym: ATOM_NAMES.index(sym.lower()) for sym in on_disk
            if sym.lower() in ATOM_NAMES}
    init_path = Path.cwd() / 'INIT.py'
    text = init_path.read_text()
    block = 'ATOMS = {\n' + ''.join(
        f'    "{sym}": {Z},\n' for sym, Z in sorted({**keep, **atoms}.items())) + '}'
    text = re.sub(r'ATOMS = \{.*?\}', block, text, count=1, flags=re.DOTALL)
    init_path.write_text(text)

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
    if radii:
        extra += ['--typor0', json.dumps(radii)]   # argv is strings only

    sub.run([sys.executable, str(init_path), *extra], check=True)
    return atoms

AO = namedtuple('AO', 'atom elem n l m u grid d R')


class H:

    def __init__(self,distance=None,frozen_core=True,grid=None,geom='geometry.xyz',
                 vo=None,lb94=None,r0_vo=None,r0=None,typor0=None):

        prepare_atoms(geom, vo=vo, lb94=lb94, r0_vo=r0_vo, r0=r0, typor0=typor0)
        atom,coords = get_coords(geom, maxlen=100)

        p_bs = Path.cwd()/'ATOMS_BS'
        p_pot = Path.cwd()/'ATOMS_POT'

        atoms = []
        self.names = []       
        self.Znum = []         

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
        self.Vconf_VO = [data['Vconf_VO'] if 'Vconf_VO' in data.files else None
                         for data in V]
        self.vo_shells = [set(map(tuple, data['vo_shells'].tolist()))
                          if 'vo_shells' in data.files else set() for data in V]
        
        self.atoms, self.coords = atom,coords         # atom = atomic numbers (natoms,)
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
        self._gkey = self._gval = None    
        self._diag_key = None           
        self.E = self.C = self.f = self.P = self.Eband = None
        self.frozen_core = frozen_core
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

    def build_grid(self,d,N=400):
        rmax = self.r[0][-1]
        r0 = self.r[0][0]
        x = np.linspace(r0,rmax,N)
        z = np.linspace(-rmax, d + rmax, N)
        X,Z = np.meshgrid(x,z,indexing='ij')
        return X,Z,x,z

    def _grid(self,d,N=400):
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
            #VJ = (np.interp(rA.ravel(), A.grid, Vn_A, right=0.0).reshape(rA.shape)
            #      + np.interp(rB.ravel(), B.grid, Vn_B, right=0.0).reshape(rB.shape)
            #      - 0.5 * (np.interp(rA.ravel(), A.grid, VphysA, right=0.0).reshape(rA.shape)
            #               + np.interp(rB.ravel(), B.grid, VphysB, right=0.0).reshape(rB.shape)
            #               + np.interp(rA.ravel(), A.grid, wall_A, right=0.0).reshape(rA.shape)
            #               + np.interp(rB.ravel(), B.grid, wall_B, right=0.0).reshape(rB.shape)))
            VJ = 0.5 * (np.interp(rA.ravel(), A.grid, VphysA, right=0.0).reshape(rA.shape)
                        + np.interp(rB.ravel(), B.grid, VphysB, right=0.0).reshape(rB.shape)
                        - np.interp(rA.ravel(), A.grid, wall_A, right=0.0).reshape(rA.shape)
                        - np.interp(rB.ravel(), B.grid, wall_B, right=0.0).reshape(rB.shape))
            b = RA * RB * AW(X, Zc, d) * X # X = rho Jacobian
            return b, b * VJ

        bS1, bV1 = integrand(Zg)
        bS2, bV2 = integrand(d - Zg)
        baseS = 0.5 * (bS1 + bS2)
        baseV = 0.5 * (bV1 + bV2)
        S_o = simpson(simpson(baseS, x=xg, axis=0), x=zg)
        V_o = simpson(simpson(baseV, x=xg, axis=0), x=zg)
        return S_o, V_o

    ONSITE_CONSISTENT = False

    ONSITE_MODE = 'epsS'

    def _radial_S(self, A, B):
        return simpson(A.u * B.u, x=A.grid)

    def _radial_H(self, A, B):
        r = A.grid
        Vphys = self.Veff[A.elem] - self.Vconf[A.elem]
        centrifugal = A.l * (A.l + 1) / (2.0 * r**2)
        return simpson(0.5 * np.gradient(A.u, r) * np.gradient(B.u, r)
                       + (centrifugal + Vphys) * A.u * B.u, x=r)

    def fill_onsite(self, A, B, mu, nu):
        if A.l != B.l or A.m != B.m:                 # orthogonal by the angular part
            return
        S_ab = self._radial_S(A, B)
        if self.ONSITE_MODE == 'radial':
            H_ab = self._radial_H(A, B)
        else:
            H_ab = 0.5 * (self.H[mu, mu] + self.H[nu, nu]) * S_ab
        self.S[mu, nu] = self.S[nu, mu] = S_ab
        self.H[mu, nu] = self.H[nu, mu] = H_ab

    def fill_diag(self, A,mu):
        norm = simpson(A.u**2, x=A.grid)             # <phi|phi> ~ 1
        if abs(1 - norm) >= 1e-3:
            raise ValueError(
                f"AO {mu} (atom {A.atom}, n{A.n} l{A.l} m{A.m}) "
                f"not normalized: <phi|phi>={norm}")
        if (A.n, A.l) in self.vo_shells[A.elem]:
            E_ray = self._radial_H(A, A)
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
        self.H[mu, mu] = (self._radial_H(A, A) if self.ONSITE_CONSISTENT
                          else self.eigN[A.elem][A.n][A.l])   # neutral on-site level

    def fill_cross(self, A, B, mu, nu):
        from Y_real import Y_real
        l1, l2 = A.l, B.l
        d = A.d[B.atom]
        Lc, Mc, Nc = (B.R - A.R) / d                    # bond direction cosines A -> B

        key = (A.atom, A.n, l1, B.atom, B.n, l2)
        if key not in self._vcache:
            self._vcache[key] = {o: self.cross_terms(A, B, o)
                                 for o in range(min(l1, l2) + 1)}
        chan = self._vcache[key]

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
        self._diag_key = None                          # stale diag results
        self.E = self.C = self.f = self.P = self.Eband = None
        self.space()                                   # basis + distance matrix + center
        self.H = np.zeros((self.N, self.N))
        self.S = np.zeros((self.N, self.N))
        self._vcache = {}                              # bond-frame V_{ll'm} per shell pair
        for mu, A in enumerate(self.basis):
            self.fill_diag(A, mu)
        self._gkey = self._gval = None                 # last grid built (see _grid)

        for mu in range(self.N):
            for nu in range(mu + 1):       # lower triangle + diagonal
                A, B = self.basis[mu], self.basis[nu]
                if A.atom == B.atom:
                    if mu != nu:
                        self.fill_onsite(A, B, mu, nu)
                else:
                    self.fill_cross(A, B, mu, nu)
        self.H0 = self.H.copy()                        # SCC overwrites self.H
        return self.H

    def occupations(self, E, tol=1e-5):
        f = np.zeros(len(E))
        left = float(self.nelec)
        k = 0
        while left > 1e-12 and k < len(E):
            g = k
            while g + 1 < len(E) and E[g + 1] - E[k] < tol:
                g += 1
            n = g - k + 1
            take = min(2.0 * n, left)
            f[k:g + 1] = take / n
            left -= take
            k = g + 1
        return f

    def diag(self, thresh=1e-6,H=None):
        if H is not None:
            self.H = H                                 # keep self.H in step: Mulliken_charge() re-diags it
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

        self.E, self.C = E, C
        self.f = self.occupations(E)                   # per-level occupation, sums to nelec
        self.P = (C * self.f) @ C.T
        self.Eband = float(self.f @ E)

        return E, C

    def band_energy(self):
        self.diag()
        return self.Eband

    def density_matrix(self):
        self.diag()
        return self.P

    def Mulliken_charge(self):
        self.diag()
        gross = np.diag(self.P @ self.S)
        natoms = len(self.atoms)
        q = np.zeros(natoms)
        for mu, ao in enumerate(self.basis):
            q[ao.atom] += gross[mu]
        Z = np.zeros(natoms)
        for atom, elem, n, l in {(ao.atom, ao.elem, ao.n, ao.l) for ao in self.basis}:
            Z[atom] += self.occupied[elem][n][l]
        return [Z[a] - q[a] for a in range(natoms)]

    def SCC(self,H0,y,h1old=None,alpha=0.3):
        def merge(old,new):
            return old*(1-alpha) + new*alpha
        Dq = -np.asarray(self.Mulliken_charge())     
        DQ = Dq[None,:]*Dq[:,None]
        Ec = 1/2 * sum(sum(DQ*y))
        Ecoul = 1/2*Dq@y@Dq
        
        e = np.array([y[i,:]@Dq for i in range(len(self.atoms))])
        eAO = np.array([e[ao.atom] for ao in self.basis])      
        h1 = 0.5 * self.S * (eAO[:,None] + eAO[None,:])                                  
        if h1old is not None:
            h1 = merge(h1old,h1)
        H = H0 + h1
        E,C = self.diag(H=H)

        return Ecoul,E,C,Dq,h1

    def SCF(self,tresh=1e-5,N=400,alpha=0.3):
        R = self.dist
        U = np.array([Ubbard[self.names[self.Z2elem[Z]]] for Z in self.atoms])
        FWHM = 1.329/U
        y=np.zeros((len(U),len(U)))
        for I in range(len(U)):
            for J in range(len(U)):
                if I==J:
                   y[I,I] = U[I]
                else:
                   CIJ = np.sqrt(4*np.log(2)/((FWHM[I])**2+FWHM[J]**2))
                   yIJ = special.erf(CIJ*R[I][J])/R[I][J]
                   y[I][J]=y[J][I]=yIJ
        H0 = self.H0 
        Ec,E,C,Dq,h1 = self.SCC(H0,y,alpha=alpha)
        for i in range(N):

            Ec_new,E_new,C_new,Dq_new,h1new = self.SCC(H0,y,h1,alpha)
            if abs(Ec_new - Ec) <= tresh and np.max(np.abs(Dq_new-Dq)) <= tresh:
                return Ec_new,E_new,C_new
            
            Ec,E,C,Dq,h1 = Ec_new, E_new,C_new,Dq_new,h1new 
        raise RuntimeError(f"SCF: no convergence in {N} iterations "
                           f"(dEc={abs(Ec_new-Ec):.2e}, dDq={np.max(np.abs(Dq_new-Dq)):.2e}, tresh={tresh:.0e})")

    def E_tot(self):
        Ec,E,C = self.SCF()
        return float(np.sum(self.P * self.H0)) + Ec, E,C  











            


        




