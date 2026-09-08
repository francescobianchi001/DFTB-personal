#!/usr/bin/python3

import json
import numpy as np
import re
import sys
from pathlib import Path
import matplotlib.pyplot as pl
from scipy.integrate import simpson
from scipy.interpolate import CubicSpline, splrep, BSpline
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

Ubbard = {'C':0.365, 'H': 0.420,          # hotbit Vpot/par/{C,H}.elm
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


def prepare_atoms(geom, atno=None, vo=None, lb94=None, r0_vo=None, r0=None, typor0=None):
    if atno is None:
        atno, _ = get_coords(geom, maxlen=100)
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
                 vo=None,lb94=None,r0_vo=None,r0=None,typor0=None, Vrep=False, coords=None,atom=None,
                 charge=0):

        if Vrep and geom is None and coords is not None and atom is not None:
            atom,coords = atom,coords
        else:
            atom,coords = get_coords(geom, maxlen=100)

        prepare_atoms(geom, atno=atom, vo=vo, lb94=lb94, r0_vo=r0_vo,
                      r0=r0, typor0=typor0)

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
        self._skkey = self._skval = None
        self._splcache = {}
        self._diag_key = None           
        self.E = self.C = self.f = self.P = self.Eband = None
        self.frozen_core = frozen_core
        self.charge = charge              # molecular charge; anion = -1
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

    NXI = NETA = 90                                   # Gauss-Legendre nodes per direction

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

    def build_sk_grid(self, d, nxi, neta):
        """Prolate spheroidal (xi, eta) Gauss-Legendre mesh for a pair at 0 and d.

        xi = (rA+rB)/d in [1, xi_max], eta = (rA-rB)/d in [-1, 1]; both nuclei sit
        at CORNERS (xi=1, eta=-+1) where the (xi^2-eta^2) Jacobian vanishes, so the
        -Z/r cusps never fall inside the domain the way they do on a (rho, z) mesh.
        rho drho dz = (d/2)^3 (xi^2 - eta^2) dxi deta.
        """
        rmax = self.r[0][-1]
        xi_max = max(1.0 + 1e-9, 2.0 * rmax / d)
        tx, wx = np.polynomial.legendre.leggauss(nxi)
        te, we = np.polynomial.legendre.leggauss(neta)
        xi = 1.0 + 0.5 * (tx + 1.0) * (xi_max - 1.0)
        XI, ET = np.meshgrid(xi, te, indexing='ij')
        W = np.outer(wx * 0.5 * (xi_max - 1.0), we) * (0.5 * d)**3 * (XI**2 - ET**2)
        rA = 0.5 * d * (XI + ET)
        rB = 0.5 * d * (XI - ET)
        z = 0.5 * d * (1.0 + XI * ET)
        rho = 0.5 * d * np.sqrt(np.clip((XI**2 - 1.0) * (1.0 - ET**2), 0.0, None))
        return rA, rB, rho, z, W

    def _sk_grid(self, d):
        key = (d, self.NXI, self.NETA)
        if self._skkey != key:
            self._skkey = key
            self._skval = self.build_sk_grid(d, self.NXI, self.NETA)
        return self._skval

    def _spline(self, key, x, y):
        """Cached cubic spline; constant below x[0], zero beyond x[-1] (as np.interp)."""
        f = self._splcache.get(key)
        if f is None:
            from scipy.interpolate import CubicSpline
            f = CubicSpline(x, y, extrapolate=False)
            self._splcache[key] = (f, x[0], x[-1], y[0])
            return self._splcache[key]
        return f

    def _ev(self, key, x, y, q):
        f, x0, x1, y0 = self._spline(key, x, y)
        out = f(np.clip(q, x0, x1))
        out = np.nan_to_num(out, nan=0.0)
        return np.where(q > x1, 0.0, out)

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
        self.nelec -= self.charge                         # neutral count + extra electrons
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
        rA, rB, X, zc, W = self._sk_grid(d)    # prolate spheroidal GL mesh for this pair

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

        # S_SK and Vint integrands on the (xi, eta) mesh; rA, rB come from the grid.
        def integrand(Zc):
            RA = self._ev(('R', A.elem, A.n, A.l), A.grid, A.u / A.grid, rA)
            RB = self._ev(('R', B.elem, B.n, B.l), B.grid, B.u / B.grid, rB)
            #VJ = (np.interp(rA.ravel(), A.grid, Vn_A, right=0.0).reshape(rA.shape)
            #      + np.interp(rB.ravel(), B.grid, Vn_B, right=0.0).reshape(rB.shape)
            #      - 0.5 * (np.interp(rA.ravel(), A.grid, VphysA, right=0.0).reshape(rA.shape)
            #               + np.interp(rB.ravel(), B.grid, VphysB, right=0.0).reshape(rB.shape)
            #               + np.interp(rA.ravel(), A.grid, wall_A, right=0.0).reshape(rA.shape)
            #               + np.interp(rB.ravel(), B.grid, wall_B, right=0.0).reshape(rB.shape)))
            VJ = 0.5 * (self._ev(('Vp', A.elem), A.grid, VphysA, rA)
                        + self._ev(('Vp', B.elem), B.grid, VphysB, rB)
                        - self._ev(('W', A.elem, A.n, A.l), A.grid, wall_A, rA)
                        - self._ev(('W', B.elem, B.n, B.l), B.grid, wall_B, rB))
            b = RA * RB * AW(X, Zc, d)     # rho Jacobian is already inside W
            return b, b * VJ

        bS, bV = integrand(zc)
        S_o = float(np.sum(bS * W))
        V_o = float(np.sum(bV * W))
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
            
            print(f"cycle {i}")
            Ec_new,E_new,C_new,Dq_new,h1new = self.SCC(H0,y,h1,alpha)
            if abs(Ec_new - Ec) <= tresh and np.max(np.abs(Dq_new-Dq)) <= tresh:
                return Ec_new,E_new,C_new
            
            Ec,E,C,Dq,h1 = Ec_new, E_new,C_new,Dq_new,h1new 
        raise RuntimeError(f"SCF: no convergence in {N} iterations "
                           f"(dEc={abs(Ec_new-Ec):.2e}, dDq={np.max(np.abs(Dq_new-Dq)):.2e}, tresh={tresh:.0e})")
    
    def E_tot(self,alpha=0.3):
        Ec,E,C = self.SCF(alpha=alpha)
        return float(np.sum(self.P * self.H0)) + Ec, E,C

class Vrep_fit:

    CHARGE = {'CH-': -1}

    def __init__(self,distance=None,frozen_core=True,grid=None,
                 vo=None,lb94=None,r0_vo=None,r0=None,typor0=None, Vrep=False,
                 alpha=0.3,Rcut=3.40,tol=0.0095,omo=None,weight=1.0,s=0.02,
                 dimer=None,other=True):
        mols=self.collect_traj()
        self.Rcut = Rcut
        self.Vpair = self.get_otherVpairs() if other else {}
        self.curves = {}
        self.deriv = []
        self.derivF = []
        self._Hkw = dict(distance=distance,frozen_core=frozen_core,grid=grid,geom=None,
                         vo=vo,lb94=lb94,r0_vo=r0_vo,r0=r0,typor0=typor0,Vrep=True)
        for name, frames in mols.items():
            q = self.CHARGE.get(name,0)

            keep = []          
            for fr in frames:
                d, bond = self.bonds(fr,Rcut,omo)
                N = int(bond.sum())
                if N == 0:
                    continue
                spread = d[bond].max()-d[bond].min()
                if spread > tol:
                    raise ValueError(f"{name} frame {fr.frame}: bonds not equal "
                                     f"(spread {spread:.2e} a0 > tol {tol:.2e})")
                keep.append((fr,d,bond,N))
            if len(keep) < 4:
                continue

            R, Ediff_tot, Vp = [], [], []
            X = np.array([k[0].coords for k in keep])
            for i,(fr,d,bond,N) in enumerate(keep):
                R.append(d[bond].mean())
                Ediff_tot.append((fr.E_DFT - self.E_wr(fr.coords,fr.atoms,q,alpha))/N)

                if fr.F is None:
                    continue
                u = X[min(i+1,len(keep)-1)] - X[max(i-1,0)]
                u /= np.linalg.norm(u)
                g = self.get_g(fr,d,bond)      # grad_X sum_bonds r_IJ ; N*dRds = g.u
                NdRds = float(g.ravel()@u.ravel())
                dEds = self.finite_diff(self.E_wr(fr.coords+s*u,fr.atoms,q,alpha),
                                        self.E_wr(fr.coords-s*u,fr.atoms,q,alpha), s)
                Vp.append(-(float((fr.F*u).sum()) + dEds)/NdRds) 

            R, Ediff_tot = np.array(R), np.array(Ediff_tot)
            o = R.argsort()
            R, Ediff_tot = R[o], Ediff_tot[o]

            E_R = CubicSpline(R, Ediff_tot)
            F_R = E_R(R,1)

            w = weight/np.sqrt(len(R))
            self.curves[name] = (R, Ediff_tot, F_R)
            self.deriv += [(Ri, Fi, w) for Ri, Fi in zip(R, F_R)]
            if len(Vp) == len(R):
                self.derivF += [(Ri, Fi, w) for Ri, Fi in zip(R, np.array(Vp)[o])]

        if dimer:
            self.append_dimer(dimer,alpha=alpha,weight=weight)   # 1.137 Ang, hotbit CH.py

    def get_otherVpairs(self,exclude=(6,1)):
        # hotbit's tab={'CH':...,'rest':'default'}: C-C and H-H repulsion sits in E_wr
        p = Path.cwd()/'Vpot'/'par'/'EurPhysJD_67_38_2013'
        V = {}
        for par in sorted(p.glob('*.par')):
            Z = tuple(sorted(ATOM_NAMES.index(s.lower()) for s in par.stem.split('_')))
            if Z == tuple(sorted(exclude)):
                continue
            rows, on = [], False
            for ln in par.read_text().splitlines():
                if ln.startswith('repulsion='):
                    on = True
                    continue
                if on:
                    fl = ln.split()
                    if len(fl) != 2:
                        break
                    rows.append([float(fl[0]),float(fl[1])])
            t = np.array(rows)
            V[Z] = (t[-1,0], CubicSpline(t[:,0],t[:,1]))
        return V


    def get_g(self,fr,d,bond):
        I,J = np.nonzero(bond)
        nkJ = (fr.coords[I]-fr.coords[J])/d[I,J][:,None]
        g = np.zeros_like(fr.coords)
        np.add.at(g,I,nkJ)
        np.add.at(g,J,-nkJ)
        return g
    
    def bonds(self,fr,Rcut,omo=None,coords=None):
        c = fr.coords if coords is None else coords
        d = np.linalg.norm(c[None,:,:]-c[:,None,:],axis=-1)
        if omo is not None:
            bond =((fr.atoms[None,:] == omo) & (fr.atoms[:,None] == omo)
                        & np.triu(np.ones_like(d,bool),1)
                        & (d < Rcut))
        else:
            bond = ((fr.atoms[None,:]-fr.atoms[:,None] != 0)
                & np.triu(np.ones_like(d,bool),1)
                & (d < Rcut))
        return d, bond

    def E_elec(self,coords,atom,charge,alpha=0.3):
        init = H(**self._Hkw, coords=coords[None], atom=atom, charge=charge)
        init.H_matrix()
        return init.E_tot(alpha)[0]

    def E_other(self,coords,atom):
        d = np.linalg.norm(coords[None,:,:]-coords[:,None,:],axis=-1)
        iu = np.triu_indices(len(atom),1)
        E = 0.0
        for (Za,Zb),(rmax,V) in self.Vpair.items():
            m = (((atom[iu[0]]==Za)&(atom[iu[1]]==Zb))
                 |((atom[iu[0]]==Zb)&(atom[iu[1]]==Za)))
            r = d[iu][m]
            r = r[r < rmax]
            E += float(V(r).sum()) if r.size else 0.0
        return E

    def E_wr(self,coords,atom,charge,alpha=0.3):
        return self.E_elec(coords,atom,charge,alpha) + self.E_other(coords,atom)

    def finite_diff(self,Ep,Em,h):
        return (Ep-Em)/(2*h)

    def append_dimer(self,R,Z=(6,1),q=0,weight=1.0,alpha=0.3,scale=1.025,
                     pools=('deriv','derivF')):
        # E_DFT'(R_eq)=0  ->  V'_rep(R) = -E_elec'(R)/N, N=1
        Z = np.asarray(Z)
        h = (scale-1)*R
        xyz = lambda r: np.array([[0.,0.,0.],[r,0.,0.]])
        dE = self.finite_diff(self.E_wr(xyz(R+h),Z,q,alpha),
                              self.E_wr(xyz(R-h),Z,q,alpha), h)
        for p in pools:
            getattr(self,p).append((R,-dE,weight))
        return R,-dE

    def integration(self,lam=None,k=3,tol=1e-4,pool='deriv'):
        x, y, w = (np.array(a) for a in zip(*getattr(self,pool)))
        o = x.argsort()
        x, y, w = x[o], y[o], w[o]

        # splrep needs strictly increasing x: merge the points that coincide,
        # weighted mean of the slopes, weights add up
        cut = np.r_[0, np.nonzero(np.diff(x) > tol)[0]+1, len(x)]
        g = [slice(a,b) for a,b in zip(cut[:-1],cut[1:])]
        x = np.array([x[i].mean() for i in g])
        y = np.array([np.dot(y[i],w[i])/w[i].sum() for i in g])
        w = np.array([w[i].sum() for i in g])

        m = x < self.Rcut
        x, y, w = x[m], y[m], w[m]

        # anchor Vrep'(Rcut) = 0, weighted so the fit cannot walk away from it
        x = np.append(x,self.Rcut); y = np.append(y,0.0); w = np.append(w,1e3*w.max())

        if lam is None:
            lam = len(x) - np.sqrt(2*len(x))
        self.tck = splrep(x, y, w, s=lam, k=min(k,len(x)-1))
        self.dVrep = BSpline(*self.tck)

        # Vrep(r) = -int_r^Rcut Vrep'  ,  Vrep(Rcut) = 0
        A = self.dVrep.antiderivative()
        self.Vrep = lambda r: A(np.minimum(r,self.Rcut)) - A(self.Rcut)
        return self.Vrep

    def collect_traj(self):
        p = Path.cwd() /'Vpot'/'traj'
        mol = {}
        frame=namedtuple("frame", 'E_DFT Natoms atoms coords frame F')
        for traj in sorted(p.glob('*.xyz')):
            npz = Path.cwd()/'Vpot'/'data'/f'{traj.stem}.npz'
            F_DFT = np.load(npz)['F']*0.0194469 if npz.exists() else None
            lines = traj.read_text().splitlines()
            frames = []
            k = 0
            while k < len(lines):
                if not lines[k].split():  
                    k += 1
                    continue
                Natoms = int(lines[k].split()[0])
                head = lines[k+1].split()       
                fr = int(head[1])
                E_DFT = float(head[4])*0.0367493
                atoms = []
                coords = []
                for line in lines[k+2:k+2+Natoms]:
                    l = line.split()
                    atoms.append(ATOM_NAMES.index(l[0].lower()))
                    coords.append([float(x)*1.8897261246 for x in l[1:4]])
                F = None if F_DFT is None or fr >= len(F_DFT) else F_DFT[fr]
                frames.append(frame(E_DFT, Natoms, np.array(atoms),
                                    np.array(coords), fr, F))
                k += 2 + Natoms

            mol[traj.stem]=frames
        return mol










            


        




