import numpy as np, io, contextlib
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from Hamiltonian import Vrep_fit

RC = 3.0236                                   # hotbit's 1.6 Ang cutoff
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e2e1dd', '#fcfcfb'

with contextlib.redirect_stdout(io.StringIO()):
    f = Vrep_fit(typor0=True, Rcut=RC)
Ve = f.integration(pool='deriv');  de, tck_e = f.dVrep, f.tck
Vf = f.integration(pool='derivF'); df = f.dVrep
rows, on = [], False
for ln in open('Vpot/par/C_H_repulsion.par'):
    if ln.startswith('repulsion='): on = True; continue
    if on:
        fl = ln.split()
        if len(fl) != 2: break
        rows.append([float(fl[0]), float(fl[1])])
hb = np.array(rows)
Vh, dVh = CubicSpline(hb[:,0],hb[:,1]), CubicSpline(hb[:,0],hb[:,1]).derivative()

r = np.linspace(1.15, RC, 400)
pe = np.array(f.deriv); pf = np.array(f.derivF)

fig, ax = plt.subplots(3,1, figsize=(7.2,9.6), sharex=True,
                       gridspec_kw=dict(height_ratios=[1.25,.85,1.1], hspace=.13))
for a in ax:
    a.set_facecolor(SURF)
    a.grid(True, color=GRID, lw=.7, zorder=0)
    a.set_axisbelow(True)
    for s in ('top','right'): a.spines[s].set_visible(False)
    for s in ('left','bottom'): a.spines[s].set_color(GRID)
    a.tick_params(colors=INK2, labelsize=9)
fig.patch.set_facecolor(SURF)

# --- V_rep
ax[0].plot(r, Vh(r), color=AQUA,   lw=3.2, alpha=.5, zorder=2, label='hotbit C_H_repulsion.par')
ax[0].plot(r, Ve(r), color=BLUE,   lw=2.0, zorder=3, label='energy route (2 splines)')
ax[0].plot(r, Vf(r), color=ORANGE, lw=2.0, ls=(0,(5,2.5)), zorder=4, label='force route (finite diff)')
ax[0].set_ylabel('$V_{rep}(r)$   [Ha]', color=INK, fontsize=10)
ax[0].set_title('C–H repulsive potential', color=INK, fontsize=12, loc='left', pad=10)
ax[0].legend(frameon=False, fontsize=9, labelcolor=INK2, loc='upper right')
ax[0].annotate('hotbit', (2.05, float(Vh(2.05))), textcoords='offset points',
               xytext=(6,10), color=AQUA, fontsize=9, weight='bold')

# --- residual vs hotbit
ax[1].axhline(0, color=AQUA, lw=3.2, alpha=.5, zorder=2)
ax[1].plot(r, 1e3*(Ve(r)-Vh(r)), color=BLUE,   lw=2.0, zorder=3)
ax[1].plot(r, 1e3*(Vf(r)-Vh(r)), color=ORANGE, lw=2.0, ls=(0,(5,2.5)), zorder=4)
ax[1].set_ylabel('$V_{rep}-V_{hotbit}$   [mHa]', color=INK, fontsize=10)
ax[1].annotate('energy',(1.30,1e3*float(Ve(1.30)-Vh(1.30))),textcoords='offset points',
               xytext=(7,-4),color=BLUE,fontsize=9,weight='bold')
ax[1].annotate('force', (1.30,1e3*float(Vf(1.30)-Vh(1.30))),textcoords='offset points',
               xytext=(7,4),color=ORANGE,fontsize=9,weight='bold')

# --- V_rep' with the pooled data
ax[2].plot(r, dVh(r), color=AQUA,   lw=3.2, alpha=.5, zorder=2)
ax[2].plot(r, de(r),  color=BLUE,   lw=2.0, zorder=3)
ax[2].plot(r, df(r),  color=ORANGE, lw=2.0, ls=(0,(5,2.5)), zorder=4)
ax[2].scatter(pe[:,0], pe[:,1], s=46, facecolor='none', edgecolor=BLUE,   lw=1.4, zorder=5)
ax[2].scatter(pf[:,0], pf[:,1], s=20, color=ORANGE, zorder=6)
ax[2].set_ylabel("$V'_{rep}(r)$   [Ha/$a_0$]", color=INK, fontsize=10)
ax[2].set_xlabel('$r$   [$a_0$]', color=INK, fontsize=10)
ax[2].set_ylim(-1.45, .12)
ax[2].annotate('fitted points: rings = energy, dots = force', (1.9,-1.28),
               color=INK2, fontsize=9)
ax[2].set_xlim(1.15, RC+.03)
fig.savefig('vrep.png',
            dpi=150, bbox_inches='tight', facecolor=SURF)
print("max |energy-hotbit| = %.4f Ha ; max |force-hotbit| = %.4f ; max |force-energy| = %.4f"
      % (np.abs(Ve(r)-Vh(r)).max(), np.abs(Vf(r)-Vh(r)).max(), np.abs(Vf(r)-Ve(r)).max()))
