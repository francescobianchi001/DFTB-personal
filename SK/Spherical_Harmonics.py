#! /usr/bin/env python3
import numpy as np
from itertools import permutations
from sympy import (
    Symbol, symbols, cos, sin, sqrt, exp, I,
    factorial, factorial2, binomial, diff,integrate, powsimp, simplify,
    Rational, pi, Matrix, Poly, Integer, expand, factor, cancel, together,
    fraction, reduced, count_ops, re, im,
)
from sympy.printing.numpy import NumPyPrinter

theta, phi = symbols('theta phi', real=True)
# theta1, theta2: the two independent polar angles of a two-center pair
# (first harmonic -> theta1, second harmonic -> theta2).
theta1, theta2 = symbols('theta1 theta2', real=True)

x = Symbol('x')

_printer = NumPyPrinter()


def associated_legendre(l, m):
    f = (x**2 - 1)**l
    poly = (-1)**m / (2**l * factorial(l)) * diff(f, x, l + m)
    return poly.subs(x, cos(theta)) * sin(theta)**m

def complex_harmonic(l, m):
    norm = sqrt((2 * l + 1) / (4 * pi) * factorial(l - m) / factorial(l + m))
    return norm * associated_legendre(l, m) * exp(I * m * phi)

def real_harmonic(l, m):
    am = abs(m)
    norm = sqrt((2 * l + 1) / (4 * pi) * factorial(l - am) / factorial(l + am))
    P = norm * associated_legendre(l, am)

    if m == 0:
        # No phi dependence here, so simplify keeps the polynomial factored
        # (e.g. (3cos^2 - 1)/4) without the product-to-sum mangling.
        return simplify(P)
    if m > 0:
        return powsimp(sqrt(2) * (-1)**m * P * cos(m * phi))
    return powsimp(sqrt(2) * (-1)**am * P * sin(am * phi))

def to_numpy(expr):
    """Render a sympy expression as a NumPy-evaluable code string."""
    return _printer.doprint(expr)

def azimutal_integration(expr1,expr2):
    # Keep the two polar angles distinct: only phi is shared and integrated.
    e1 = expr1.subs(theta, theta1)
    e2 = expr2.subs(theta, theta2)
    A_int = powsimp(integrate(e1*e2, (phi,0,2*pi)))
    return A_int

def build(l_max=2):
    Y = {}
    A_int = {}
    m_sum = {}
    for l in range(0, l_max + 1):
        for m in range(-l, l + 1):
            Y[(l, m)] = real_harmonic(l, m)
    for l in range(0, l_max + 1):
        for j in range(0, l + 1):              
            m_count = 0
            for m in range(0, min(l, j) + 1):  
                A_int[(l, j, m)] = azimutal_integration(Y[(l, m)], Y[(j, m)])
                m_count += A_int[(l,j,m)]
            m_sum[(l,j)] = m_count
    return Y, A_int,m_sum

### Slater-Koster rotation table: the same harmonics, rotated onto the bond ###

# The angular weights above are built in the frame where the bond IS the z axis,
# so they only ever produce the sigma/pi/delta integrals V_{l j m}. For a bond
# pointing along the direction cosines (L, M, N) the matrix element of a pair of
# lab-frame orbitals is a fixed linear combination of those same integrals,
#
#     E_{l1 m1, l2 m2}(L, M, N) = sum_m  D^{l1}_{m,m1} D^{l2}_{m,m2} V_{l1 l2 |m|}
#
# because a rotation cannot mix different l. The D's come from rotating the real
# regular solid harmonics C_lm = r^l Y_lm (Racah normalised: the r^l keeps them
# polynomial, so the projection below is exact),
#
#     C_lm(Rinv u) = sum_mu D^l_{mu,m} C_l,mu(u),
#
# and are extracted by projecting on the unit sphere. The table depends only on
# (l, m) -- never on the radial part -- so several radial functions per l all
# reuse the one table.

cx, cy, cz = symbols('cx cy cz', real=True)      # cartesian, lab frame
u1, u2, u3 = symbols('u1 u2 u3', real=True)      # cartesian, bond frame
L, M, N = symbols('L M N', real=True)            # direction cosines of the bond
s = Symbol('s', positive=True)                   # sqrt(L**2 + M**2)

SPHERE = L**2 + M**2 + N**2 - 1                  # vanishes on the unit sphere

def solid_legendre(l, m, w, r2):
    # Polynomial form of the theta part: what P_l^m becomes once the sin(theta)^m
    # is pulled out into the (x + iy)^m factor of solid_harmonic.
    tot = 0
    for k in range((l - m) // 2 + 1):
        tot += ((-1)**k * Rational(1, 2**l) * binomial(l, k)
                * binomial(2 * l - 2 * k, l)
                * factorial(l - 2 * k) / factorial(l - 2 * k - m)
                * r2**k * w**(l - 2 * k - m))
    return expand(tot)

def solid_harmonic(l, m, a=cx, b=cy, c=cz):
    """Real regular solid harmonic r^l Y_lm, Racah normalised.
    m > 0 cosine type, m < 0 sine type -- same convention (and sign) as
    real_harmonic, up to the l-dependent constant sqrt(4 pi / (2 l + 1))."""
    am = abs(m)
    r2 = a**2 + b**2 + c**2
    norm = sqrt(Rational(2 - (1 if m == 0 else 0))) * sqrt(
        Rational(int(factorial(l - am)), int(factorial(l + am))))
    ang = expand((a + I * b)**am)
    ang = re(ang) if m >= 0 else im(ang)
    return expand(norm * solid_legendre(l, am, c, r2) * expand(ang))

def sphere_average(expr, coords):
    # Average of a polynomial over the unit sphere. Odd powers integrate to zero,
    # the even ones to a ratio of double factorials -- no quadrature needed.
    a, b, c = coords
    p = Poly(expand(expr), a, b, c)
    tot = 0
    for (i, j, k), coeff in p.terms():
        if i % 2 or j % 2 or k % 2:
            continue
        tot += coeff * (factorial2(i - 1) * factorial2(j - 1) * factorial2(k - 1)
                        / factorial2(i + j + k + 1))
    return simplify(tot)

# Rinv sends zhat onto the bond direction (L, M, N). The leftover freedom of
# rotating about the bond axis cancels because +m and -m are always summed
# together, so any choice of the two transverse columns gives the same table.
Rinv = Matrix([[L * N / s, -M / s, L],
               [M * N / s,  L / s, M],
               [-s,         0,     N]])
_rot = Rinv * Matrix([u1, u2, u3])
ROT = {cx: _rot[0], cy: _rot[1], cz: _rot[2]}

def reduce_direction_cosines(e):
    """Reduce modulo s**2 = L**2 + M**2 and L**2 + M**2 + N**2 = 1, which is what
    turns the raw rational expressions into the published polynomials. The result
    is not unique, so try a few monomial orders and keep the most compact one."""
    e = expand(e)
    for _ in range(8):
        e = expand(e.subs(s**2, L**2 + M**2))
    if e.has(s):
        raise RuntimeError('odd power of s survived: ' + str(e))

    best = None
    for order in ('grevlex', 'grlex'):
        for perm in permutations((L, M, N)):
            cur = e
            for _ in range(10):
                cur = cancel(together(expand(cur)))
                num, den = fraction(cur)
                num = reduced(expand(num), [SPHERE], *perm, order=order, domain='EX')[1]
                den = reduced(expand(den), [SPHERE], *perm, order=order, domain='EX')[1]
                new = cancel(expand(num) / expand(den))
                if new == cur:
                    break
                cur = new
            cur = factor(simplify(cur))
            if best is None or count_ops(cur) < count_ops(best):
                best = cur
    return best

_D = {}

def rotation_D(l):
    """D^l_{mu,m}(L, M, N): the real-harmonic rotation matrix for one l."""
    if l in _D:
        return _D[l]
    basis = {mu: solid_harmonic(l, mu, u1, u2, u3) for mu in range(-l, l + 1)}
    norms = {mu: sphere_average(basis[mu]**2, (u1, u2, u3)) for mu in basis}
    D = {}
    for m in range(-l, l + 1):
        rotated = expand(solid_harmonic(l, m).subs(ROT, simultaneous=True))
        for mu in range(-l, l + 1):
            proj = sphere_average(expand(rotated * basis[mu]), (u1, u2, u3))
            D[(mu, m)] = simplify(proj / norms[mu])
    _D[l] = D
    return D

_SK = {}

def sk_coefficients(l1, m1, l2, m2):
    """{|mu|: coefficient of V_{l1 l2 |mu|}} for the (l1 m1 | l2 m2) element.
    Only the non-vanishing channels are returned."""
    if (l1, m1, l2, m2) in _SK:
        return _SK[(l1, m1, l2, m2)]
    D1, D2 = rotation_D(l1), rotation_D(l2)
    out = {}
    for mu in range(0, min(l1, l2) + 1):
        if mu == 0:
            c = D1[(0, m1)] * D2[(0, m2)]
        else:                                   # +mu and -mu share the same V
            c = D1[(mu, m1)] * D2[(mu, m2)] + D1[(-mu, m1)] * D2[(-mu, m2)]
        c = reduce_direction_cosines(c)
        if simplify(c) != 0:
            out[mu] = c
    _SK[(l1, m1, l2, m2)] = out
    return out

def build_rotations(l_max=2):
    rotations = {}
    for l1 in range(0, l_max + 1):
        for l2 in range(0, l_max + 1):
            for m1 in range(-l1, l1 + 1):
                for m2 in range(-l2, l2 + 1):
                    for mu, c in sk_coefficients(l1, m1, l2, m2).items():
                        rotations[(l1, m1, l2, m2, mu)] = c
    return rotations

LABEL = {0: {0: 's'},
         1: {-1: 'y', 0: 'z', 1: 'x'},
         2: {-2: 'xy', -1: 'yz', 0: 'z2', 1: 'zx', 2: 'x2-y2'}}
GREEK = {0: 'sig', 1: 'pi', 2: 'del'}

def published_table():
    """The subset of Slater & Koster (1954) Table I used to check the generator."""
    h = Rational(1, 2)
    r3 = sqrt(3)
    z2 = N**2 - (L**2 + M**2) / 2
    d = L**2 - M**2
    return {
        ('s', 's'):   {0: Integer(1)},
        ('s', 'x'):   {0: L},
        ('x', 'x'):   {0: L**2, 1: 1 - L**2},
        ('x', 'y'):   {0: L * M, 1: -L * M},
        ('x', 'z'):   {0: L * N, 1: -L * N},
        ('s', 'xy'):  {0: r3 * L * M},
        ('s', 'x2-y2'): {0: r3 * h * d},
        ('s', 'z2'):  {0: z2},
        ('x', 'xy'):  {0: r3 * L**2 * M, 1: M * (1 - 2 * L**2)},
        ('x', 'yz'):  {0: r3 * L * M * N, 1: -2 * L * M * N},
        ('x', 'zx'):  {0: r3 * L**2 * N, 1: N * (1 - 2 * L**2)},
        ('x', 'x2-y2'): {0: r3 * h * L * d, 1: L * (1 - d)},
        ('y', 'x2-y2'): {0: r3 * h * M * d, 1: -M * (1 + d)},
        ('z', 'x2-y2'): {0: r3 * h * N * d, 1: -N * d},
        ('x', 'z2'):  {0: L * z2, 1: -r3 * L * N**2},
        ('z', 'z2'):  {0: N * z2, 1: r3 * N * (L**2 + M**2)},
        ('xy', 'xy'): {0: 3 * L**2 * M**2,
                       1: L**2 + M**2 - 4 * L**2 * M**2,
                       2: N**2 + L**2 * M**2},
        ('xy', 'yz'): {0: 3 * L * M**2 * N,
                       1: L * N * (1 - 4 * M**2),
                       2: L * N * (M**2 - 1)},
        ('xy', 'zx'): {0: 3 * L**2 * M * N,
                       1: M * N * (1 - 4 * L**2),
                       2: M * N * (L**2 - 1)},
        ('xy', 'x2-y2'): {0: Rational(3, 2) * L * M * d,
                          1: 2 * L * M * (M**2 - L**2),
                          2: h * L * M * d},
        ('yz', 'x2-y2'): {0: Rational(3, 2) * M * N * d,
                          1: -M * N * (1 + 2 * d),
                          2: M * N * (1 + d / 2)},
        ('zx', 'x2-y2'): {0: Rational(3, 2) * N * L * d,
                          1: N * L * (1 - 2 * d),
                          2: -N * L * (1 - d / 2)},
        ('xy', 'z2'): {0: r3 * L * M * z2, 1: -2 * r3 * L * M * N**2,
                       2: r3 * h * L * M * (1 + N**2)},
        ('yz', 'z2'): {0: r3 * M * N * z2,
                       1: r3 * M * N * (L**2 + M**2 - N**2),
                       2: -r3 * h * M * N * (L**2 + M**2)},
        ('zx', 'z2'): {0: r3 * L * N * z2,
                       1: r3 * L * N * (L**2 + M**2 - N**2),
                       2: -r3 * h * L * N * (L**2 + M**2)},
        ('z2', 'z2'): {0: z2**2, 1: 3 * N**2 * (L**2 + M**2),
                       2: Rational(3, 4) * (L**2 + M**2)**2},
        ('x2-y2', 'x2-y2'): {0: Rational(3, 4) * d**2,
                             1: L**2 + M**2 - d**2,
                             2: N**2 + d**2 / 4},
    }

def verify_rotations(npts=40, tol=1e-11):
    """Compare the generated table with the published one on random bond
    directions. Cheap once build_rotations has filled the cache."""
    inv = {name: (l, m) for l, dd in LABEL.items() for m, name in dd.items()}

    rng = np.random.default_rng(20260722)
    pts = rng.normal(size=(npts, 3))
    pts /= np.linalg.norm(pts, axis=1)[:, None]

    nfail = 0
    ref = published_table()
    for (o1, o2), rr in ref.items():
        l1, m1 = inv[o1]
        l2, m2 = inv[o2]
        got = sk_coefficients(l1, m1, l2, m2)
        for mu in set(rr) | set(got):
            e = expand(got.get(mu, 0) - rr.get(mu, 0))
            worst = max(abs(float(re(e.subs({L: a, M: b, N: c}).evalf())))
                        for a, b, c in pts)
            if worst >= tol:
                nfail += 1
                print(f'  FAIL <{o1}|{o2}> {GREEK[mu]}  maxdev={worst:.2e}')
                print(f'       got {got.get(mu, 0)}')
                print(f'       ref {rr.get(mu, 0)}')
    print(f'rotations: {len(ref)} published entries checked, {nfail} mismatches')
    return nfail

def show_rotations(l1, l2):
    print(f"--- l={l1}  l'={l2} ---")
    for m1 in sorted(LABEL[l1], key=lambda t: -t):
        for m2 in sorted(LABEL[l2], key=lambda t: -t):
            c = sk_coefficients(l1, m1, l2, m2)
            terms = '  '.join(f'+ ({v}) V{GREEK[k]}' for k, v in sorted(c.items()))
            print(f'  E[{LABEL[l1][m1]:>6},{LABEL[l2][m2]:>6}] = {terms or "0"}')
    print()

### defining the new varible in terms of x and z for the integration ###

Z,X,R = symbols('z x R', real=True)

sub_map={
        cos(theta1): Z /sqrt(X**2 +Z**2),
        sin(theta1): X/sqrt(X**2 +Z**2),

        cos(theta2): (Z-R)/sqrt(X**2 +(Z-R)**2),
        sin(theta2): X/sqrt(X**2 +(Z-R)**2),
}
lmax = 2
Y, A_int, m_sum = build(l_max=lmax)

# m-resolved angular weights W_{l,j,m}(x, z, R): substitute the cylindrical
# coordinates into each bond-type integral separately (NOT summed over m), so
# every Slater-Koster channel (sigma=0, pi=1, delta=2) stays its own term for
# the overlap matrix.  Built for both (l, j) orders since theta1 -> z/r_A is
# tied to the first harmonic and theta2 -> (z-R)/r_B to the second.
angular_weights = {}
for l in range(0, lmax + 1):
    for j in range(0, lmax + 1):
        for m in range(0, min(l, j) + 1):
            angular_weights[(l, j, m)] = azimutal_integration(
                Y[(l, m)], Y[(j, m)]).subs(sub_map)

def write_module(Y, A_int, angular_weights, rotations, path='Y_real.py'):
    with open(path, 'w') as fh:
        fh.write('#! /usr/bin/env python3\n')
        fh.write('"""Auto-generated real spherical harmonics in NumPy notation.\n')
        fh.write('Do not edit by hand."""\n\n')
        fh.write('import numpy\n\n\n')
        fh.write('class Y_real:\n')
        fh.write('    """Real spherical harmonics keyed by (l, m)."""\n\n')
        fh.write('    harmonics = {\n')
        for (l, m), expr in Y.items():
            fh.write(f'        ({l}, {m}): lambda theta, phi: {to_numpy(expr)},\n')
        fh.write('    }\n\n')
        fh.write('    # Azimuthal (phi) integrals  int_0^2pi Y_lm(theta1) Y_jm(theta2) dphi,\n')
        fh.write('    # keyed (l, j, m); theta1 is the first harmonic angle, theta2 the second.\n')
        fh.write('    azimutal = {\n')
        for (l, j, m), expr in A_int.items():
            fh.write(f'        ({l}, {j}, {m}): lambda theta1, theta2: {to_numpy(expr)},\n')
        fh.write('    }\n\n')
        fh.write('    # m-resolved angular weights in cylindrical (x, z, R), keyed (l, j, m):\n')
        fh.write('    #   x = cylindrical radius, z = axial coordinate, R = internuclear distance.\n')
        fh.write('    # theta1, theta2 already substituted via cos t1 = z/sqrt(x^2+z^2),\n')
        fh.write('    # cos t2 = (z-R)/sqrt(x^2+(z-R)^2).  m = 0(sigma), 1(pi), 2(delta).\n')
        fh.write('    # S_{l j m}(R) = int R_l(r_A) R_j(r_B) angular_weights[(l,j,m)] x dx dz\n')
        fh.write('    angular_weights = {\n')
        for (l, j, m), expr in angular_weights.items():
            fh.write(f'        ({l}, {j}, {m}): lambda x, z, R: {to_numpy(expr)},\n')
        fh.write('    }\n\n')
        fh.write('    # Slater-Koster rotation table, keyed (l1, m1, l2, m2, m):\n')
        fh.write('    #   E_{l1 m1, l2 m2}(L, M, N) = sum_m rotations[(l1,m1,l2,m2,m)](L,M,N)\n')
        fh.write('    #                                     * V_{l1 l2 m}\n')
        fh.write('    # (L, M, N) = direction cosines of the bond A -> B, m = 0(sigma) 1(pi)\n')
        fh.write('    # 2(delta) -- the same channels as angular_weights, whose integrals are\n')
        fh.write('    # exactly the V_{l1 l2 m}.  Keys absent from the dict are zero.\n')
        fh.write('    rotations = {\n')
        for (l1, m1, l2, m2, m), expr in rotations.items():
            fh.write(f'        ({l1}, {m1}, {l2}, {m2}, {m}): '
                     f'lambda L, M, N: {to_numpy(expr)},\n')
        fh.write('    }\n')


if __name__ == '__main__':
    Y, A_int, m_sum = build(l_max=lmax)
    for (l, m), expr in Y.items():
        print(f'Y[l={l}, m={m:+d}](theta, phi) = {to_numpy(expr)}')
    # Same lmax as the angular weights: both describe the same s/p/d basis.
    rotations = build_rotations(l_max=lmax)
    verify_rotations()
    write_module(Y, A_int, angular_weights, rotations)
    print('\nWrote Y_real.py')
