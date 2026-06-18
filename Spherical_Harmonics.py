#! /usr/bin/env python3
import numpy as np
from sympy import (
    Symbol, symbols, cos, sin, sqrt, exp, I,
    factorial, diff,integrate, powsimp, simplify, Rational, pi,
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

def write_module(Y, A_int, angular_weights, path='Y_real.py'):
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
        fh.write('    }\n')


if __name__ == '__main__':
    Y, A_int, m_sum = build(l_max=2)
    for (l, m), expr in Y.items():
        print(f'Y[l={l}, m={m:+d}](theta, phi) = {to_numpy(expr)}')
    write_module(Y, A_int, angular_weights)
    print('\nWrote Y_real.py')
