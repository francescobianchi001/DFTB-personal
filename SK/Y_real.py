#! /usr/bin/env python3
"""Auto-generated real spherical harmonics in NumPy notation.
Do not edit by hand."""

import numpy


class Y_real:
    """Real spherical harmonics keyed by (l, m)."""

    harmonics = {
        (0, 0): lambda theta, phi: (1/2)/numpy.sqrt(numpy.pi),
        (1, -1): lambda theta, phi: (1/2)*numpy.sqrt(3)*numpy.sin(phi)*numpy.sin(theta)/numpy.sqrt(numpy.pi),
        (1, 0): lambda theta, phi: (1/2)*numpy.sqrt(3)*numpy.cos(theta)/numpy.sqrt(numpy.pi),
        (1, 1): lambda theta, phi: (1/2)*numpy.sqrt(3)*numpy.sin(theta)*numpy.cos(phi)/numpy.sqrt(numpy.pi),
        (2, -2): lambda theta, phi: (1/4)*numpy.sqrt(15)*numpy.sin(2*phi)*numpy.sin(theta)**2/numpy.sqrt(numpy.pi),
        (2, -1): lambda theta, phi: (1/2)*numpy.sqrt(15)*numpy.sin(phi)*numpy.sin(theta)*numpy.cos(theta)/numpy.sqrt(numpy.pi),
        (2, 0): lambda theta, phi: (1/4)*numpy.sqrt(5)*(3*numpy.cos(theta)**2 - 1)/numpy.sqrt(numpy.pi),
        (2, 1): lambda theta, phi: (1/2)*numpy.sqrt(15)*numpy.sin(theta)*numpy.cos(phi)*numpy.cos(theta)/numpy.sqrt(numpy.pi),
        (2, 2): lambda theta, phi: (1/4)*numpy.sqrt(15)*numpy.sin(theta)**2*numpy.cos(2*phi)/numpy.sqrt(numpy.pi),
    }

    # Azimuthal (phi) integrals  int_0^2pi Y_lm(theta1) Y_jm(theta2) dphi,
    # keyed (l, j, m); theta1 is the first harmonic angle, theta2 the second.
    azimutal = {
        (0, 0, 0): lambda theta1, theta2: 1/2,
        (1, 0, 0): lambda theta1, theta2: (1/2)*numpy.sqrt(3)*numpy.cos(theta1),
        (1, 1, 0): lambda theta1, theta2: (3/2)*numpy.cos(theta1)*numpy.cos(theta2),
        (1, 1, 1): lambda theta1, theta2: (3/4)*numpy.sin(theta1)*numpy.sin(theta2),
        (2, 0, 0): lambda theta1, theta2: numpy.sqrt(5)*((3/4)*numpy.cos(theta1)**2 - 1/4),
        (2, 1, 0): lambda theta1, theta2: (1/4)*numpy.sqrt(15)*(3*numpy.cos(theta1)**2 - 1)*numpy.cos(theta2),
        (2, 1, 1): lambda theta1, theta2: (3/4)*numpy.sqrt(5)*numpy.sin(theta1)*numpy.sin(theta2)*numpy.cos(theta1),
        (2, 2, 0): lambda theta1, theta2: (5/8)*(3*numpy.cos(theta1)**2 - 1)*(3*numpy.cos(theta2)**2 - 1),
        (2, 2, 1): lambda theta1, theta2: (15/4)*numpy.sin(theta1)*numpy.sin(theta2)*numpy.cos(theta1)*numpy.cos(theta2),
        (2, 2, 2): lambda theta1, theta2: (15/16)*numpy.sin(theta1)**2*numpy.sin(theta2)**2,
    }

    # m-resolved angular weights in cylindrical (x, z, R), keyed (l, j, m):
    #   x = cylindrical radius, z = axial coordinate, R = internuclear distance.
    # theta1, theta2 already substituted via cos t1 = z/sqrt(x^2+z^2),
    # cos t2 = (z-R)/sqrt(x^2+(z-R)^2).  m = 0(sigma), 1(pi), 2(delta).
    # S_{l j m}(R) = int R_l(r_A) R_j(r_B) angular_weights[(l,j,m)] x dx dz
    angular_weights = {
        (0, 0, 0): lambda x, z, R: 1/2,
        (0, 1, 0): lambda x, z, R: (1/2)*numpy.sqrt(3)*(-R + z)/numpy.sqrt(x**2 + (-R + z)**2),
        (0, 2, 0): lambda x, z, R: numpy.sqrt(5)*((3/4)*(-R + z)**2/(x**2 + (-R + z)**2) - 1/4),
        (1, 0, 0): lambda x, z, R: (1/2)*numpy.sqrt(3)*z/numpy.sqrt(x**2 + z**2),
        (1, 1, 0): lambda x, z, R: (3/2)*z*(-R + z)/(numpy.sqrt(x**2 + z**2)*numpy.sqrt(x**2 + (-R + z)**2)),
        (1, 1, 1): lambda x, z, R: (3/4)*x**2/(numpy.sqrt(x**2 + z**2)*numpy.sqrt(x**2 + (-R + z)**2)),
        (1, 2, 0): lambda x, z, R: (1/4)*numpy.sqrt(15)*z*(3*(-R + z)**2/(x**2 + (-R + z)**2) - 1)/numpy.sqrt(x**2 + z**2),
        (1, 2, 1): lambda x, z, R: (3/4)*numpy.sqrt(5)*x**2*(-R + z)/(numpy.sqrt(x**2 + z**2)*(x**2 + (-R + z)**2)),
        (2, 0, 0): lambda x, z, R: numpy.sqrt(5)*((3/4)*z**2/(x**2 + z**2) - 1/4),
        (2, 1, 0): lambda x, z, R: (1/4)*numpy.sqrt(15)*(-R + z)*(3*z**2/(x**2 + z**2) - 1)/numpy.sqrt(x**2 + (-R + z)**2),
        (2, 1, 1): lambda x, z, R: (3/4)*numpy.sqrt(5)*x**2*z/((x**2 + z**2)*numpy.sqrt(x**2 + (-R + z)**2)),
        (2, 2, 0): lambda x, z, R: (5/8)*(3*z**2/(x**2 + z**2) - 1)*(3*(-R + z)**2/(x**2 + (-R + z)**2) - 1),
        (2, 2, 1): lambda x, z, R: (15/4)*x**2*z*(-R + z)/((x**2 + z**2)*(x**2 + (-R + z)**2)),
        (2, 2, 2): lambda x, z, R: (15/16)*x**4/((x**2 + z**2)*(x**2 + (-R + z)**2)),
    }
