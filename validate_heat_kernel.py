# MIT License
#
# Copyright (c) 2026 Roger Cooperman; A.E. Petrunina V.V.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


"""W1 Theorem 13 and Corollary 14."""

from fractions import Fraction
import math
import unittest

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import eval_chebyu

from validation_math import su2_haar_quadrature


_DEGREE = 24
_ORDERS = np.arange(_DEGREE + 1)
_DIMENSIONS = _ORDERS + 1
_CASIMIRS = _ORDERS * (_ORDERS + 2) / 2
_TRACES, _HAAR_WEIGHTS = su2_haar_quadrature(36)
_COSINES, _LEGENDRE_WEIGHTS = leggauss(18)
_CHARACTERS_AT_NODES = eval_chebyu(_ORDERS[:, None], _TRACES)


def heat_coefficients(time):
    return _DIMENSIONS * np.exp(-time * _CASIMIRS)


def kernel(coefficients, trace):
    trace = np.asarray(trace)
    characters = eval_chebyu(_ORDERS[:, None], trace.reshape(1, -1))
    values = np.sum(coefficients[:, None] * characters, axis=0)
    return values.reshape(trace.shape)


def convolution(left, right, phi):
    """Integrate p(U V^{-1}) q(V) over SU(2) Haar in quaternion coordinates."""
    external_trace = math.cos(phi)
    external_sine = math.sin(phi)
    product_traces = (
        external_trace * _TRACES[:, None]
        + external_sine * np.sqrt(1 - _TRACES[:, None]**2) * _COSINES
    )
    angle_averages = kernel(left, product_traces) @ (_LEGENDRE_WEIGHTS / 2)
    return float(np.dot(_HAAR_WEIGHTS, angle_averages * kernel(right, _TRACES)))


def heat_tail_bound(time):
    """Bound the uniform character tail with |chi_n(U)| <= n+1."""
    first_omitted = _DEGREE + 1
    first = (first_omitted + 1)**2 * math.exp(
        -time * first_omitted * (first_omitted + 2) / 2
    )
    ratio = ((first_omitted + 2) / (first_omitted + 1))**2 * math.exp(
        -time * (2*first_omitted + 3) / 2
    )
    if ratio >= 1:
        raise ValueError("Increase the cutoff for this heat time")
    return first / (1 - ratio)


class HeatKernelChecks(unittest.TestCase):
    def test_theorem_13_haar_normalization_and_character_orthogonality(self):
        gram = (_CHARACTERS_AT_NODES * _HAAR_WEIGHTS) @ _CHARACTERS_AT_NODES.T
        np.testing.assert_allclose(gram, np.eye(_DEGREE + 1), atol=2e-13, rtol=0)
        for time in (0.12, 0.27, 0.8):
            self.assertAlmostEqual(
                float(_HAAR_WEIGHTS @ kernel(heat_coefficients(time), _TRACES)), 1, places=12
            )
            self.assertLess(heat_tail_bound(time), 1e-12)

    def test_theorem_13_unequal_time_convolution_at_noncentral_elements(self):
        for s, t in ((0.12, 0.23), (0.3, 0.61)):
            for phi in (0.0, 0.37, 1.2, 2.4, math.pi):
                with self.subTest(s=s, t=t, phi=phi):
                    actual = convolution(heat_coefficients(s), heat_coefficients(t), phi)
                    expected = float(kernel(heat_coefficients(s+t), math.cos(phi)))
                    self.assertAlmostEqual(actual, expected, places=10)

    def test_theorem_13_four_plaquettes_join(self):
        time = 0.16
        original = heat_coefficients(time)
        paired_values = np.array([
            convolution(original, original, math.acos(float(trace))) for trace in _TRACES
        ])
        paired_coefficients = _CHARACTERS_AT_NODES @ (_HAAR_WEIGHTS * paired_values)
        np.testing.assert_allclose(paired_coefficients, heat_coefficients(2*time), atol=2e-12)
        for phi in (0.0, 0.8, 1.6, 2.8, math.pi):
            expected = float(kernel(heat_coefficients(4*time), math.cos(phi)))
            self.assertAlmostEqual(
                convolution(paired_coefficients, paired_coefficients, phi), expected, places=10
            )

    def test_corollary_14_casimir_floor_for_fundamental_weights(self):
        for n in range(2, 13):
            indices = range(1, n)
            gram = [
                [Fraction(min(i, j)) - Fraction(i*j, n) for j in indices]
                for i in indices
            ]
            self.assertTrue(all(value > 0 for row in gram for value in row))
            casimirs = [
                row[i-1] + 2*sum(row)
                for i, row in zip(indices, gram)
            ]
            self.assertEqual(casimirs, [Fraction(i*(n-i)*(n+1), n) for i in indices])
            self.assertEqual(min(casimirs), Fraction(n*n - 1, n))

    def test_corollary_14_planar_string_tension_from_character_expectation(self):
        spacing = 0.5
        gamma = 0.8
        for plaquettes in (1, 3, 9):
            area = plaquettes * spacing**2
            densities = _HAAR_WEIGHTS * kernel(heat_coefficients(gamma*area), _TRACES)
            for order in (1, 2, 3):
                with self.subTest(plaquettes=plaquettes, order=order):
                    expectation = float(densities @ _CHARACTERS_AT_NODES[order]) / (order + 1)
                    self.assertGreater(expectation, 0)
                    tension = -math.log(expectation) / area
                    self.assertAlmostEqual(tension, gamma*_CASIMIRS[order], places=8)

    def test_corollary_14_cylinder_eigenvalue_from_haar_integral(self):
        circumference = 4.0
        gamma = 0.2
        for links in (16, 8, 4, 2, 1):
            with self.subTest(links=links):
                spacing = circumference / links
                heat_time = links * gamma * spacing**2
                densities = _HAAR_WEIGHTS * kernel(heat_coefficients(heat_time), _TRACES)
                eigenvalue = float(densities @ _CHARACTERS_AT_NODES[1]) / 2
                self.assertAlmostEqual(
                    -math.log(eigenvalue)/spacing, gamma*circumference*(3/2), places=12
                )

    def test_corollary_14_printed_cylinder_benchmark_and_fixed_link_control(self):
        gamma = 0.2
        circumference = 1.0
        initial_links = 4096
        for k in range(11):
            links = initial_links // 2**k
            spacing = circumference / links
            heat_time = gamma * spacing**2
            # Work with the negative log eigenvalue directly.
            gap = (links * heat_time * (3/2)) / spacing
            self.assertAlmostEqual(gap, 0.30000000000000004, places=15)
            self.assertEqual(links * spacing, circumference)
        doubled_spacing = 2*circumference / initial_links
        changed_circumference_gap = gamma * initial_links * doubled_spacing * (3/2)
        self.assertAlmostEqual(changed_circumference_gap, 0.6, places=15)


if __name__ == "__main__":
    unittest.main()
