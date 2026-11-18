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


"""W1 Theorem 15 and Proposition 16."""

from collections import Counter
import math
import unittest

import numpy as np
from scipy.special import eval_chebyu, eval_gegenbauer
import sympy as sp

from validation_math import rayleigh_gap, su2_haar_quadrature


def quaternion_product(left, right):
    a, b, c, d = left
    e, f, g, h = right
    return (
        a*e - b*f - c*g - d*h,
        a*f + b*e + c*h - d*g,
        a*g - b*h + c*e + d*f,
        a*h + b*g - c*f + d*e,
    )


def radial_forms(traces, weights, cutoff):
    """Build forms on nonconstant SU(2) characters with the paper's metric."""
    order = np.arange(1, cutoff + 1)[:, None]
    values = eval_chebyu(order, traces)
    derivatives = 2 * eval_gegenbauer(order - 1, 2, traces)
    means = values @ weights
    variance = (values * weights) @ values.T - np.outer(means, means)
    metric_weights = 0.5 * (1 - traces*traces) * weights
    energy = (derivatives * metric_weights) @ derivatives.T
    return energy, variance


class HierarchyChecks(unittest.TestCase):
    def test_theorem_15_su2_haar_invariance_under_left_and_right_products(self):
        q = sp.symbols("a b c d", real=True)
        identity_quaternion = sp.eye(4)
        basis = [tuple(identity_quaternion[:, i]) for i in range(4)]
        left = sp.Matrix.hstack(*(sp.Matrix(quaternion_product(q, e)) for e in basis))
        right = sp.Matrix.hstack(*(sp.Matrix(quaternion_product(e, q)) for e in basis))
        expected = sum(coordinate**2 for coordinate in q) * sp.eye(4)
        self.assertEqual((left.T*left - expected).applyfunc(sp.expand), sp.zeros(4))
        self.assertEqual((right.T*right - expected).applyfunc(sp.expand), sp.zeros(4))

    def test_theorem_15_exact_group_block_of_sixteen(self):
        # Q8 is the finite quaternion subgroup of SU(2).
        states = [
            tuple(sign if coordinate == axis else 0 for coordinate in range(4))
            for axis in range(4)
            for sign in (1, -1)
        ]
        self.assertNotEqual(
            quaternion_product(states[2], states[4]), quaternion_product(states[4], states[2])
        )
        counts = Counter({(1, 0, 0, 0): 1})
        for _ in range(16):
            next_counts = Counter()
            for left, count in counts.items():
                for right in states:
                    next_counts[quaternion_product(left, right)] += count
            counts = next_counts
        self.assertEqual(counts, Counter({state: 8**15 for state in states}))
        self.assertEqual(sum(counts.values()), 8**16)

    def test_theorem_15_full_rate_one_heat_bath_spectrum(self):
        for states, coordinates in ((2, 4), (3, 3), (8, 2)):
            with self.subTest(states=states, coordinates=coordinates):
                one_identity = np.eye(states)
                one_projection = np.ones((states, states)) / states
                size = states**coordinates
                product_identity = np.eye(size)
                minus_generator = np.zeros((size, size))
                for refreshed in range(coordinates):
                    projection = np.ones((1, 1))
                    for coordinate in range(coordinates):
                        factor = one_projection if coordinate == refreshed else one_identity
                        projection = np.kron(projection, factor)
                    minus_generator += product_identity - projection
                spectrum = np.linalg.eigvalsh(minus_generator)
                expected = [
                    k
                    for k in range(coordinates + 1)
                    for _ in range(math.comb(coordinates, k) * (states - 1)**k)
                ]
                np.testing.assert_allclose(spectrum, expected, atol=2e-13, rtol=0)
                self.assertAlmostEqual(spectrum[1], 1, places=13)

    def test_proposition_16_same_metric_and_bounded_density(self):
        traces, haar = su2_haar_quadrature(72)
        for cutoff in (6, 12):
            reference_energy, reference_variance = radial_forms(traces, haar, cutoff)
            orders = np.arange(1, cutoff+1)
            np.testing.assert_allclose(reference_energy, np.diag(orders*(orders+2)/2), atol=3e-12)
            np.testing.assert_allclose(reference_variance, np.eye(cutoff), atol=3e-13)
            reference_gap = rayleigh_gap(reference_energy, reference_variance)
            self.assertAlmostEqual(reference_gap, 1.5, places=12)
            for g in (0.2, 0.45, 0.7, 0.9):
                with self.subTest(cutoff=cutoff, g=g):
                    # R = -oscillation*x^2, whose tilt favors both central elements.
                    oscillation = 1.6 * g**4
                    unnormalized = haar * np.exp(oscillation * traces**2)
                    normalizing = float(np.sum(unnormalized))
                    lower_density = 1/normalizing
                    upper_density = math.exp(oscillation)/normalizing
                    energy, variance = radial_forms(traces, unnormalized/normalizing, cutoff)
                    self.assertGreaterEqual(
                        np.linalg.eigvalsh(energy - lower_density*reference_energy)[0], -1e-12
                    )
                    self.assertGreaterEqual(
                        np.linalg.eigvalsh(upper_density*reference_variance - variance)[0], -1e-12
                    )
                    multiplier = lower_density / upper_density
                    self.assertAlmostEqual(multiplier, math.exp(-oscillation), places=14)
                    ratio = rayleigh_gap(energy, variance) / reference_gap
                    self.assertLess(ratio, 1)
                    self.assertGreaterEqual(ratio, multiplier - 1e-12)
                    self.assertLessEqual(1-ratio, -math.expm1(-oscillation) + 1e-12)
                    self.assertLessEqual(-math.expm1(-oscillation), oscillation)


if __name__ == "__main__":
    unittest.main()
