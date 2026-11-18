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


"""W1 Lemmas 9 and 10, Proposition 11, and Corollary 12."""

import math
import unittest

from scipy.special import gamma, gammaincc
import sympy as sp


class LargeFieldChecks(unittest.TestCase):
    def test_lemma_09_cutoff_exponents(self):
        delta = sp.symbols("delta", real=True)
        cutoff_power = 1 - delta/2
        second_cutoff_power = delta/2
        self.assertEqual(sp.expand(2*cutoff_power - 2), -delta)
        self.assertEqual(sp.expand(2*second_cutoff_power - 2), -(2 - delta))
        for delta_value in (0.1, 0.5, 1.0, 1.7, 1.99):
            for g in (1e-3, 0.17, 0.7):
                with self.subTest(delta=delta_value, g=g):
                    cutoff = g**(1 - delta_value/2)
                    alternate = g**(delta_value/2)
                    self.assertGreater(cutoff, g)
                    self.assertLess(cutoff, 1)
                    self.assertTrue(math.isclose((cutoff/g)**2, g**(-delta_value), rel_tol=1e-12))
                    self.assertTrue(
                        math.isclose((alternate/g)**2, g**(-(2 - delta_value)), rel_tol=1e-12)
                    )

    def test_lemma_10_gaussian_exponential_moment(self):
        z = sp.symbols("z", real=True)
        density = sp.exp(-z*z/2) / sp.sqrt(2*sp.pi)
        moment = sp.integrate(sp.exp(z*z/4) * density, (z, -sp.oo, sp.oo))
        self.assertEqual(moment, sp.sqrt(2))

    def test_lemma_10_gaussian_tail_against_chi_square(self):
        for dimension in (1, 2, 3, 9, 25):
            for radius_over_g in (0, 0.15, 1, 2.5, 4, 10, 25):
                with self.subTest(dimension=dimension, radius_over_g=radius_over_g):
                    exact_tail = gammaincc(dimension/2, radius_over_g**2/2)
                    markov_bound = math.exp(
                        (dimension/2) * math.log(2) - radius_over_g**2/4
                    )
                    self.assertLessEqual(exact_tail, markov_bound * (1 + 1e-13))

    def test_proposition_11_stationarity_and_spectrum(self):
        epsilon, eigenvalue = sp.symbols("epsilon lambda", positive=True)
        rate = epsilon / (1 - epsilon)
        q = sp.Matrix([
            [-rate, rate, 0],
            [sp.Rational(1, 2), -1, sp.Rational(1, 2)],
            [0, rate, -rate],
        ])
        pi = sp.diag((1 - epsilon)/2, epsilon, (1 - epsilon)/2)
        self.assertEqual((pi*q - q.T*pi).applyfunc(sp.simplify), sp.zeros(3))
        self.assertEqual((q * sp.ones(3, 1)).applyfunc(sp.simplify), sp.zeros(3, 1))
        polynomial = (eigenvalue*sp.eye(3) - q).det()
        expected = eigenvalue * (eigenvalue + rate) * (eigenvalue + 1/(1 - epsilon))
        self.assertEqual(sp.simplify(polynomial - expected), 0)
        odd = sp.Matrix([1, 0, -1])
        self.assertEqual(((q + rate*sp.eye(3))*odd).applyfunc(sp.simplify), sp.zeros(3, 1))
        self.assertEqual(sp.limit(rate, epsilon, 0, dir="+"), 0)

    def test_corollary_12_integral_majorant(self):
        endpoint_u = 5.0
        decrement = 0.17
        for c in (0.2, 0.7):
            for delta in (0.5, 1.0, 1.5, 2.0, 3.0):
                with self.subTest(c=c, delta=delta):
                    power = delta/2
                    shape = 1/power
                    endpoint = c * endpoint_u**power
                    integral = gamma(shape) * gammaincc(shape, endpoint)
                    integral /= decrement * power * c**shape
                    ceiling = math.exp(-endpoint) + integral
                    finite_sum = math.fsum(
                        math.exp(-c * (endpoint_u + decrement*j)**power)
                        for j in range(5001)
                    )
                    self.assertTrue(math.isfinite(ceiling))
                    self.assertLessEqual(finite_sum, ceiling * (1 + 1e-13))
                    if delta == 2:
                        geometric = math.exp(-endpoint) / -math.expm1(-c * decrement)
                        self.assertTrue(math.isclose(finite_sum, geometric, rel_tol=1e-13))


if __name__ == "__main__":
    unittest.main()
