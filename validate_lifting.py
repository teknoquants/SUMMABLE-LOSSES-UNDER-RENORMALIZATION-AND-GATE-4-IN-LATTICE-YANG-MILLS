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


"""W1 Lemma 17, Theorem 18, and the conditional estimates on pages 9-10."""

from fractions import Fraction
import math
import unittest

import numpy as np
import sympy as sp


class LiftingChecks(unittest.TestCase):
    def test_lemma_17_exact_variance_and_orthogonality(self):
        block = [0, 0, 1, 1, 1, 2]
        masses = [Fraction(n, 21) for n in (1, 2, 3, 4, 5, 6)]
        values = [Fraction(n, d) for n, d in ((2, 3), (3, 1), (-3, 7), (4, 1), (1, 2), (0, 1))]
        block_mass = [
            sum(mass for mass, group in zip(masses, block) if group == i)
            for i in range(3)
        ]
        conditional_mean = [
            sum(mass*value for mass, value, group in zip(masses, values, block) if group == i)
            / block_mass[i]
            for i in range(3)
        ]
        mean = sum(mass*value for mass, value in zip(masses, values))
        resid = [value - conditional_mean[group] for value, group in zip(values, block)]
        for i in range(3):
            self.assertEqual(sum(mass*r for mass, r, group in zip(masses, resid, block) if group == i), 0)

        coarse = sum(mass*(value - mean)**2 for mass, value in zip(block_mass, conditional_mean))
        fiber = sum(mass*r*r for mass, r in zip(masses, resid))
        variance = sum(
            masses[i]*masses[j]*(values[i] - values[j])**2
            for i in range(len(masses))
            for j in range(i)
        )
        self.assertGreater(coarse, 0)
        self.assertGreater(fiber, 0)
        self.assertEqual(variance, coarse + fiber)
        self.assertEqual(
            sum(mass*r*(conditional_mean[group]-mean) for mass, r, group in zip(masses, resid, block)),
            0,
        )

    def test_lemma_17_eliminated_mode_can_set_the_gap(self):
        one_spin = np.array([[0.5, -0.5], [-0.5, 0.5]])
        retained_mode = np.array([1, 1, -1, -1])
        eliminated_mode = np.array([1, -1, 1, -1])
        for slow_rate in (0.2, 0.017, 1e-5):
            with self.subTest(slow_rate=slow_rate):
                fine = np.kron(one_spin, np.eye(2)) + slow_rate*np.kron(np.eye(2), one_spin)
                gap = np.linalg.eigvalsh(fine)[1]
                self.assertAlmostEqual(gap, slow_rate, places=13)
                np.testing.assert_allclose(fine @ retained_mode, retained_mode)
                np.testing.assert_allclose(fine @ eliminated_mode, slow_rate*eliminated_mode, atol=1e-15)
                np.testing.assert_array_equal(retained_mode.reshape(2, 2).var(axis=1), [0, 0])
                np.testing.assert_array_equal(eliminated_mode.reshape(2, 2).var(axis=1), [1, 1])

    def test_theorem_18_gap_lifting_in_common_physical_units(self):
        one_spin = np.array([[0.5, -0.5], [-0.5, 0.5]])
        endpoint_floor = 0.8
        for losses in ((), (0.18,), (0.1, 0.15, 0.04, 0.0, 0.07)):
            with self.subTest(losses=losses):
                k_last = len(losses)
                physical = [0.0] * k_last + [0.92]
                for k in range(k_last - 1, -1, -1):
                    physical[k] = (1 - losses[k])*physical[k+1]
                # Store a lattice-unit Hamiltonian at every scale, then recover its physical gap.
                spacing = [0.025*2**k for k in range(k_last + 1)]
                lattice_operators = [a*gap*one_spin for a, gap in zip(spacing, physical)]
                measured = [
                    np.linalg.eigvalsh(operator)[1]/a
                    for a, operator in zip(spacing, lattice_operators)
                ]
                self.assertGreaterEqual(measured[-1], endpoint_floor)
                for k, loss in enumerate(losses):
                    self.assertAlmostEqual(measured[k], (1 - loss)*measured[k+1], places=14)
                margin = max(losses, default=0)
                floor = endpoint_floor * math.exp(-math.fsum(losses)/(1 - margin))
                self.assertGreaterEqual(measured[0], floor)

    def test_theorem_18_forward_comparison_is_insufficient(self):
        fine_gap = Fraction(1, 100)
        coarse_gap = Fraction(1)
        renormalized_reference = Fraction(1)
        factor = Fraction(1)
        self.assertGreaterEqual(coarse_gap, factor * renormalized_reference)
        self.assertLess(fine_gap, factor * coarse_gap)

    def test_open_loss_criterion_quartic_domination(self):
        x, power, rate = sp.symbols("x p c", positive=True)
        expression = x**power * sp.exp(-rate*x)
        derivative = sp.simplify(sp.diff(expression, x) / (x**(power-1)*sp.exp(-rate*x)))
        self.assertEqual(derivative, power - rate*x)
        maximum = expression.subs(x, power/rate)
        self.assertEqual(sp.simplify(maximum - (power/(sp.E*rate))**power), 0)
        for delta in (0.5, 1.0, 1.9):
            for c in (0.3, 1.0, 2.0):
                with self.subTest(delta=delta, c=c):
                    power_value = 4/delta
                    max_log = power_value*(math.log(power_value/c) - 1)
                    for g in (0.01, 0.05, 0.3, 0.9, 1.25):
                        log_ratio = -c/g**delta - 4*math.log(g)
                        self.assertLessEqual(log_ratio, max_log + 1e-13)

    def test_open_loss_criterion_weak_product_and_intermediate_steps(self):
        sf, lf, c, delta = 0.4, 0.02, 0.9, 1.3
        endpoint_x = 0.12
        decrement = 0.095
        effective = sf + lf*(4/(math.e*c*delta))**(4/delta)
        self.assertLessEqual(effective*endpoint_x**2, 0.5)
        weak_floor = math.exp(-2*effective * (endpoint_x**2 + endpoint_x/decrement))
        intermediate_factors = (0.90, 0.87, 0.99)
        bridge_floor = 0.85**len(intermediate_factors)
        self.assertGreaterEqual(math.prod(intermediate_factors), bridge_floor)
        for last in (0, 20, 400):
            weak_factors = []
            for step in range(last + 1):
                g_squared = 1 / (1/endpoint_x + decrement*(last - step))
                loss = sf*g_squared**2 + lf*math.exp(-c / g_squared**(delta/2))
                self.assertLessEqual(loss, effective*g_squared**2)
                weak_factors.append(1 - loss)
            with self.subTest(last=last):
                self.assertGreaterEqual(math.prod(weak_factors), weak_floor)
                fine_gap = 0.42
                for factor in reversed((*weak_factors, *intermediate_factors)):
                    fine_gap *= factor
                endpoint_floor = 0.4
                self.assertGreaterEqual(fine_gap, endpoint_floor * bridge_floor * weak_floor)


if __name__ == "__main__":
    unittest.main()
