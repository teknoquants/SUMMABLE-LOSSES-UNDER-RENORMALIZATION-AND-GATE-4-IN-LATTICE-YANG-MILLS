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


"""W1 Lemma 1, Proposition 2, Theorem 3, and Corollary 4."""

from fractions import Fraction
import math
import unittest

from validation_math import log_loss_product


class ProductChecks(unittest.TestCase):
    def test_lemma_01_log_bounds(self):
        for loss in (0.0, 1e-12, 0.04, 0.25, 0.5, 0.85, 0.999):
            with self.subTest(loss=loss):
                log_cost = -math.log1p(-loss)
                self.assertGreaterEqual(log_cost, loss)
                if loss <= 0.5:
                    self.assertLessEqual(log_cost, 2 * loss)

    def test_lemma_01_exact_telescoping_examples(self):
        for last in (2, 8, 32, 128):
            with self.subTest(last=last):
                summable = math.prod(1 - Fraction(1, n * n) for n in range(2, last + 1))
                harmonic = math.prod(1 - Fraction(1, n) for n in range(2, last + 1))
                self.assertEqual(summable, Fraction(last + 1, 2 * last))
                self.assertGreater(summable, Fraction(1, 2))
                self.assertEqual(harmonic, Fraction(1, last))

    def test_proposition_02_uniform_bound_and_converse(self):
        trajectories = [[], [0.0], [0.8, 0.01], [0.01] * 100, [0.4] * 4]
        total_bound = max(math.fsum(trajectory) for trajectory in trajectories)
        margin = max(max(trajectory, default=0.0) for trajectory in trajectories)
        products = [math.prod(1 - loss for loss in trajectory) for trajectory in trajectories]
        floor = math.exp(-total_bound / (1 - margin))
        smallest = min(products)
        self.assertGreaterEqual(smallest, floor)
        for trajectory in trajectories:
            self.assertLessEqual(max(trajectory, default=0.0), 1 - smallest + 1e-14)
            self.assertLessEqual(math.fsum(trajectory), -math.log(smallest) + 1e-14)

    def test_proposition_02_single_bad_step(self):
        previous = Fraction(1)
        for n in (2, 10, 1000, 10**9):
            loss = 1 - Fraction(1, n)
            self.assertLess(loss, 1)
            self.assertEqual(1 - loss, Fraction(1, n))
            self.assertLess(1 - loss, previous)
            previous = 1 - loss

    def test_theorem_03_superquadratic_sum_and_product(self):
        endpoint_x = 0.15
        decrement = 0.1
        constant = 1.25
        for epsilon in (0.4, 2.0, 3.0):
            power = 1 + epsilon / 2
            margin = constant * endpoint_x**power
            ceiling = constant * (
                endpoint_x**power + 2 * endpoint_x**(epsilon / 2) / (decrement * epsilon)
            )
            self.assertLess(margin, 1)
            for last in (0, 16, 256, 4096):
                with self.subTest(epsilon=epsilon, last=last):
                    losses = [
                        constant / (1 / endpoint_x + decrement * (last - k)) ** power
                        for k in range(last + 1)
                    ]
                    self.assertLessEqual(max(losses), margin + 1e-14)
                    self.assertLessEqual(math.fsum(losses), ceiling + 1e-14)
                    self.assertGreaterEqual(log_loss_product(losses), -ceiling / (1 - margin))

    def test_theorem_03_running_remainder_checkpoint(self):
        # u is g**(-2). Solve u_previous - A + B/u_previous = u_next backward.
        endpoint_u = 8.0
        one_loop = 0.13
        remainder_bound = 0.25
        self.assertLessEqual(remainder_bound / endpoint_u, one_loop / 2)
        u_next = endpoint_u
        for steps in range(1, 513):
            target = u_next + one_loop
            u = (target + math.sqrt(target * target - 4 * remainder_bound)) / 2
            self.assertAlmostEqual(u_next - u + one_loop, remainder_bound / u, places=12)
            self.assertGreaterEqual(u - u_next, one_loop / 2)
            self.assertGreaterEqual(u, endpoint_u + (one_loop / 2) * steps)
            u_next = u

    def test_corollary_04_quadratic_lower_loss(self):
        # u = 2, A_+ = c = 1. The product telescopes to 1/(J+2).
        for last in (0, 16, 256, 4096):
            with self.subTest(last=last):
                losses = [1 / (2 + last - k) for k in range(last + 1)]
                lower_loss_sum = math.log1p((last + 1) / 2)
                self.assertGreaterEqual(math.fsum(losses), lower_loss_sum)
                log_product = log_loss_product(losses)
                self.assertLessEqual(log_product, -lower_loss_sum)
                self.assertAlmostEqual(log_product, -math.log(last + 2), places=12)


if __name__ == "__main__":
    unittest.main()
