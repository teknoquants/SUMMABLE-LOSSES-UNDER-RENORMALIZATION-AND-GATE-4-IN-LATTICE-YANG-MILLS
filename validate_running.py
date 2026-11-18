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


"""W1 Proposition 5 and the two numerical tables in section 7."""

import math
import unittest

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import sympy as sp

from validation_math import log_loss_product


TABLE_1 = (
    (2, 0.5, 12.938633, 12, 0.938633, 14.59513, 23.01361),
    (2, 1.0, 5.571000, 5, 0.571000, 11.38926, 26.86186),
    (2, 1.5, 3.206992, 3, 0.206992, 11.01592, 37.16150),
    (3, 0.5, 8.271160, 8, 0.271160, 11.67822, 25.24500),
    (3, 1.0, 3.482067, 3, 0.482067, 7.34395, 16.51622),
    (3, 1.5, 1.971561, 1, 0.971561, 3.81641, 7.61577),
)
TABLE_2 = (
    (10, 1.66194, 0.25205, 1.90341e-2, 0.596781),
    (100, 9.01151, 0.89836, 1.78182e-9, 0.161816),
    (1000, 24.25482, 1.25082, 4.91539e-23, 0.079901),
    (10000, 41.81720, 1.30242, 2.46605e-38, 0.072065),
)


def flow_coefficients(n):
    b0 = 11 * n / (48 * math.pi**2)
    b1 = 17 * n**2 / (384 * math.pi**4)
    return 4 * b0 * math.log(2), 8 * b1 * math.log(2)


def invert_two_loop(n, start):
    """Return k_star and x_k values at integer scales strictly before k_star."""
    alpha, beta = flow_coefficients(n)

    def primitive(x):
        return -1 / (alpha * x) + beta / alpha**2 * math.log((alpha + beta * x) / x)

    initial = primitive(start)
    hit = primitive(6.0) - initial
    last = math.ceil(hit) - 1
    x_values = [start]
    for k in range(1, last + 1):
        root = brentq(lambda x: primitive(x) - initial - k, start, 6.0, xtol=1e-13)
        x_values.append(root)
    return hit, x_values


def two_loop_row(n, start, hit, x_values):
    last = len(x_values) - 1
    return (
        n,
        start,
        hit,
        last,
        hit - last,
        math.fsum(x_values),
        math.fsum(x*x for x in x_values),
    )


def one_loop_row(last):
    a, _ = flow_coefficients(2)
    x_values = [1 / (6 + a * (last - k)) for k in range(last + 1)]
    squared_values = [x*x for x in x_values]
    return (
        last,
        math.fsum(x_values),
        math.fsum(squared_values),
        math.exp(log_loss_product(2*x for x in x_values)),
        math.exp(log_loss_product(2*x_squared for x_squared in squared_values)),
    )


def benchmark_report():
    """Format recomputed values to the precision used in the paper."""
    lines = [
        "Table 1 of W1 - truncated two-loop extrapolation to g^2 = 6",
        "N    g0^2       k_star   k_last   fraction     sum(g^2)     sum(g^4)",
    ]
    for n, start, *_ in TABLE_1:
        _, _, hit, last, fraction, sum_x, sum_x_squared = two_loop_row(
            n, start, *invert_two_loop(n, start)
        )
        lines.append(
            f"{n:1d}    {start:.1f}   {hit:11.6f}   {last:6d}   {fraction:.6f}"
            f"   {sum_x:10.5f}   {sum_x_squared:10.5f}"
        )
    lines += [
        "",
        "Table 2 of W1 - exact one-loop inputs with illustrative losses",
        "     K     sum(g^2)     sum(g^4)           P2         P4",
    ]
    for last, *_ in TABLE_2:
        _, sum_x, sum_x_squared, p2, p4 = one_loop_row(last)
        lines.append(
            f"{last:6d}   {sum_x:10.5f}   {sum_x_squared:10.5f}   {p2:12.5e}   {p4:.6f}"
        )
    return "\n".join(lines)


class RunningChecks(unittest.TestCase):
    def test_proposition_05_symbolic_coupling_conversion(self):
        gs, x, b0, b1 = sp.symbols("g_s x b_0 b_1", positive=True)
        # Log(mu) decreases by log(2) per block step.
        step_flow = sp.log(2) * (b0 * gs**3 + b1 * gs**5)
        converted = sp.diff(gs**2 / 2, gs) * step_flow
        actual = converted.subs(gs, sp.sqrt(2*x)).expand()
        expected = (sp.log(2) * (4*b0*x**2 + 8*b1*x**3)).expand()
        self.assertEqual(actual, expected)
        standard = sp.diff(gs**-2, gs) * step_flow.subs(b1, 0)
        b2_convention = sp.diff(2*gs**-2, gs) * step_flow.subs(b1, 0)
        self.assertEqual(sp.simplify(standard), -2*b0*sp.log(2))
        self.assertEqual(sp.simplify(b2_convention), -4*b0*sp.log(2))
        a, _ = flow_coefficients(2)
        self.assertAlmostEqual(a, 11 * math.log(2) / (6 * math.pi**2), places=15)
        self.assertAlmostEqual(a, 0.12875590341658527, places=15)

    def test_table_1_symbolic_primitive(self):
        x, alpha, beta = sp.symbols("x alpha beta", positive=True)
        primitive = -1/(alpha*x) + beta/alpha**2 * sp.log((alpha + beta*x)/x)
        residual = sp.diff(primitive, x) - 1/(x**2 * (alpha + beta*x))
        self.assertEqual(sp.simplify(residual), 0)

    def test_table_1_rows_by_inversion_and_ode(self):
        for n, start, *expected in TABLE_1:
            with self.subTest(n=n, start=start):
                hit, x_values = invert_two_loop(n, start)
                _, _, _, last, fraction, sum_x, sum_x_squared = two_loop_row(
                    n, start, hit, x_values
                )
                self.assertEqual(
                    (round(hit, 6), last, round(fraction, 6), round(sum_x, 5), round(sum_x_squared, 5)),
                    tuple(expected),
                )
                self.assertLess(last, hit)
                self.assertLess(hit, last + 1)
                alpha, beta = flow_coefficients(n)

                def rhs(_, values):
                    x = values[0]
                    return [x*x * (alpha + beta*x)]

                def terminal_event(_, values):
                    return values[0] - 6.0

                terminal_event.terminal = True
                terminal_event.direction = 1
                integration = solve_ivp(
                    rhs,
                    (0, 2*hit),
                    [start],
                    events=terminal_event,
                    dense_output=True,
                    rtol=2e-12,
                    atol=2e-13,
                )
                self.assertTrue(integration.success, integration.message)
                self.assertEqual(len(integration.t_events[0]), 1)
                self.assertAlmostEqual(integration.t_events[0][0], hit, places=9)
                np.testing.assert_allclose(
                    integration.sol(np.arange(last + 1))[0], x_values, rtol=1e-9, atol=1e-11
                )

    def test_table_2_rows_and_quartic_floor(self):
        a, _ = flow_coefficients(2)
        ceiling = 1/36 + 1/(6*a)
        floor = math.exp(-2*ceiling / (1 - 1/18))
        self.assertAlmostEqual(floor, 0.0608107599, places=10)
        for last, expected_x, expected_x_squared, expected_p2, expected_p4 in TABLE_2:
            with self.subTest(last=last):
                _, sum_x, sum_x_squared, p2, p4 = one_loop_row(last)
                self.assertEqual(round(sum_x, 5), expected_x)
                self.assertEqual(round(sum_x_squared, 5), expected_x_squared)
                self.assertEqual(float(f"{p2:.5e}"), expected_p2)
                self.assertEqual(round(p4, 6), expected_p4)
                self.assertLessEqual(sum_x_squared, ceiling)
                self.assertGreaterEqual(p4, floor)


if __name__ == "__main__":
    unittest.main()
