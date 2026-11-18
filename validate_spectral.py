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


"""W1 Lemma 6, Corollary 7, and Proposition 8."""

import math
import unittest

import numpy as np

from validation_math import graph_forms, log_loss_product, rayleigh_gap


class SpectralChecks(unittest.TestCase):
    def test_lemma_06_form_and_variance_comparison(self):
        reference = [0.1, 0.2, 0.3, 0.4]
        target = [0.15, 0.15, 0.32, 0.38]
        edges = [(0, 1, 0.32), (1, 2, 0.51), (2, 3, 0.22), (0, 3, 0.28)]
        edge_factors = [0.86, 0.80, 0.96, 0.80]
        target_edges = [
            (i, j, conductance * factor)
            for (i, j, conductance), factor in zip(edges, edge_factors)
        ]
        alpha = 0.2
        zeta = 0.5
        ref_energy, ref_variance = graph_forms(reference, edges)
        energy, variance = graph_forms(target, target_edges)
        energy_difference = energy - (1 - alpha) * ref_energy
        variance_difference = (1 + zeta) * ref_variance - variance
        self.assertGreaterEqual(np.linalg.eigvalsh(energy_difference)[0], -1e-13)
        self.assertGreaterEqual(np.linalg.eigvalsh(variance_difference)[0], -1e-13)

        ratio = rayleigh_gap(energy, variance) / rayleigh_gap(ref_energy, ref_variance)
        self.assertGreaterEqual(ratio, (1 - alpha) / (1 + zeta) - 1e-13)
        self.assertLessEqual(max(0, 1 - ratio), (alpha + zeta) / (1 + zeta) + 1e-13)
        self.assertLessEqual((alpha + zeta) / (1 + zeta), alpha + zeta)

    def test_corollary_07_quartic_comparison_along_running_family(self):
        masses = [0.25, 0.5, 0.25]
        edges = [(0, 1, 0.4), (1, 2, 0.7), (0, 2, 0.2)]
        ref_energy, variance = graph_forms(masses, edges)
        ref_gap = rayleigh_gap(ref_energy, variance)
        constant = 0.75
        endpoint_x = 1 / 6
        decrement = 0.12
        margin = constant * endpoint_x**2
        ceiling = constant * (endpoint_x**2 + endpoint_x / decrement)
        for last in (0, 12, 64):
            losses = []
            for k in range(last + 1):
                x = 1 / (1 / endpoint_x + decrement * (last - k))
                alpha = constant * x**2
                perturbed = [
                    (i, j, conductance * (1 - fraction * alpha))
                    for (i, j, conductance), fraction in zip(edges, (1.0, 0.25, 0.6))
                ]
                energy, _ = graph_forms(masses, perturbed)
                loss = max(0.0, 1 - rayleigh_gap(energy, variance) / ref_gap)
                self.assertLessEqual(loss, alpha + 1e-13)
                losses.append(loss)
            with self.subTest(last=last):
                self.assertLessEqual(max(losses), margin + 1e-13)
                self.assertLessEqual(math.fsum(losses), ceiling + 1e-13)
                self.assertGreaterEqual(log_loss_product(losses), -ceiling / (1 - margin))

    def test_proposition_08_same_action_different_spectral_gap(self):
        masses = np.array([0.2, 0.5, 0.3])
        conductance = np.array([[0, 0.35, 0.09], [0.35, 0, 0.28], [0.09, 0.28, 0]])
        generator = conductance / masses[:, None]
        generator -= np.diag(generator.sum(axis=1))
        root_mass = np.sqrt(masses)
        symmetric = -root_mass[:, None] * generator / root_mass[None, :]
        ref_gap = np.linalg.eigvalsh(symmetric)[1]
        np.testing.assert_allclose(masses @ generator, 0.0, atol=1e-14)
        self.assertGreater(ref_gap, 0)

        constant = 0.9
        for x in (0.005, 0.15, 0.8):
            with self.subTest(g_squared=x):
                scaled = (1 - constant * x) * generator
                np.testing.assert_allclose(masses @ scaled, 0.0, atol=1e-14)
                np.testing.assert_allclose(masses[:, None] * scaled, (masses[:, None] * scaled).T)
                new_gap = np.linalg.eigvalsh(-root_mass[:, None] * scaled / root_mass[None, :])[1]
                self.assertAlmostEqual(new_gap / ref_gap, 1 - constant * x, places=13)
                self.assertAlmostEqual((1 - new_gap / ref_gap) / x, constant, places=11)


if __name__ == "__main__":
    unittest.main()
