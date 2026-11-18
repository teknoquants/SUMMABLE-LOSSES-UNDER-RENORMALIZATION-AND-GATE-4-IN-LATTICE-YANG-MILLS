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


"""Shared numerical primitives for the W1 checks."""

import math

import numpy as np
from scipy.linalg import eigh


def log_loss_product(losses):
    """Return log(prod(1 - loss)) without multiplying small factors."""
    return math.fsum(math.log1p(-loss) for loss in losses)


def graph_forms(masses, edges):
    """Build energy and variance matrices with the last observable value fixed to zero."""
    masses = np.asarray(masses, dtype=float)
    energy = np.zeros((len(masses), len(masses)))
    for left, right, conductance in edges:
        energy[left, left] += conductance
        energy[right, right] += conductance
        energy[left, right] -= conductance
        energy[right, left] -= conductance
    variance = np.diag(masses) - np.outer(masses, masses)
    return energy[:-1, :-1], variance[:-1, :-1]


def rayleigh_gap(energy, variance):
    """Return the smallest generalized eigenvalue on coordinates modulo constants."""
    return float(eigh(energy, variance, eigvals_only=True, subset_by_index=(0, 0))[0])


def su2_haar_quadrature(order):
    """Return trace nodes and normalized Haar weights for SU(2) class functions."""
    theta = np.pi * np.arange(1, order + 1) / (order + 1)
    return np.cos(theta), (2 / (order + 1)) * np.sin(theta) ** 2
