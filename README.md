# Exercise W1 Python validation suite

This suite accompanies *Summable Losses Under Renormalization and Gate G4 in Lattice Yang–Mills* by Roger Cooperman and A.E. Petrunina V.V., XISOSIS, 18 November 2026. The source is the 13-page PDF named W1_Summable_Losses_and_Gate_G4_in_Lattice_Yang-Mills.pdf. Its SHA-256 is b14b482a498e35ba0539e740a09e5215756fc4d3dfa99eb586c532fad2518268.

## Run the checks

Extract the ZIP, open a terminal in the extracted folder, and use Python 3.11 or later.

```sh
python -m pip install -r requirements.txt
python run_all.py
```

The runner prints the named tests, returns a nonzero exit status on failure, and prints the recomputed input tables after success. To run a single group, use a command such as `python -m unittest -v validate_heat_kernel`. VALIDATION_RESULTS.txt records the versions used and the successful run.

## Coverage

The 18 numbered results in the PDF comprise five lemmas, four corollaries, four theorems and five propositions. The paper contains no formally stated conjecture.

| PDF | Result | Module | Validation |
| --- | --- | --- | --- |
| p. 4 | Lemma 1, infinite products | validate_products.py | Logarithmic bounds and exact rational telescoping examples with positive and vanishing limits |
| p. 4 | Proposition 2, uniform margin | validate_products.py | Sum and margin product floor, converse bounds, and the single-step counterexample |
| p. 5 | Theorem 3, running-coupling summability | validate_products.py | Integral-test bounds for three exponents and a solved recurrence with a nonzero remainder |
| p. 5 | Corollary 4, quadratic lower loss | validate_products.py | Harmonic lower bound and the exact telescoping product |
| p. 5 | Proposition 5, B2 coupling convention | validate_running.py | Symbolic chain rule, factor-of-two conversion, and the SU(2) value of the one-loop decrement |
| p. 6 | Lemma 6, sufficient spectral comparison | validate_spectral.py | Energy and variance matrix inequalities and independently computed generalized eigenvalues |
| p. 6 | Corollary 7, quartic remainder | validate_spectral.py | Finite-model Rayleigh gaps with quartic perturbations, then sum, margin, and product bounds along running trajectories |
| p. 6 | Proposition 8, fixed-action time rescaling | validate_spectral.py | Same stationary law and a quadratic relative loss in the directly computed generator spectrum |
| p. 6 | Lemma 9, cutoff exponent | validate_large_fields.py | Symbolic exponent identities and numerical checks for both cutoffs |
| p. 6 | Lemma 10, Gaussian bound | validate_large_fields.py | Symbolic exponential moment and comparison with chi-square tail probabilities |
| p. 7 | Proposition 11, small-probability bottleneck | validate_large_fields.py | Exact detailed balance, characteristic polynomial, and collapsing gap |
| p. 7 | Corollary 12, stretched-exponential summability | validate_large_fields.py | Incomplete gamma integral bound and the exact geometric-series case |
| p. 7 | Theorem 13, exact planar blocking | validate_heat_kernel.py | Direct SU(2) Haar convolution at central and noncentral elements, with two convolved pairs recombined into a four-plaquette block |
| p. 8 | Corollary 14, string tension and cylinder gap | validate_heat_kernel.py | Casimir calculations, Haar-character integrals, physical gap at all 11 printed scales, and a fixed-link control |
| p. 8 | Theorem 15, ultralocal hierarchy | validate_hierarchy.py | Symbolic quaternion isometries, exact counts for a 16-variable quaternion subgroup block, and the full spectrum of finite product heat baths |
| p. 9 | Proposition 16, fixed-block quartic perturbation | validate_hierarchy.py | Same-metric SU(2) character forms and density comparison at two representation cutoffs |
| p. 9 | Lemma 17, coarse and fiber variance | validate_lifting.py | Exact rational variance split and a slow eliminated-mode example |
| p. 9 | Theorem 18, conditional gap transfer | validate_lifting.py | Normalization of lattice-unit spectra, backward product bound, and a forward-comparison counterexample |
| pp. 9–10 | Proposed loss bound (5) and endpoint | validate_lifting.py | Symbolic domination by a quartic loss and a model trajectory with an intermediate product and positive endpoint |
| pp. 10–11 | Section 7, Tables 1 and 2 | validate_running.py | Inverse primitive and independent event-resolved ODE for six rows, plus exact one-loop inputs and loss products for four rows |

## Scope of the results

The checks use exact rational arithmetic, symbolic identities, and numerical examples. The heat-kernel tests use the SU(2) character series through degree 24 and a uniform bound on its omitted tail at the tested heat times. The fixed-block density comparison uses two finite character spaces. Table 1 is an extrapolation of the truncated two-loop ODE to g² = 6.

The interacting four-dimensional Wilson loss estimate in equation (5), the corresponding volume-uniform comparison on the full physical space, and the certified infrared endpoint remain open. The conditional checks exercise the product bound and the endpoint arithmetic on explicit models.

## MIT License

```text
MIT License

Copyright (c) 2026 Roger Cooperman; A.E. Petrunina V.V.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
