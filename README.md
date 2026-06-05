# Computational Algorithm for Equilibrium Optimal Transport
This repository is for the experiment in the paper "Computational Algorithm for Equilibrium Optimal Transport".

## Project Overview
This project implements a **quadratically regularized equilibrium optimal transport (Eqm OT)** experimental framework, including:

1. **Inner OT**: standard quadratic (L2) regularized OT to compute conditional transport plans.
2. **Outer OT**: inner conditional couplings weighted quadratic (Weighted L2) OT for computing the final coupling.
3. **Solver comparisons**:
   - APDAGD (Adaptive Primal-Dual Accelerated Gradient Descent)
   - Sinkhorn (weighted L2 version coordinate updates)
   - Dual Solver (L-BFGS-B based)

The project enables reproducing experiments and generating plots for runtime, sparsity, and unrounded marginal constraint violations.

## Dependencies

- Python >= 3.9
- NumPy
- SciPy
- Matplotlib
- Jupyter Notebook / JupyterLab

Install dependencies with:
```bash
pip install numpy scipy matplotlib jupyter
```

## Usage

1. Place `eqm_ot_experiments.ipynb` and all `.py` files in the same folder.
2. Open the notebook and check the experiment configuration:
```python
RUN_EXPERIMENTS = False  # Default: False, to preview notebook without running full experiments
```
3. Set experiment parameters: `n_values`, `eps_values`, iteration limits, and regularization coefficient `eta`.
4. Set `RUN_EXPERIMENTS = True` and run the notebook to perform experiments and generate plots.

## Experiment Description

1. **APDAGD vs Sinkhorn**
   - Compares adaptive gradient descent vs weighted L2 Sinkhorn for Eqm OT.
   - Inner OT: standard quadratic L2 regularization.
   - Outer OT: weighted quadratic regularization.

2. **APDAGD vs Dual L-BFGS**
   - Compares APDAGD vs L-BFGS-B dual solver.
   - Supports the same Eqm OT problem.

3. **Metrics**:
   - `runtime`: elapsed computation time
   - `avg_sparsity`: fraction of zero entries in transport plans
   - `outer_violation`, `inner_mean_violation`, `inner_max_violation`: unrounded marginal constraint violations

## Reproducibility

- Random seeds are fixed for each `n` and `eps` to ensure repeatability.
- Use notebook plotting utilities `plot_metric_vs_n` and `plot_metric_vs_log_inv_eps` to generate runtime, sparsity, and violation plots.
- Plots are saved in `plots/` folder.

## Code Design

- **Solver**: `apdagd.py`, `sinkhorn.py`, `dual_l2_solver.py`
  - Unified interface: `fit(problem, L=None, max_iter=None)`
  - Optional logging: `log=True/False`

- **Problem**: `base.py`, `euclidean.py`
  - `BaseProblem`: abstract interface for OT problems
  - `EuclideanRegularizedOTProblem` and `WeightedEuclideanRegularizedOTProblem`: concrete implementations
  - `solve_nested_euclidean_ot`: handles Eqm OT problem flow

- **Notebook**: `eqm_ot_experiments.ipynb`
  - Combined from two original notebooks
  - Includes configuration, running experiments, plotting, and summarizing results

## Notes

- Weighted L2 Sinkhorn converges slower than APDAGD; adjust iteration limits if needed.
- All experiments use unrounded transport plans for sparsity and constraint-violation metrics.
- Keep all `.py` files in the same folder as the notebook for correct imports on GitHub.
- The original code of the APDAGD and Sinkhorn can be accessed through https://github.com/MuXauJl11110/Euclidean-Regularised-Optimal-Transport.
- The original code of the dual solver method can be accessed through https://github.com/mblondel/smooth-ot.
- If you have any questions, please contact yzhang854@connect.hkust-gz.edu.cn.
