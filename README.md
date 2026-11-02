
# Gradient Boosting from Scratch: the Math, the Code, and When It Beats Simpler Models

**Author:** pakeezaafzaal · Python, NumPy, SciPy; scikit-learn used only as a reference

## What this project does

It implements three machine-learning algorithms from scratch in NumPy (L2-regularised **logistic regression**, a **CART decision tree**, and **gradient boosting**), checks each against scikit-learn, and then asks an honest question: *when does boosting actually beat simpler models?*

## Key findings

- **The implementations are correct.** The logistic-regression gradient matches finite differences to about 1e-10, and the fitted model matches scikit-learn's solution to about 6e-7 once scikit-learn's stopping tolerance is tightened. Trees match scikit-learn exactly up to depth 3.
- **Boosting did not beat linear models on two small real datasets.** On breast cancer, logistic regression had the lower log-loss (0.076 vs 0.119); on diabetes, linear regression had the lower RMSE (54.6 vs 57.4), though that gap is within noise.
- **It does win once the signal is nonlinear.** In a controlled synthetic experiment, linear regression won only when the signal was purely linear. At 20% nonlinear signal, boosting was already clearly better, and the gap grew steadily after that.

## The math in brief

**Logistic regression.** With $p = \sigma(w^\top x + b)$, the objective is
$J(w,b) = \frac1n\sum_i \big[\log(1+e^{z_i}) - y_i z_i\big] + \frac{\lambda}{2}\lVert w\rVert^2$.
Its gradient is $\nabla_w J = \frac1n X^\top (p - y) + \lambda w$, and $J$ is convex because the Hessian $\frac1n X^\top D X + \lambda I$, with $D=\mathrm{diag}(p_i(1-p_i))$, is positive semi-definite. Since $p(1-p)\le \tfrac14$, the gradient is Lipschitz, so gradient descent with step $1/L$ converges. The match with scikit-learn's `C` is $\lambda = 1/(Cn)$.

**Decision tree (CART).** A split on feature $j$ at threshold $t$ is chosen to maximise
$S = \sum_k c_{L,k}^2/n_L + \sum_k c_{R,k}^2/n_R$,
which is equivalent to minimising the weighted child impurity (Gini for classification, variance for regression). Cumulative sums over the sorted feature give $S$ for every threshold in one pass.

**Gradient boosting.** Boosting is gradient descent in function space: start from a constant $F_0$ and update $F_m = F_{m-1} + \nu\, h_m$, where $h_m$ is a small regression tree fitted to the negative gradient $r_i = -\partial \ell / \partial F(x_i)$. For squared error, $r_i = y_i - F(x_i)$. For logistic loss, $r_i = y_i - \sigma(F(x_i))$, and each leaf uses one Newton step, $\sum r_i / \sum p_i(1-p_i)$, as scikit-learn does.

## Validation against scikit-learn

| Check | Result |
|---|---|
| Logistic gradient vs finite differences | max error about 1.3e-10 |
| Logistic coefficients vs scikit-learn (`tol=1e-12`) | max difference about 5.7e-7; objectives agree to about 1e-15 |
| Logistic accuracy / AUC (breast cancer, test) | 0.958 / 0.995, identical to scikit-learn |
| Trees, depth 1-3 (breast cancer) | 100% identical predictions and leaf counts |
| Trees, depth 2-3 (diabetes) | max prediction difference 0.0 |
| Trees, depth 5 | 5 of 143 classification predictions differ (first mismatch: two splits with the identical score 9.0); 1 of 111 regression predictions differs, cause not identified |
| Boosting (100 trees, depth 3, lr 0.1), breast cancer | accuracy 0.958 both; AUC 0.9783 vs 0.9777; log-loss 0.1955 vs 0.1953 |
| Boosting, diabetes | test RMSE about 62.5 for both; largest single-prediction difference 9.8 |

The remaining boosting differences are consistent with tie-breaking between equally good splits: a single tree fitted to the same residuals gave identical training predictions (difference about 6e-14) but different test predictions, and scikit-learn's own predictions change by up to about 6.6 across random seeds. This is an explanation supported by those measurements, not a proof.

**Convergence of gradient descent.** On the training data the loss gap shrinks linearly, by about 0.9984 per iteration (0.99831-0.99847 across windows), taking 18,730 iterations. The Hessian at the solution has smallest eigenvalue 0.0023 and about 10 of 31 eigenvalues near the penalty level, which matches the slow convergence in order of magnitude. The slowest-mode rate predicted from it (0.9986) was not observed exactly.

## Benchmark: repeated 5-fold cross-validation, 3 repeats

Number of trees chosen by early stopping on an inner validation split in each training fold. Intervals use the corrected resampled t-test (Nadeau and Bengio).

**Breast cancer (classification)**

| Model | Log-loss | Accuracy | AUC |
|---|---|---|---|
| Prior (majority class) | 0.660 | 0.627 | 0.500 |
| Logistic regression (mine) | 0.076 | 0.975 | 0.995 |
| Decision tree, depth 3 (mine) | 0.323 | 0.923 | 0.928 |
| Gradient boosting (mine) | 0.119 | 0.960 | 0.991 |
| Gradient boosting (scikit-learn, 100 trees) | 0.116 | 0.960 | 0.991 |

**Diabetes (regression)**

| Model | RMSE |
|---|---|
| Mean baseline | 77.0 |
| Linear regression (scikit-learn) | 54.6 |
| Decision tree, depth 3 (mine) | 61.1 |
| Gradient boosting (mine) | 57.4 |
| Gradient boosting (scikit-learn, 100 trees) | 58.1 |

Paired differences (95% interval): boosting minus logistic regression on log-loss, +0.044 [0.0004, 0.087], which only just excludes zero; on AUC, -0.004 [-0.011, 0.003]. On diabetes, boosting minus linear regression is +2.8 RMSE [-0.5, 6.1], so the two cannot be separated. My boosting versus scikit-learn's shows no detectable difference on either dataset.

## When does boosting win? A synthetic experiment

Signal = $\sqrt{1-a}\cdot$(linear part) $+ \sqrt{a}\cdot$(nonlinear part) + noise (sd 0.5, so the best possible RMSE is 0.5). The nonlinear part (a product of two features, a symmetric band, and an even cosine) has zero linear correlation with every feature. 400 training rows, 3,000 test rows, 10 independent datasets per setting.

| Nonlinear share $a$ | Linear RMSE | Boosting RMSE | Boosting minus linear (95% CI) |
|---|---|---|---|
| 0.0 | 0.504 | 0.564 | +0.059 [0.052, 0.067] |
| 0.2 | 0.679 | 0.628 | -0.052 [-0.060, -0.043] |
| 0.4 | 0.820 | 0.652 | -0.168 [-0.181, -0.155] |
| 0.6 | 0.932 | 0.676 | -0.257 [-0.269, -0.244] |
| 0.8 | 1.041 | 0.670 | -0.371 [-0.383, -0.359] |
| 1.0 | 1.130 | 0.650 | -0.480 [-0.501, -0.458] |

![Crossover experiment](reports/figures/fig4_crossover.png)

## Limitations

- **Small datasets.** Breast cancer has 569 rows and diabetes 442, so differences between models are noisy.
- **The synthetic design favours boosting.** The nonlinear terms were chosen so a linear model cannot capture them at all. Real data usually contains nonlinearity that a linear model partly captures, so the crossover point there could be later.
- **Fixed settings.** Only the number of trees was tuned. Learning rate, depth, the 200-tree cap, noise level and sample size were fixed, and early stopping often chose a number close to the cap in the synthetic runs.
- **Approximate intervals.** Fold sizes in the interval formula are approximate, and folds from repeated cross-validation are not independent.
- **Probabilities were clipped** to [0.001, 0.999] for all models so log-loss is comparable.
- **Numbers can shift slightly** between library versions (small differences were seen between two machines).
- **Not production code.** The implementations are for learning and are far slower than scikit-learn.

## Project layout

```
src/            logistic.py, tree.py, boosting.py
01_baselines.ipynb            baselines and validation against scikit-learn 
02_benchmark.ipynb            repeated cross-validation benchmark 
03_synthetic_crossover.ipynb  nonlinearity crossover experiment
reports/        figures and result tables
requirements.txt
```

Run the notebooks from the project folder with `numpy scipy pandas scikit-learn matplotlib` installed. Each notebook cell is self-contained.

## Data and credits

Real datasets: the breast cancer (Wisconsin diagnostic) and diabetes datasets bundled with scikit-learn. 
