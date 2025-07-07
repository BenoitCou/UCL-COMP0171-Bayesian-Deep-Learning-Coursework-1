# COMP0171 — Bayesian Deep Learning • Coursework 1  
MSc in Machine Learning, University College London (2024 / 25)

This repository contains all the material submitted for **Coursework 1** of the COMP0171 *Bayesian Deep Learning* module.  
It includes:

| Path                                           | Description                                                                                                                                                                                                                            |
| ---------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `(Part 1) Seven scientists.ipynb`              | Jupyter notebook that builds a Gaussian model for seven laboratory means, implements an MCMC sampler in PyTorch, and reports posterior diagnostics & summaries.                                |
| `(Part 2) Bayesian classifiers.ipynb`          | Jupyter notebook that trains Bayesian logistic regressors with linear/quadratic/cubic feature maps, applies the Laplace approximation, computes model evidence, and visualises predictive uncertainty.                               |
| `data.pt`                                      | Torch tensor dataset used in Part 2.                                                                                                                                                                                                  |
| `requirements.txt`                             | Python dependencies (PyTorch ≥ 2.1, numpy, matplotlib, jupyter, arviz, tqdm).                                                                                                                                                          |

---

## Quick-start

```bash
# 1. Clone the repo
git clone https://github.com/BenoitCou/UCL-COMP0171-Bayesian-Deep-Learning-Coursework-1
cd UCL-COMP0171-Bayesian-Deep-Learning-Coursework-1

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Launch the notebooks
jupyter notebook "(Part 1) Seven scientists.ipynb"
jupyter notebook "(Part 2) Bayesian classifiers.ipynb"
```

## Coursework Overview  

A concise tour of what you will find — and reproduce — in this repository:

**Part I – Seven scientists**

- **Heteroskedastic Gaussian model** — modelled each scientist’s measurement as  
  $$x_i \sim \mathcal{N}(\mu, \sigma_i^2)$$  
  with unknown true mean $$\mu$$ and individual noise levels $$\sigma_i$$. Priors used were  
  $$\mu \sim \mathcal{N}(0, \alpha^2), \quad \sigma_i \sim \text{Exp}(\beta)$$  
  with $$\alpha = 50$$ and $$\beta = 0.5$$.
- **Custom MCMC sampler** — implemented Metropolis-Hastings in PyTorch to sample jointly from $$p(\mu, \sigma_1,\dots,\sigma_7 | x)$$ using Normal random-walk proposals. All 8 parameters were updated simultaneously in a single accept/reject step.  
- **Diagnostics** — monitored trace plots and histograms for $$\mu$$ and $$\sigma_i$$; acceptance rate reported (≈ 7%) and burn-in appropriately discarded. Posterior boxplots helped visualize which scientists had higher uncertainty.
- **Posterior summaries** — estimated $$\mathbb{E}[\mu] \approx 9.79$$ and computed posterior probability $$\Pr(\mu < 9) \approx 3.8\%$$ using post-burn-in MCMC samples.
- **Alternate proposal strategy** — suggested a sequential update approach where $$\mu$$ and each $$\sigma_i$$ can be proposed and accepted independently, potentially improving acceptance rate and convergence speed. Prototype `mcmc_step_improved` function included.
- **Short answers** — discussed benefits of symmetric Normal proposals for local exploration and detailed the tradeoffs between joint vs. coordinate-wise proposals in MCMC, including a practical implementation of partial updates to avoid rejecting otherwise good partial proposals.


**Part II – Bayesian linear classifiers**

- **Bayesian logistic regression model** — modelled binary labels using a logistic link:
  $$\mathbf{w} \sim \mathcal{N}(0, \sigma^2 I), \quad y_i \sim \text{Bernoulli}(\text{Logistic}(\mathbf{w}^\top \phi(\mathbf{x}_i)))$$  
  with two feature choices:  
  - Simple: $$\phi(\mathbf{x}) = [1, x_1, x_2]$$  
  - Quadratic: $$\phi(\mathbf{x}) = [1, x_1, x_2, x_1x_2, x_1^2, x_2^2]$$  
  Custom cubic features were also implemented.
- **MAP estimation** — used PyTorch autograd and `Adagrad` optimizer to maximize the log joint distribution and recover the mode $$\mathbf{w}_{MAP}$$. Training curves tracked loss over iterations for convergence diagnostics.
- **Laplace approximation** — estimated the posterior covariance as the inverse negative Hessian at $$\mathbf{w}_{MAP}$$; used this Gaussian approximation to compute predictive probabilities and marginal likelihood (model evidence).
- **Model comparison** — computed model evidence via Laplace approximation:
  $$\log p(y | X) \approx \log p(y, \mathbf{w}_{MAP}) - \frac{1}{2} \log|\Sigma| + \frac{D}{2} \log(2\pi)$$  
  Compared feature sets based on evidence and test accuracy to assess overfitting vs underfitting.
- **Custom feature design** — engineered a feature set with polynomial terms up to degree 3:
  $$\phi(x) = [1, x_1, x_2, x_1^2, x_2^2, x_1^3, x_2^3, x_1 x_2]$$  
  Achieved test accuracy of 92% and better log-evidence than baseline models.

- **Short answers** — discussed metrics for model comparison (accuracy vs evidence), how Laplace approximation mitigates overfitting through posterior uncertainty, and how feature engineering affects generalization.



---

## Marks obtained  

**Grade**: **94 / 100**

| Part | Score | Lecturer’s feedback |
| ---- | ----- | ------------------------------ |
| Part 1 | 13 / 14 | Generally good implementation. For short answer, a couple points. A minor one: I think that maybe it would be worth thinking a bit more about the proposal standard deviations and how to set them (what is a good target acceptance rate?). For part 2, this is an interesting idea but I think you would want to be careful to demonstrate that this still has the correct target distribution (e.g. making sure it still satisfies detailed balance). Was looking also for some comments on when this approach might be appropriate (e.g. what sorts of target distributions, dimensionality, etc). |
| Part 2 | 20 / 21 | Sign error in model evidence implementation; otherwise good implementation and short answer. |

---

## Repository structure  

```kotlin
UCL-COMP0171-Bayesian-Deep-Learning-CW1/
├── (Part 1) Seven scientists.ipynb       # Hierarchical MCMC implementation & analysis
├── (Part 2) Bayesian classifiers.ipynb   # Bayesian logistic regression, Laplace, evidence
├── data.pt                               # Torch tensor dataset used in Part 2
└── requirements.txt                      # Python dependencies

