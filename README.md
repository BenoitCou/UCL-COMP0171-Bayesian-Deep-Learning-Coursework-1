# COMP0171 — Bayesian Deep Learning • Coursework 1  
MSc in Machine Learning, University College London (2024 / 25)

This repository contains all the material submitted for **Coursework 1** of the COMP0171 *Bayesian Deep Learning* module.  
It includes:

| Path                                           | Description                                                                                                                                                                                                                            |
| ---------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `(Part 1) Seven scientists.ipynb`              | Jupyter notebook that builds a hierarchical Gaussian model for seven laboratory means, implements a blocked Metropolis-within-Gibbs sampler in PyTorch, and reports posterior diagnostics & summaries.                                |
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

- **Hierarchical Gaussian model** — modelled the seven laboratory means $$\theta_i$$ with a shared latent mean $$\mu$$ and precision $$\tau$$; placed conjugate priors $$ \mu \sim \mathcal N(\mu_0,\sigma_0^2)$$ and $$\tau \sim \text{Gamma}(\alpha,\beta)$$.  
- **Blocked Metropolis-within-Gibbs sampler** — custom PyTorch implementation that jointly proposes $$(\mu,\tau)$$ and individually updates the $$\theta_i$$; proposal scales tuned to reach an acceptance rate ≈ 30 %.  
- **Diagnostics** — trace-plots, running-mean diagnostics and effective‐sample-size estimates to check mixing; burn-in and thinning chosen accordingly.  
- **Posterior summaries** — reported posterior means ± 95 % credible intervals for each scientist’s true mean and for the shared $$\mu$$; predicted the next measurement for Scientist 3.  
- **Extra credit** — treated $$\alpha,\beta$$ as unknown, added a second Gibbs block and recovered their posterior (2 bonus marks).  
- **Short answers** — discussed choice of proposal s.d.s and target acceptance rate; analysed why a fully joint proposal might violate detailed balance without extra care.  

**Part II – Bayesian linear classifiers**

- **Feature spaces** — compared a *linear* feature map $$\phi_\text{lin}(x)=[1,x_1,x_2]$$ and a *quadratic* map adding $$x_1^2,x_2^2,x_1x_2$$.  
- **MAP estimation** — maximised the log-posterior of Bayesian logistic regression with an $$ \mathcal N(0,\sigma_w^2I)$$ prior using LBFGS; verified gradient correctness with finite differences.  
- **Laplace approximation** — computed the Hessian at the MAP, constructed a Gaussian approximation to $$p(\mathbf w\mid\mathcal D)$$, and visualised predictive mean ± std over a grid.  
- **Model evidence** — used the Laplace log-evidence to compare linear vs quadratic models (sign error in one term, noted in feedback).  
- **Custom features** — engineered a cubic/interaction map that achieved the highest evidence among all tested φ; plotted the corresponding decision boundary.  
- **Short answers** — reflected on appropriateness of Laplace in high dimensions, on evidence-vs-dimensionality trade-offs, and on when richer φ pays off.  

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

