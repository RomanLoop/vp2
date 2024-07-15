# Portfolio Optimization with Physics-Inspired Graph Neural Networks
## A Master's Thesis Specialization Project

The main objective of this specialization project is to apply the approach of Schuetz et al. to the S&P 500 assets, constructing a portfolio of uncorrelated assets and comparing its volatility and performance with benchmarks.

## Introduction
A well-known optimization problem in finance is the risk diversification of a portfolio. The objective is to construct a portfolio from a subset of 𝑛 assets selected from a large universe of 𝑁 assets, aiming to exhibit the lowest possible risk. These minimum risk portfolios are particularly valuable to relatively risk-averse investors. In practice, the universe of assets can be very large.

The 𝑁 assets in the universe can be modeled as a graph, where the assets are represented by vertices, and their relationships are derived from the correlations between them. These correlations are pivotal in finding minimum risk portfolios, as they determine the overall portfolio risk. Ideally, the minimum risk portfolio would consist of uncorrelated assets.

When constructing low-risk portfolios, one may aim to find the largest possible set of uncorrelated assets. Computationally, this poses a major challenge as the set of possible solution candidates grows exponentially with the number of nodes. This makes a brute force approach infeasible for large sets. While there are approximation algorithms, none of them scale effectively to graphs with thousands or even hundreds of thousands of nodes.

Schuetz et al. developed a self-supervised deep learning approach to approximate the MIS on large graphs. Their graph neural network-based approach claims to be very fast and accurately. The promising solution runtime of $~n^{1.7}$ scales very well compared to the solution runtime of the Boppana-Halldorsson algorithm – a state of the art algorithm for the MIS problem – with a solution runtime of $~n^{2.9}$
The main goal of this project is to apply the approach of Schuetz et al. the S&P500 assets. Building a portfolio of uncorrelated assets and compare its performance with benchmarks.

## Documentation 
A detailed documentation of the project and all results can be found in `docs/TechnicalReport_VP2_GNN-MIS_RomanLoop_V2.pdf`
The original paper *"Combinatorial Optimization with Physics-Inspired Graph Neural Networks"* from Schuetz et al. can be found on [arXiv](https://arxiv.org/pdf/2107.01188.pdf). 
The code of the authors is available on [Github](https://github.com/amazon-science/co-with-gnns-example).

The main code (end-to-end process) can be found in the `backtest.ipynb` notebook. All code regarding hyperparameter optimization is contained in `hyperparameter_optimization_my.ipynb` and `hyperparameter_optimization_raytune.ipynb`. Helper functions and reusable code can be found in `utils` pyhton files.

## Python setup
Please note I have provided a `requirements.txt` file, which defines the environment required to run this code. The code has been run with Python 3.9. and is not tested with any other Python version. Thus, I suggest to create a virtual environment with Python 3.9:
```python -m venv .venv```
