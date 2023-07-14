## Portfolio Optimization with Physics-Inspired Graph Neural Networks

The main objective of this spezalication project is to apply the approach of Schuetz et al. to the S&P500 assets to construct a portfolio of uncorrelated assets and compare its volatility and performance with benchmarks.

## Introduction

A well-known optimization problem in finance is risk diversification of a portfolio. The optimization objective is to construct a portfolio, whose constituents consist of a subset of n assets selected from a large universe of N assets, exhibiting the lowest possible risk. These minimum risk portfolios are of great interest to investors that are relatively risk averse. In practice, the universe can  potentially be very large. The N assets in the universe can be modelled as a graph in which the assets are represented by the vertices and their relationships are derived from the correlations between the assets. The correlations play a pivotal role in finding minimum risk portfolios since the overall portfolio risk is determined by them.. In the ideal case, the minimum risk portfolio contains assets that are uncorrelated.

When constructing low risk portfolios, one may strive for finding the largest possible set of uncorrelated assets. From a computational point of view this poses a major challenge as the set of possible solution candidates grows exponentially with the number of nodes. This implies that a brute force approach is infeasible for large sets. There are approximation algorithms, but none of them scale to graphs with thousands or even hundreds of thousands of nodes. 

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
