### Standard benchmarks

We gracefully made use of [Durasov et al.](https://github.com/cvlab-epfl/zigzag/tree/main)'s code for standard experiments.

#### Toy Regression Dataset
Uncertainty Estimation in Regression: The objective is to predict $\( y \)$-values for $\( x \)$-data points sampled from the range $\( x \in [-1, 3] \)$, using a third-degree polynomial with added Gaussian noise. 
[notebook](#)

#### Boston Housing dataset
For regression tasks, anomalous examples can still produce outputs that fall within the training range, \(\mathcal Y\). Therefore, we use a dataset where at least one covariate exhibits a clearly multimodal distribution. By intentionally withholding one mode from the model during training, we ensure that $\(P_\text{test}(X) \neq P_\text{train}(X)\)$.
[notebook](#)

#### **MNIST vs FashionMNIST:** 
We train a simple CNN on the MNIST dataset and compare the uncertainty estimates for images from the test sets from MNIST and FashionMNIST. 
[notebook](#)
