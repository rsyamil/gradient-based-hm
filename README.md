# gradient-based-hm

Tutorials for gradient-based history matching for high-fidelity models and also using their reduced representations (with PCA and Autoencoders).

## Prerequisites

The dataset used in this demo repository is the digit-MNIST images, **X**, with the forward model being a linear operator **G** and the resulting simulated responses denoted as **Y**. The physical system is represented as **Y=G(X)** or **D=G(M)**. More description on the dataset is available [here (in readme)](https://github.com/rsyamil/cnn-regression). Here you will find demos for dimensionality reduction of the digit-MNIST images using [autoencoders](https://github.com/rsyamil/dimensionality-reduction-autoencoders) and [PCA](https://github.com/rsyamil/dimensionality-reduction-classic).

![ForwardModel](/readme/forwardmodel.png)

We are interested in learning the inverse mapping **M=G'(D)** which is not trivial if the **M** is non-Gaussian (which is the case with the digit-MNIST images) and G is nonlinear (in this demo we assume a linear operator). Such complex mapping (also known as history-matching) may result in solutions that are non-unique with features that may not be consistent. In gradient-based history-matching, the objective function that we want to minimize is the following, where **d_obs** is the field observation, **m** is our variable of interest and we assume that the forward operator **G** sufficiently represents the linear/nonlinear (i.e. multi-phase fluid flow, heat-diffusion etc) physical systems.

![Eq1](/readme/eq1_loss.png)

## Linear Least-Square Solution

![Eq2](/readme/eq2_llsq.png)

![llsq](/readme/llsq.png)

## Gradient-based History Matching

![Eq3](/readme/eq3_grad.png)

![Eq4](/readme/eq4_iter_update.png)

![grad_full_dim](/readme/grad_full_dim.png)

![grad_full_dim_comp](/readme/grad_full_dim_comp.png)

## Gradient-based History Matching (with PCA)

![Eq5](/readme/eq5_pca_m.png)

![Eq6](/readme/eq6_pca_iter_update.png)

![grad_pca_dim](/readme/grad_pca_dim.png)

![grad_pca_dim_comp](/readme/grad_pca_dim_comp.png)

## Gradient-based History Matching (with AE)


