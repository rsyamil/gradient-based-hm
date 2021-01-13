# gradient-based-hm

Tutorials for gradient-based history matching for high-fidelity models and also using their reduced representations (with PCA and Autoencoders).

## Prerequisites

The dataset used in this demo repository is the digit-MNIST images, **X**, with the forward model being a linear operator **G** and the resulting simulated responses denoted as **Y**. The physical system is represented as **Y=G(X)** or **D=G(M)**. More description on the dataset is available [here (in readme)](https://github.com/rsyamil/cnn-regression). Here you will find demos for dimensionality reduction of the digit-MNIST images using [autoencoders](https://github.com/rsyamil/dimensionality-reduction-autoencoders) and [PCA](https://github.com/rsyamil/dimensionality-reduction-classic).

![ForwardModel](/readme/forwardmodel.png)

We are interested in learning the inverse mapping **M=G'(D)** which is not trivial if the **M** is non-Gaussian (which is the case with the digit-MNIST images) and G is nonlinear (in this demo we assume a linear operator). Such complex mapping (also known as history-matching) may result in solutions that are non-unique with features that may not be consistent. In gradient-based history-matching, the objective function that we want to minimize is the following, where **d_obs** is the field observation, **m** is our variable of interest and we assume that the forward operator **G** sufficiently represents the linear/nonlinear (i.e. multi-phase fluid flow, heat-diffusion etc) physical systems.

![Eq1](/readme/eq1_loss.png)

## Linear Least-Square Solution

The simple closed-form solution:

![Eq2](/readme/eq2_llsq.png)

As per [this notebook](https://github.com/rsyamil/gradient-based-hm/blob/main/linear-least-square.ipynb), the inversion solution can reproduce the **d_obs** but the solution shows no realism with respect to the set of training models. 

![llsq](/readme/llsq.png)

## Gradient-based History Matching

The gradient for the loss function:

![Eq3](/readme/eq3_grad.png)

The update equation:

![Eq4](/readme/eq4_iter_update.png)

Run the optimization process as per [this notebook](https://github.com/rsyamil/gradient-based-hm/blob/main/gradient-full-dim.ipynb)

![grad_full_dim](/readme/grad_full_dim.png)

The inversion solution also is not satisfactory but still can reproduce the **d_obs**.

![grad_full_dim_comp](/readme/grad_full_dim_comp.png)

## Gradient-based History Matching (with PCA)

The poor inversion solutions we have seen above are caused by the non-Gaussian features in the digit-MNIST dataset. In [this notebook](https://github.com/rsyamil/gradient-based-hm/blob/main/gradient-pca-dim.ipynb), we represent the images as PCA coefficients. Refer to [this tutorial](https://github.com/rsyamil/dimensionality-reduction-classic) on how to do that. 

![Eq5](/readme/eq5_pca_m.png)

Then with chain-rule, the update equation simply becomes:

![Eq6](/readme/eq6_pca_iter_update.png)

The minimization process:

![grad_pca_dim](/readme/grad_pca_dim.png)

Here we see that the realism of the inversion solution is better preserved, at the same time it can also reproduce the **d_obs**.

![grad_pca_dim_comp](/readme/grad_pca_dim_comp.png)

## Gradient-based History Matching (with AE)

We can also use autoencoders for dimensionality reduction as we did [here](https://github.com/rsyamil/dimensionality-reduction-autoencoders).

![Eq7](/readme/eq7_ae_m.png)

Similar to PCA, the update equation then becomes:

![Eq8](/readme/eq8_ae_iter_update.png)

Where the Jacobian (dm/dzm) is obtained from the decoder. See pending [issues](https://github.com/rsyamil/gradient-based-hm/issues).

![grad_ae_dim_comp](/readme/grad_ae_dim_comp.png)




