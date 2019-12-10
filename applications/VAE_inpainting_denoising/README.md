# tramp_demo_vae
## Demonstration of the TRAMP package on real data-set

This [notebook](Demo_TRAMP.ipynb) illustrates the <a href="https://github.com/sphinxteam/tramp"> TRAMP package</a> on simple models: denoising and inpainting of MNIST or Fashion-MNIST samples, using a VAE prior trained on the same data-set. <br/>

For demonstration we use a simple <b> VAE prior </b>: ([Dense(20, 400) + bias -> relu  -> Dense(400, 784) + bias -> sigmoid ]). <a href='https://keras.io/examples/variational_autoencoder/'> See details </a>


## Requirements
* numpy / pandas / scipy / matplotlib
* networkx==1.11
* daft

## Details
* <b>model</b>: 
    * 'inpainting': $n_{rem}$ = percentage of the image deleted
    * 'denoising': $\Delta$ = variance of the additive noise
* <b>data</b> 
    * 'mnist'
    * 'fashion_mnist'
* <b>category</b> $\in [0:9]$
* <b>max_iter</b>: maximum number of iterations
* <b>seed</b>: None if $0$
* <b> plot_prior_sample</b>: shows prior samples


* Notations: 
    * $x^\star$: drawn from <b> data </b>
    * $x_{\rm obs}$: <b>model</b>($x^\star$)
    * $\hat{x}$: estimator of $x^\star$