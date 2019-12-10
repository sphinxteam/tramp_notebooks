import numpy as np
import matplotlib.pyplot as plt
from VAE_inpainting_denoising.model_prior_real_data import Model_Prior


def run_demo(model='inpainting', data='mnist',category=0, n_rem=1, Delta=0.01,
             seed=0, max_iter=1000,
             plot_prior_sample=False, plot_truth_vs_pred=False, 
             plot_evolution=False, save_fig=False):
    ## Choose Model ##
    if model == 'inpainting':
        model_params = {'name': 'inpainting',
                        'N': 784, 'N_rem': 28 * int(n_rem/100 * 28), 'alpha': 0, 'type': 'band'}
    elif model == 'denoising':
        model_params = {'name': 'denoising',
                        'N': 784, 'alpha': 0}
    else:
        print('Models avalable for demonstration: [inpainting, denoising]')
        raise NotImplementedError

    ## Choose Data ##
    if data in ['mnist', 'fashion_mnist']:
        data_params = {'name': data, 'category': category}
        prior_params = {'name': 'VAE', 'type': data,
                        'id': '20_relu_400_sigmoid_784_bias'}
    else:
        print('Dataset available for demonstration: [mnist, fashion_mnist]')
        raise NotImplementedError

    ## Run EP ##
    EP = Model_Prior(model_params=model_params, data_params=data_params,
                     prior_params=prior_params, Delta=Delta, seed=seed,
                     plot_prior_sample=plot_prior_sample, plot_truth_vs_pred=plot_truth_vs_pred)
    EP.setup()
    output, output_tracking = EP.run_ep(
        max_iter=max_iter, check_decreasing=False)
    mse_ep, mse = EP.compute_mse(output)
    EP.plot_truth_vs_prediction(x_pred=EP.x_pred['x'], save_fig=save_fig, block=False)

    ## Plot evolution during training
    if plot_evolution:
        print(len(output_tracking))
        for i in range(len(output_tracking)):
            if i % int(max_iter/10) == 0:
                x_pred = output_tracking.iloc[i]['r']
                EP.plot_truth_vs_prediction(x_pred=output_tracking.iloc[i]['r'], save_fig=save_fig, block=False)



if __name__ == '__main__':
    run_demo(model='inpainting', data='mnist', max_iter=1000, n_rem=20, Delta=2, seed=2, 
            plot_truth_vs_pred=True, plot_evolution=True)
