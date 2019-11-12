import numpy as np
import pandas as pd
from tramp.channels import LinearChannel
from tramp.ensembles import GaussianEnsemble
from tramp.algos.metrics import mean_squared_error, overlap
from tramp.experiments import save_experiments


def run_posterior(N, alpha, v_noise, r0, v0, **kwargs):
    M = int(alpha*N)
    W = GaussianEnsemble(M, N).generate()
    channel = LinearChannel(W)
    tau_z = r0**2 + v0
    tau_x = channel.second_moment(tau_z)
    z_true = r0 + np.sqrt(v0)*np.random.randn(N)
    x_true = W @ z_true
    y = x_true + np.sqrt(v_noise)*np.random.randn(M)
    az = 1/v0
    bz = r0 * np.ones(N)/v0
    ax = 1/v_noise
    bx = y/v_noise
    rz, vz = channel.compute_backward_posterior(az, bz, ax, bx)
    rx, vx = channel.compute_forward_posterior(az, bz, ax, bx)
    mz = tau_z - vz
    mx = tau_x - vx
    result = dict(N=N, M=M, tau_z=tau_z, tau_x=tau_x, mz=mz, mx=mx, vz=vz, vx=vx)
    result["mse_z"] = mean_squared_error(rz, z_true)
    result["mse_x"] = mean_squared_error(rx, x_true)
    result["overlap_z"] = overlap(rz, z_true)
    result["overlap_x"] = overlap(rx, x_true)
    return result


def run_error(N, alpha, ax, az):
    M = int(alpha*N)
    W = GaussianEnsemble(M, N).generate()
    channel = LinearChannel(W)
    n_eff = channel.compute_n_eff(az, ax)
    vz = channel.compute_backward_variance(az, ax)
    vx = channel.compute_forward_variance(az, ax)
    result = dict(
        vz=vz, vx=vx,
        vz_scaled=az*vz, vx_scaled=ax*vx,
        n_eff=n_eff
    )
    return result


if __name__ == "__main__":
    save_experiments(
        run_posterior, "channel_linear_posterior.csv",
        N=1000, r0=0, v0=1,
        alpha=10**np.linspace(-2, 1, 51),
        v_noise=[0.1, 0.3, 1.],
        replicate=np.arange(20)
    )
    save_experiments(
        run_error, "channel_linear_error_wrt_alpha.csv",
        N=500, alpha=10**np.linspace(-2, 1., 51), az=1., ax=[0.3, 1., 10.]
    )
    save_experiments(
        run_error, "channel_linear_error_wrt_ax.csv",
        N=500, alpha=[0.3, 1.0, 3.], az=1., ax=10**np.linspace(-3, 3, 51)
    )
