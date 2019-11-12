import numpy as np, matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from tramp.priors import GaussianPrior, get_prior
from tramp.channels import GaussianChannel
from tramp.variables import SISOVariable as V, SILeafVariable as O
from tramp.experiments import save_experiments


def check_highdim(prior_type, N, ax, plot=False):
    # generate instance
    prior = get_prior(size=N, prior_type=prior_type)
    teacher = (
        prior @ V(id="x") @ GaussianChannel(var=1/ax) @ O(id="rx")
    ).to_model()
    sample = teacher.sample()
    bx = ax*sample["rx"]
    # second moment
    tau_x = prior.second_moment()
    tau_x_emp = (sample["x"]**2).mean()
    # single instance
    rx_hat, vx_hat = prior.compute_forward_posterior(ax, bx)
    mx_hat = (sample["x"]*rx_hat).mean()
    qx_hat = (rx_hat**2).mean()
    mse_x = np.mean((sample["x"]-rx_hat)**2)
    A_hat = prior.compute_log_partition(ax, bx) / N
    if plot:
        plt.plot(np.arange(N), sample["x"], label="x true")
        plt.plot(np.arange(N), rx_hat, label="x pred")
        plt.legend()
    # average
    vx_avg = prior.compute_forward_error(ax)
    mx_avg = tau_x - vx_avg
    A_avg = prior.compute_free_energy(ax)
    return dict(
        tau_x=tau_x, tau_x_emp=tau_x_emp,
        vx_hat=vx_hat, A_hat=A_hat, mx_hat=mx_hat, qx_hat=qx_hat, mse_x=mse_x,
        vx_avg=vx_avg, A_avg=A_avg, mx_avg=mx_avg
    )

def on_progress(i, total):
    print(f"{i}/{total}")

if __name__=="__main__":
    save_experiments(
        check_highdim, "check_highdim.csv",
        prior_type=["gaussian", "gauss_bernouilli", "binary"],
        N=10000, ax=np.linspace(1, 10, 50)[1:],
        on_progress=on_progress
    )
