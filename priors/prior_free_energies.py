import numpy as np
from tramp.priors import get_prior
from tramp.experiments import run_experiments, qplot


def run_prior(ax, prior_type, **prior_kwargs):
    prior = get_prior(size=1, prior_type=prior_type, **prior_kwargs)
    A = prior.free_energy(ax=ax)
    vx = prior.compute_forward_error(ax=ax)
    tau_x = prior.second_moment()
    mx = tau_x - vx
    I = 0.5 * tau_x * ax - A
    A_dual = 0.5 * ax * mx - A
    I_dual = I - 0.5 * ax * vx
    return dict(A=A, A_dual=A_dual, I_dual=I_dual, I=I, tau_x=tau_x, vx=vx, mx=mx)


if __name__ == "__main__":
    df1 = run_experiments(
        run_prior, ax=np.linspace(0, 10, 100),
        prior_type="gauss_bernouilli", rho=[0.3, 0.5, 0.7]
    )
    df2 = run_experiments(
        run_prior, ax=np.linspace(0, 10, 100),
        prior_type="binary", p_pos=[0.5, 0.8, 0.9]
    )
    df3 = run_experiments(
        run_prior, ax=np.linspace(0,10,100),
        prior_type="gaussian", mean_prior=[0, 1]
    )
    df = df1.append(
        df2, ignore_index=True, sort=False
    ).append(df3, ignore_index=True, sort=False)
    df.to_csv("prior_free_energies.csv", index=False)
