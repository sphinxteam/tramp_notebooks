import numpy as np
from tramp.likelihoods import AbsLikelihood
from tramp.experiments import save_experiments


def run_error(tau, v_scaled):
    v = v_scaled * tau
    a = 1/v
    if (a * tau - 1 < 0):
        return dict(a=a, v=v)
    s = np.sqrt(a * (a * tau - 1))
    likelihood = AbsLikelihood(np.array([1]))
    error = likelihood.compute_backward_error(a, tau)
    error_scaled = error/tau
    return dict(a=a, v=v, s=s, error_scaled=error_scaled, error=error)


if __name__ == "__main__":
    save_experiments(
        run_error, "likelihood_abs_error.csv",
        tau=[0.1, 1.0, 10.], v_scaled=np.linspace(0, 1, 51)[1:]
    )
