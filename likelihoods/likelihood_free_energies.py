import numpy as np
from tramp.likelihoods import get_likelihood
from tramp.experiments import save_experiments, qplot


def run_likelihood(az, tau_z, likelihood_type):
    y = np.array([1])
    likelihood = get_likelihood(y, likelihood_type)
    A = likelihood.free_energy(az, tau_z)
    vz = likelihood.compute_backward_error(az, tau_z)
    mz = tau_z - vz
    I = 0.5 * tau_z * az - A + 0.5 * np.log(2*np.pi*tau_z/np.e)
    A_dual = 0.5 * az * mz - A
    I_dual = I - 0.5 * az * vz
    return dict(A=A, A_dual=A_dual, I_dual=I_dual, I=I, tau_z=tau_z, vz=vz, mz=mz)


if __name__ == "__main__":
    save_experiments(
        run_likelihood, "likelihood_free_energies.csv",
        az=np.linspace(1, 10, 100), tau_z=1,
        likelihood_type=["abs", "sgn", "gaussian"]
    )
