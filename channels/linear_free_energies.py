import numpy as np
from tramp.channels import LinearChannel, MarchenkoPasturChannel
from tramp.ensembles import GaussianEnsemble
from tramp.experiments import save_experiments, qplot


M, N = 2000, 1000
alpha = M/N
F = GaussianEnsemble(M, N).generate()
channels = {
    "empirical": LinearChannel(F), 
    "marchenko": MarchenkoPasturChannel(alpha)
}
sources=["empirical", "marchenko"]

def run_linear(source, az, ax, tau_z):
    channel = channels[source]
    alpha = channel.alpha
    A = channel.free_energy(az, ax, tau_z)
    vz = channel.compute_backward_error(az, ax, tau_z)
    vx = channel.compute_forward_error(az, ax, tau_z)
    tau_x = channel.second_moment(tau_z)
    mz = tau_z - vz
    mx = tau_x - vx
    I = 0.5 * (tau_z * az + alpha * tau_x * ax) - A + 0.5 * np.log(2*np.pi*tau_z/np.e)
    A_dual = 0.5 * (az * mz + alpha * ax * mx) - A
    I_dual = I - 0.5 * (az * vz + alpha * ax * vx)
    return dict(
        alpha = alpha, A=A, A_dual=A_dual, I=I, I_dual=I_dual,
        tau_z=tau_z, vz=vz, mz=mz, tau_x=tau_x, vx=vx, mx=mx
    )


if __name__ == "__main__":
    save_experiments(
        run_linear, "linear_free_energies.csv",
        source=sources, az=10**np.linspace(0, 1, 21), ax=10**np.linspace(-1, 1, 21), tau_z=1
    )
    save_experiments(
        run_linear, "linear_free_energies_wrt_ax.csv",
        source=sources, az=[1, 2], ax=np.linspace(0, 4, 51)[1:], tau_z=1
    )
    save_experiments(
        run_linear, "linear_free_energies_wrt_az.csv",
        source=sources, ax=[1, 2], az=np.linspace(1, 4, 51)[1:], tau_z=1
    )
