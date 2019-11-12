import numpy as np
from tramp.experiments import TeacherStudentScenario, run_experiments 
from tramp.models import GaussianDenoiser

def run_denoiser(N, prior_type, var_noise, **kwargs):
    denoiser = GaussianDenoiser(N, prior_type, var_noise, **kwargs)
    scenario = TeacherStudentScenario(denoiser)
    scenario.setup()
    scenario.infer(max_iter=250)
    result = dict(
        v_se=scenario.v_se["x"], n_iter_se=scenario.n_iter_se,
        v_ep=scenario.v_ep["x"], n_iter_ep=scenario.n_iter_ep,
        mse=scenario.score["mse"]["x"], overlap=scenario.score["overlap"]["x"]
    )
    return result

if __name__=="__main__":
    df1 = run_experiments(
        run_denoiser, N=1000, prior_type="gaussian", 
        var_noise = 10**np.linspace(-2,2,50)
    )
    df2 = run_experiments(
        run_denoiser, N=1000, prior_type="binary", 
        var_noise = 10**np.linspace(-2,2,50), p_pos=[0.25, 0.5]
    )
    df3 = run_experiments( 
        run_denoiser,  N=1000, prior_type="gauss_bernouilli", 
        var_noise = 10**np.linspace(-2,2,50), rho=[0.25, 0.5]
    )
    df = df1.append(df2, ignore_index=True).append(df3, ignore_index=True)
    df.to_csv("denoiser_experiments.csv", index=False)
    