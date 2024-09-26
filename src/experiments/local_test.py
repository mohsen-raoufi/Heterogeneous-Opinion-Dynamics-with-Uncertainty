import os
import numpy as np

from experiment import ExperimentParameters, run_experiment, NetworkParams
from util import dump_config

if __name__ == "__main__":
    center = 5.0
    width = 5.0
    meas_noise_int_vals = (np.geomspace(1, 11, 20, endpoint=True) - 1) * 0.5
    no_meas_noise_center_vals = [0.0]
    meas_noise_center_vals = np.linspace(2.5, 7.5, 10)
    env_noise_std_vals = np.linspace(0, 10, 10)
    network_param1_vals = (np.geomspace(1, 11, 20, endpoint=True) - 1) * 0.03
    own_weight_vals = np.linspace(0, 1, 20)
    no_own_weight_vals = [0.0]
    agent_num_vals = [100]  # np.linspace(10, 100, 5)       # TODO should we at least sample a few values here?
    no_network_hetero = [0.3]
    no_inf_het = [0.0]
    fixed_env_noise_std_vals = [5.0]
    corr_net_inf_vals = np.linspace(-1.0, 1.0, 11)
    no_corr_inf_het = [0]
    fixed_center_vals = [5.0]
    additional_agent_nums = [25]

    # envstd_0_mnint_0_mncenter_0_np1_0_ow_0_a_n_0_run_64
    envstd = fixed_env_noise_std_vals[0]
    mnint = meas_noise_int_vals[0]
    mncenter = no_meas_noise_center_vals[0]
    np1 = network_param1_vals[[0, 7, 19]]
    ow_vals = own_weight_vals[[1,10]]
    seed = 343432
    for ow_i, ow in enumerate(ow_vals):
        for n_i, n in enumerate(np1): # (np.geomspace(1, 11, 20, endpoint=True) - 1) * 0.5:
            # for k_reg in [20]: # [4, 6, 10]: # for K_reg test
            for agent_type in ["Naive", "NaiveLO", "Bayes"]: # , "BayesCICORR", "NaiveCORR", "NaiveLOCORR"]:
                if agent_type == "Bayes":
                    ow = 0.0
                params = ExperimentParameters(agent_type=agent_type,
                                            steps=10,
                                            environment_type="GaussianFixedSTD",
                                            seed=seed,
                                            
                                            network_params=NetworkParams(network_type="centralized_random_fixed_mdeg", # "centralized_random_fixed_mdeg", # "k_regular", # "centralized_random_fixed_mdeg",
                                                                        n_agents=25,
                                                                        min_agent_measurement_noise=max(mncenter-0.5 * mnint, 0.0001),
                                                                        max_agent_measurement_noise=mncenter+0.5 * mnint,
                                                                        scalar_param1=n, #0.4,
                                                                        correlation_network_information=0,
                                                                        weight_own_belief=ow,
                                                                        weight_others_belief=1-ow))
                path = "/home/mohsen/Project/colab/collective-decison-making-with-direl/results/test_OB/"+params.agent_type+"_np_"+str(n_i)+"_ow_"+str(ow_i)
                os.makedirs(path, exist_ok=True)
                dump_config(params, os.path.join(path, "config"))
                run_experiment(params=params, file_path=path)
