import subprocess
from experiment import ExperimentParameters
from models.collective import NetworkParams
from util import dump_config
import os
import datetime
import numpy as np


def run_certainty_centrality_grid_search(file_path, agent_type, meas_noise_int_vals, meas_noise_center_vals,
                                         network_param1_vals, agent_num_vals,
                                         env_noise_std_vals, own_weight_vals, corr_inf_het_vals,
                                         network_type, environtment_type,
                                         use_meas_noise_center=False):
    print("Creating Defintion Files...\n")
    name = file_path.split("/")[-1]
    os.makedirs(os.path.join(file_path, "results"))
    slurm_output_path = os.path.join(file_path, "runs")
    os.makedirs(slurm_output_path)
    job_file_path = slurm_output_path + "/job.bash"
    f = open(job_file_path, "w")
    f.write("#!/bin/bash\n")
    f.write("#SBATCH --job-name=scioi_colab_27_35_"+str(name)+"\n")
    f.write("#SBATCH --output=" + slurm_output_path + "/output.txt # output file\n")
    f.write("#SBATCH --error=" + slurm_output_path + "/error.txt # output file\n")
    f.write("#SBATCH --partition=ex_scioi_node # partition to submit to\n")
    f.write("#SBATCH -a 1-100\n")
    f.write("#SBATCH --ntasks-per-core=1\n")
    f.write("#SBATCH --time=10-00:00 # Runtime in D-HH:MM\n")
    f.write("#SBATCH --mem-per-cpu=2000 # memory in MB per cpu allocated\n")
    f.write("source $HOME/miniconda3/bin/activate\n")
    f.write("conda activate domip\n")
    f.write("export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/miniconda3/lib/\n")

    for meas_noise_int_idx, meas_noise_int in enumerate(meas_noise_int_vals):
        for meas_noise_center_idx, meas_noise_center in enumerate(meas_noise_center_vals):
            for network_param1_idx, network_param1 in enumerate(network_param1_vals):
                for agent_n_idx, agent_n in enumerate(agent_num_vals):
                    for env_noise_std_idx, env_noise_std in enumerate(env_noise_std_vals):
                        for own_weight_val_idx, own_weight_val in enumerate(own_weight_vals):
                            for corr_inf_het_idx, corr_inf_het_val in enumerate(corr_inf_het_vals):
                                run_name = "envstd_" + str(env_noise_std_idx) + "_mnint_" + str(
                                    meas_noise_int_idx) + "_mncenter_" + str(meas_noise_center_idx) + "_np1_" + str(
                                    network_param1_idx) + "_ow_" + str(own_weight_val_idx) + "_a_n_" + str(
                                    agent_n_idx) + "_corr_"+str(corr_inf_het_idx)+"_run_"
                                local_path = os.path.join(file_path, "results", run_name)
                                for i in range(1, 101):
                                    used_env_std_noise = max(env_noise_std, 0.0001)
                                    if not use_meas_noise_center:
                                        meas_noise_min = used_env_std_noise
                                    else:
                                        meas_noise_min = meas_noise_center - 0.5 * meas_noise_int
                                    meas_noise_min = max(meas_noise_min, 0.0001)
                                    meas_noise_max = meas_noise_min + meas_noise_int
                                    os.makedirs(local_path+str(i))
                                    exp_params = ExperimentParameters(run_name=run_name+str(i),
                                                                      visualize=False, seed=i,
                                                                      env_noise_std=used_env_std_noise,
                                                                      agent_type=agent_type,
                                                                      environment_type=environtment_type,
                                                                      network_params=NetworkParams(
                                                                          scalar_param1=network_param1,
                                                                          max_agent_measurement_noise=meas_noise_max,
                                                                          min_agent_measurement_noise=meas_noise_min,
                                                                          n_agents=int(agent_n),
                                                                          weight_own_belief=own_weight_val,
                                                                          weight_others_belief=1.0 - own_weight_val,
                                                                          network_type=network_type,
                                                                          correlation_network_information=corr_inf_het_val,
                                                                      )
                                                                      )
                                    dump_config(exp_params, local_path + str(i) + "/config")
                                f.write(
                                    "python $HOME/colab/collective-decison-making-with-direl/src/experiment.py " +
                                    local_path + "${SLURM_ARRAY_TASK_ID}\n")
    f.write("exit\n")
    f.close()
    print("Definition Files Created\n")
    print("Starting Job Array\n")
    subprocess.Popen(["sbatch", job_file_path])
    print("Done\n")


if __name__ == "__main__":
    # meas_noise_int_vals = np.linspace(0, 5, 10)
    meas_noise_int_vals = (np.geomspace(1, 11, 20, endpoint=True) - 1) * 0.5
    no_meas_noise_center_vals = [0.0]
    meas_noise_center_vals = np.linspace(2.5, 7.5, 10)
    env_noise_std_vals = np.linspace(0, 10, 10)
    # network_param1_vals = np.linspace(0, 0.3, 10)
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
    additional_agent_nums = [25, 1000, 10000]

    # ###################################################################################################################
    # # These are for network heterogeneity
    # no_meas_int_vals = [0.0]
    #
    # # Normal Naive Grid Search
    # run_certainty_centrality_grid_search(
    #     "../results/N" + str(agent_num_vals[0]) + "_" + datetime.datetime.now().strftime(
    #         "%Y-%m-%d-%H-%M-%S") + "_network_search_Naive_centralized_random_fixed_mdeg",
    #     "Naive", no_meas_int_vals, no_meas_noise_center_vals, network_param1_vals, agent_num_vals,
    #     env_noise_std_vals, own_weight_vals,
    #     "centralized_random_fixed_mdeg", environtment_type="GaussianFixedSTD")
    #
    # # Normal Bayes Grid Search
    # run_certainty_centrality_grid_search(
    #     "../results/N" + str(agent_num_vals[0]) + "_" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    #     + "_network_search_Bayes_centralized_random_fixed_mdeg", "Bayes",
    #     no_meas_int_vals, no_meas_noise_center_vals,
    #     network_param1_vals, agent_num_vals, env_noise_std_vals, no_own_weight_vals,
    #     "centralized_random_fixed_mdeg", environtment_type="GaussianFixedSTD")
    #
    # # Normal BayesCI Grid Search
    # run_certainty_centrality_grid_search(
    #     "../results/N" + str(agent_num_vals[0]) + "_" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    #     + "_network_search_BayesCI_centralized_random_fixed_mdeg", "BayesCI",
    #     # + "_test_grid_search_Bayes_spatial_random", "Bayes",
    #     no_meas_int_vals, no_meas_noise_center_vals,
    #     network_param1_vals, agent_num_vals, env_noise_std_vals, no_own_weight_vals,
    #     "centralized_random_fixed_mdeg", environtment_type="GaussianFixedSTD")
    #
    # ###################################################################################################################
    # # These are for certainty heterogeneity
    #
    # # Normal Naive Grid Search
    # run_certainty_centrality_grid_search(
    #     "../results/N" + str(agent_num_vals[0]) + "_" + datetime.datetime.now().strftime(
    #         "%Y-%m-%d-%H-%M-%S") + "_certainty_search_Naive_centralized_random_fixed_mdeg",
    #     "Naive", meas_noise_int_vals, meas_noise_center_vals, no_network_hetero, agent_num_vals,
    #     fixed_env_noise_std_vals, own_weight_vals,
    #     "centralized_random_fixed_mdeg", environtment_type="GaussianExternalSTD", use_meas_noise_center=True)
    #
    # # Normal Bayes Grid Search
    # run_certainty_centrality_grid_search(
    #     "../results/N" + str(agent_num_vals[0]) + "_" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    #     + "_certainty_search_Bayes_centralized_random_fixed_mdeg", "Bayes",
    #     meas_noise_int_vals, meas_noise_center_vals,
    #     no_network_hetero, agent_num_vals, fixed_env_noise_std_vals, no_own_weight_vals,
    #     "centralized_random_fixed_mdeg", environtment_type="GaussianExternalSTD", use_meas_noise_center=True)
    #
    # # Normal BayesCI Grid Search
    # run_certainty_centrality_grid_search(
    #     "../results/N" + str(agent_num_vals[0]) + "_" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    #     + "_certainty_search_BayesCI_centralized_random_fixed_mdeg", "BayesCI",
    #     meas_noise_int_vals, meas_noise_center_vals,
    #     no_network_hetero, agent_num_vals, fixed_env_noise_std_vals, no_own_weight_vals,
    #     "centralized_random_fixed_mdeg", environtment_type="GaussianExternalSTD", use_meas_noise_center=True)
    #


    # #################################################################################################################
    # # grid search over the combination of parameters
    # network_type_str = "centralized_random" # centralized_random # centralized_random_fixed_mdeg # spatial_normal # spatial_uniform # spatial

    # # Normal Naive Grid Search
    # run_certainty_centrality_grid_search(
    #     "../results/N" + str(agent_num_vals[0]) + "_" + datetime.datetime.now().strftime(
    #         "%Y-%m-%d-%H-%M-%S") + "_certainty_search_Naive_" + network_type_str,
    #     "Naive", meas_noise_int_vals, fixed_center_vals, network_param1_vals, agent_num_vals,
    #     fixed_env_noise_std_vals, own_weight_vals, no_corr_inf_het,
    #     network_type_str, environtment_type="GaussianExternalSTD", use_meas_noise_center=True)

    # # Normal Bayes Grid Search
    # run_certainty_centrality_grid_search(
    #     "../results/N" + str(agent_num_vals[0]) + "_" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    #     + "_certainty_search_Bayes_" + network_type_str, "Bayes",
    #     meas_noise_int_vals, fixed_center_vals,
    #     network_param1_vals, agent_num_vals, fixed_env_noise_std_vals, no_own_weight_vals, no_corr_inf_het,
    #     network_type_str, environtment_type="GaussianExternalSTD", use_meas_noise_center=True)

    # # Normal BayesCI Grid Search
    # run_certainty_centrality_grid_search(
    #     "../results/N" + str(agent_num_vals[0]) + "_" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    #     + "_certainty_search_BayesCI_" + network_type_str, "BayesCI",
    #     meas_noise_int_vals, fixed_center_vals,
    #     network_param1_vals, agent_num_vals, fixed_env_noise_std_vals, no_own_weight_vals, no_corr_inf_het,
    #     network_type_str, environtment_type="GaussianExternalSTD", use_meas_noise_center=True)

    # # Naive Local Optimal Grid Search
    # run_certainty_centrality_grid_search(
    #     "../results/N" + str(agent_num_vals[0]) + "_" + datetime.datetime.now().strftime(
    #         "%Y-%m-%d-%H-%M-%S") + "_certainty_search_NaiveLO_" + network_type_str,
    #     "NaiveLO", meas_noise_int_vals, fixed_center_vals, network_param1_vals, agent_num_vals,
    #     fixed_env_noise_std_vals, no_own_weight_vals, no_corr_inf_het,
    #     network_type_str, environtment_type="GaussianExternalSTD", use_meas_noise_center=True)

    #################################################################################################################
    # grid search over the combination of parameters, with correlation
    network_type_str = "centralized_random_fixed_mdeg"
    
    # # Normal Naive Grid Search
    run_certainty_centrality_grid_search(
        "../results/N" + str(agent_num_vals[0]) + "_" + datetime.datetime.now().strftime(
            "%Y-%m-%d-%H-%M-%S") + "_corelationWithMeas_search_Naive_" + network_type_str,
        "NaiveCORR", no_inf_het, fixed_center_vals, network_param1_vals, agent_num_vals,
        fixed_env_noise_std_vals, own_weight_vals, corr_net_inf_vals,
        "centralized_random_fixed_mdeg", environtment_type="GaussianExternalSTD", use_meas_noise_center=True)

    # Normal Bayes Grid Search
    run_certainty_centrality_grid_search(
        "../results/N" + str(agent_num_vals[0]) + "_" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        + "_corelationWithMeas_search_Bayes_" + network_type_str, "BayesCORR",
        no_inf_het, fixed_center_vals,
        network_param1_vals, agent_num_vals, fixed_env_noise_std_vals, no_own_weight_vals, corr_net_inf_vals,
        "centralized_random_fixed_mdeg", environtment_type="GaussianExternalSTD", use_meas_noise_center=True)

    # Normal BayesCI Grid Search
    run_certainty_centrality_grid_search(
        "../results/N" + str(agent_num_vals[0]) + "_" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        + "_corelationWithMeas_search_BayesCI_" + network_type_str, "BayesCICORR",
        no_inf_het, fixed_center_vals,
        network_param1_vals, agent_num_vals, fixed_env_noise_std_vals, no_own_weight_vals, corr_net_inf_vals,
        "centralized_random_fixed_mdeg", environtment_type="GaussianExternalSTD", use_meas_noise_center=True)

    # Naive Local Optimal Grid Search
    run_certainty_centrality_grid_search(
        "../results/N" + str(agent_num_vals[0]) + "_" + datetime.datetime.now().strftime(
            "%Y-%m-%d-%H-%M-%S") + "_corelationWithMeas_search_NaiveLO_" + network_type_str,
        "NaiveLOCORR", no_inf_het, fixed_center_vals, network_param1_vals, agent_num_vals,
        fixed_env_noise_std_vals, no_own_weight_vals, corr_net_inf_vals,
        "centralized_random_fixed_mdeg", environtment_type="GaussianExternalSTD", use_meas_noise_center=True)

    # #################################################################################################################
    # # testing other agent numbers
    # # Normal Naive Grid Search
    # run_certainty_centrality_grid_search(
    #     "../results/N" + str(agent_num_vals[0]) + "_" + datetime.datetime.now().strftime(
    #         "%Y-%m-%d-%H-%M-%S") + "_certainty_search_Naive_centralized_random_fixed_mdeg",
    #     "Naive", meas_noise_int_vals, fixed_center_vals, no_network_hetero, additional_agent_nums,
    #     network_param1_vals, own_weight_vals,
    #     "centralized_random_fixed_mdeg", environtment_type="GaussianExternalSTD", use_meas_noise_center=True)
    #
    # # Normal Bayes Grid Search
    # run_certainty_centrality_grid_search(
    #     "../results/N" + str(agent_num_vals[0]) + "_" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    #     + "_certainty_search_Bayes_centralized_random_fixed_mdeg", "Bayes",
    #     meas_noise_int_vals, fixed_center_vals,
    #     network_param1_vals, additional_agent_nums, fixed_env_noise_std_vals, no_own_weight_vals,
    #     "centralized_random_fixed_mdeg", environtment_type="GaussianExternalSTD", use_meas_noise_center=True)
    #
    # # Normal BayesCI Grid Search
    # run_certainty_centrality_grid_search(
    #     "../results/N" + str(agent_num_vals[0]) + "_" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    #     + "_certainty_search_BayesCI_centralized_random_fixed_mdeg", "BayesCI",
    #     meas_noise_int_vals, fixed_center_vals,
    #     network_param1_vals, additional_agent_nums, fixed_env_noise_std_vals, no_own_weight_vals,
    #     "centralized_random_fixed_mdeg", environtment_type="GaussianExternalSTD", use_meas_noise_center=True)

    # ################################################################################################################
    # # These are for over/underconfident
    # meas_noise_center_vals_overunder = np.linspace(2.5, 7.5, 11)
    # # Normal Bayes Grid Search
    # run_certainty_centrality_grid_search(
    #     "../results/N" + str(agent_num_vals[0]) + "_" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    #     + "_overunderconfident_search_Bayes_centralized_random_fixed_mdeg", "Bayes",
    #     meas_noise_int_vals, meas_noise_center_vals_overunder,
    #     no_network_hetero, agent_num_vals, fixed_env_noise_std_vals, no_own_weight_vals,
    #     "centralized_random_fixed_mdeg", environtment_type="GaussianFixedSTD", use_meas_noise_center=True)
    #
    # # Normal BayesCI Grid Search
    # run_certainty_centrality_grid_search(
    #     "../results/N" + str(agent_num_vals[0]) + "_" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    #     + "_overunderconfident_search_BayesCI_centralized_random_fixed_mdeg", "BayesCI",
    #     meas_noise_int_vals, meas_noise_center_vals_overunder,
    #     no_network_hetero, agent_num_vals, fixed_env_noise_std_vals, no_own_weight_vals,
    #     "centralized_random_fixed_mdeg", environtment_type="GaussianFixedSTD", use_meas_noise_center=True)
