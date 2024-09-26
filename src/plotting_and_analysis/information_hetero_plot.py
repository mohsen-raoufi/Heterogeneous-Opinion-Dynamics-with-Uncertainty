import os
from check_single_run import show_beliefs_for_timestep, get_data_single_run
import numpy as np
import matplotlib.pyplot as plt
from util import save_and_or_show_plot
from experiments.experiment import ExperimentParameters, run_experiment, NetworkParams
from util import dump_config

font_size = 10.5


def plot_for_two_different_timesteps(exp_data, axs, t1=1, t2=9):
    high_bar, low_bar, n_agents, pArr, zArr = get_relavant_data(exp_data)
    max_val1 = show_beliefs_for_timestep(axs[0], high_bar, t1, low_bar, n_agents, pArr, zArr)
    max_val2 = show_beliefs_for_timestep(axs[1], high_bar, t2, low_bar, n_agents, pArr, zArr)
    return max_val1, max_val2, low_bar, high_bar


def get_relavant_data(exp_data):
    E_a, E_p, E_t, adjc, G, avgZArr, clustCoef, degreeDist, eigVal, eigVec, env_noise_std, \
        mean_agent_measurement_noise, n_agents, range_agent_measurement_noise, scalar_param1, steps, \
        weight_own_belief, z_gt, zArr, pArr, correlation_network_information = exp_data
    zArr = zArr - z_gt
    low_bar = np.min(zArr[:, :] - 3 * pArr[:, :])
    high_bar = np.max(zArr[:, :] + 3 * pArr[:, :])
    return high_bar, low_bar, n_agents, pArr, zArr


def plot_different_heteros(data_low, data_high, method_name):
    fig, axs = plt.subplots(2, 2)
    fig.set_size_inches(6, 4)
    fig.set_dpi(700)
    fig.suptitle("Different Heterogeneities of Information for "+method_name)
    max_val11, max_val12, x_low1, x_high1 = plot_for_two_different_timesteps(data_low, axs[0])
    max_val21, max_val22, x_low2, x_high2 = plot_for_two_different_timesteps(data_high, axs[1])
    max_val = max_val11 * 2
    x_low = min(x_low1, x_low2)
    x_high = max(x_high1, x_high2)
    for ax in axs.reshape(-1):
        ax.plot([x_low, x_high], [0.0, 0.0], color="black")
        ax.set_xlim(x_low, x_high)
        ax.set_ylim(-0.1 * max_val, max_val * 1.1)
    axs[1, 0].set_xlabel("State")
    axs[1, 1].set_xlabel("State")
    axs[0, 0].set_ylabel("Bel(State)")
    axs[1, 0].set_ylabel("Bel(State)")
    axs[0, 0].set_title("Initial Belief")
    axs[0, 1].set_title("Final Belief")
    plt.tight_layout(w_pad=7)


def plot_inf_het_gen(data_r1, data_r2, data_r3):
    high_bars = []
    low_bars = []
    for i in [*data_r1, *data_r2, *data_r3]:
        high_bar, low_bar, _, _, _ = get_relavant_data(i)
        high_bars.append(high_bar)
        low_bars.append(low_bar)
    high_bar = max(high_bars) * 0.8
    low_bar = min(low_bars) * 0.8
    fig, axs = plt.subplots(3, 3, sharex="all", sharey="all")
    max_vals = []
    for i, d in enumerate(data_r1):
        _, _, n_agents, pArr, zArr = get_relavant_data(d)
        max_val = show_beliefs_for_timestep(axs[i, 0], high_bar, 10, low_bar, n_agents, pArr, zArr)
        max_vals.append(max_val)
        for tick in axs[0, i].xaxis.get_major_ticks():
            tick.tick1line.set_visible(False)
            tick.tick2line.set_visible(False)
            tick.label1.set_visible(False)
            tick.label2.set_visible(False)
    for i, d in enumerate(data_r2):
        _, _, n_agents, pArr, zArr = get_relavant_data(d)
        max_val = show_beliefs_for_timestep(axs[i, 1], high_bar, 1, low_bar, n_agents, pArr, zArr)
        max_vals.append(max_val)
        for tick in axs[1, i].xaxis.get_major_ticks():
            tick.tick1line.set_visible(False)
            tick.tick2line.set_visible(False)
            tick.label1.set_visible(False)
            tick.label2.set_visible(False)
    for i, d in enumerate(data_r3):
        _, _, n_agents, pArr, zArr = get_relavant_data(d)
        max_val = show_beliefs_for_timestep(axs[i, 2], high_bar, 10, low_bar, n_agents, pArr, zArr)
        max_vals.append(max_val)
    for i in range(3):
        for tick in axs[i, 1].yaxis.get_major_ticks():
            tick.tick1line.set_visible(False)
            tick.tick2line.set_visible(False)
            tick.label1.set_visible(False)
            tick.label2.set_visible(False)
        for tick in axs[i, 2].yaxis.get_major_ticks():
            tick.tick1line.set_visible(False)
            tick.tick2line.set_visible(False)
            tick.label1.set_visible(False)
            tick.label2.set_visible(False)
    max_val = np.median(max_vals) * 1.2
    for ax in axs.reshape(-1):
        ax.grid(True)
        # ax.plot([low_bar, high_bar], [0.0, 0.0], color="black")
        ax.set_xlim(low_bar, high_bar)
        ax.set_ylim(-0.1 * max_val, max_val * 1.1)
    axs[2, 0].set_xlabel("Opinion $x$", fontsize=font_size)
    axs[2, 1].set_xlabel("Opinion $x$", fontsize=font_size)
    axs[2, 2].set_xlabel("Opinion $x$", fontsize=font_size)
    axs[0, 1].set_title("Initial opinion\n with uncertainty\n", fontsize=font_size)
    axs[0, 0].set_title("Final opinion\nwith uncertainty\nBI-UD", fontsize=font_size)            #$p(x_{t=10}^{[i]})$")
    axs[0, 2].set_title("Final opinion\nwith uncertainty\nBI-AI", fontsize=font_size)       #$p(x_{t=10}^{[i]})$")
    axs[0, 0].set_ylabel("No\ninformation\nheterogeneity\n$p_\mathrm{inf} = 0$\n\n$p(x)$", fontsize=font_size)
    axs[1, 0].set_ylabel("Medium\ninformation\nheterogeneity\n$p_\mathrm{inf} = 2.5$\n\n$p(x)$", fontsize=font_size)
    axs[2, 0].set_ylabel("High\ninformation\nheterogeneity\n$p_\mathrm{inf} = 5$\n\n$p(x)$", fontsize=font_size)
    fig.set_size_inches(10.5, 5)
    plt.tight_layout(w_pad=1)



if __name__=="__main__":
    center = 5.0
    for width in [0.0, 2.5, 5.0]:
        for agent_type in ["Bayes", "BayesCI"]:
            params = ExperimentParameters(agent_type=agent_type,
                                          steps=10,
                                          environment_type="GaussianExternalSTD",
                                          seed=2,
                                          network_params=NetworkParams(network_type="centralized_random_fixed_mdeg",
                                                                       n_agents=25,
                                                                       min_agent_measurement_noise=center - 0.5 * width,
                                                                       max_agent_measurement_noise=center + 0.5 * width))
            # path = "/home/mohsen/Project/colab/collective-decison-making-with-direl/results/test/"
            path = "/media/vito/TOSHIBA EXT/single_runs_collectives/" + params.agent_type + "_" + str(width)
            os.makedirs(path, exist_ok=True)
            dump_config(params, os.path.join(path, "config"))
            run_experiment(params=params, file_path=path)

    path_bayes_low = "/media/vito/TOSHIBA EXT/single_runs_collectives/Bayes_0.0"
    path_bayes_middle = "/media/vito/TOSHIBA EXT/single_runs_collectives/Bayes_2.5"
    path_bayes_high = "/media/vito/TOSHIBA EXT/single_runs_collectives/Bayes_5.0"

    path_ci_low = "/media/vito/TOSHIBA EXT/single_runs_collectives/BayesCI_0.0"
    path_ci_middle = "/media/vito/TOSHIBA EXT/single_runs_collectives/BayesCI_2.5"
    path_ci_high = "/media/vito/TOSHIBA EXT/single_runs_collectives/BayesCI_5.0"

    plot_dir = "/home/vito/ws_domip/src/collective-decison-making-with-direl/plots/single_runs"
    os.makedirs(plot_dir, exist_ok=True)

    data_bayes_low = get_data_single_run(path_bayes_low)
    data_bayes_middle = get_data_single_run(path_bayes_middle)
    data_bayes_high = get_data_single_run(path_bayes_high)
    data_ci_low = get_data_single_run(path_ci_low)
    data_ci_middle = get_data_single_run(path_ci_middle)
    data_ci_high = get_data_single_run(path_ci_high)

    data_bayes = [data_bayes_low, data_bayes_middle, data_bayes_high]
    data_ci = [data_ci_low, data_ci_middle, data_ci_high]

    plot_inf_het_gen(data_ci, data_bayes , data_bayes)
    save_and_or_show_plot(plot_dir, "het_inf_gen.pdf", False)


