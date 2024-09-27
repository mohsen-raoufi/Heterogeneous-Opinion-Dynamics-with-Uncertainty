import os
from check_single_run import show_beliefs_for_timestep, get_data_single_run
import numpy as np
import matplotlib.pyplot as plt

from plotting_and_analysis.information_hetero_plot import get_relavant_data
from plotting_and_analysis.postPro.postpro_centrality_certainty import post_pro_network_centrality_certainty
from util import save_and_or_show_plot
from experiments.experiment import ExperimentParameters, run_experiment, NetworkParams
from util import dump_config
import networkx as nx
import matplotlib as mpl
# from palettable.matplotlib import Viridis_3, Plasma_20, Inferno_20
from palettable.cartocolors.sequential import DarkMint_7




default_cmap = DarkMint_7.get_mpl_colormap()


def run_preprocessing(path):
    df = post_pro_network_centrality_certainty(path)
    A = df.adjc_time[0]
    A[np.eye(n_agents).astype(bool)] = 0
    G = nx.from_numpy_array(A)
    centrality = df.centrality_time[1]
    uncertainty = df.uncertainty_time[1]
    min_uncertainty = np.min(uncertainty)
    max_uncertainty = np.max(uncertainty)
    min_centrality = np.min(centrality)
    max_centrality = np.max(centrality)
    return G, centrality, uncertainty, min_uncertainty, max_uncertainty, min_centrality, max_centrality


def make_indepdent_mismodeling_illustration():
    global pathes, params, path, n_agents
    pathes = []
    for mint in [5.0, 2.5, 0.0]:
        for mcenter in [2.5, 5.0, 7.5]:
            params = ExperimentParameters(agent_type="Bayes",
                                          steps=timesteps,
                                          environment_type="GaussianFixedSTD",
                                          seed=6,
                                          env_noise_std=5.0,
                                          true_value=0.0,
                                          network_params=NetworkParams(network_type="centralized_random_fixed_mdeg",
                                                                       n_agents=n_agents,
                                                                       scalar_param1=0.4,  # 0.4,
                                                                       max_agent_measurement_noise=mcenter + mint,
                                                                       min_agent_measurement_noise=max(mcenter - mint,
                                                                                                       0.0001)))
            path = "mismodeling_runs/run/mcenter_" + str(mcenter) + "_pinf_" + str(mint)
            os.makedirs(path, exist_ok=True)
            dump_config(params, os.path.join(path, "config"))
            run_experiment(params=params, file_path=path)
            pathes.append(path)
    data = [get_data_single_run(p) for p in pathes]
    fig, axs = plt.subplots(3, 3, sharex="all", sharey="all")
    fig.set_size_inches(4, 4)
    fig.set_dpi(dpi)
    high_bars = []
    low_bars = []
    max_vals = []
    for i in data:
        high_bar, low_bar, _, _, _ = get_relavant_data(i)
        high_bars.append(high_bar)
        low_bars.append(low_bar)
    high_bar = max(high_bars) * 0.6
    low_bar = min(low_bars) * 0.6
    for i in range(9):
        _, _, n_agents, pArr, zArr = get_relavant_data(data[i])
        max_val = show_beliefs_for_timestep(axs[int(int(i) / 3), int(int(i) % 3)], high_bar, 1, low_bar, n_agents, pArr,
                                            zArr)
        max_vals.append(max_val)
    max_val = max(max_vals) * 0.6
    for ax in axs.reshape(-1):
        # ax.grid(True)
        # ax.plot([low_bar, high_bar], [0.0, 0.0], color="black")
        ax.set_xlim(low_bar, high_bar)
        ax.set_ylim(-0.05 * max_val, max_val)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        # sns.despine(ax=ax, top=False, right=False, bottom=False, left=False, trim=True)
    plt.tight_layout(pad=-1.0)
    save_and_or_show_plot(plot_dir, "mismodeling_independent.png", False)


def make_correlated_midemodelling_illustration():
    global pathes, params, path
    pathes = []
    for net_param in [0.0, 0.07, 0.3]:
        for cor in [-1.0, 0.0, 1.0]:
            params = ExperimentParameters(agent_type="BayesCORRMismodel",
                                          steps=timesteps,
                                          environment_type="GaussianFixedSTD",
                                          seed=7,
                                          env_noise_std=5.0,
                                          true_value=0.0,
                                          network_params=NetworkParams(network_type="centralized_random_fixed_mdeg",
                                                                       n_agents=n_agents,
                                                                       scalar_param1=net_param,  # 0.4,
                                                                       correlation_network_information=cor))
            path = "mismodeling_runs/run/np_" + str(net_param) + "_cor_" + str(cor)
            os.makedirs(path, exist_ok=True)
            dump_config(params, os.path.join(path, "config"))
            run_experiment(params=params, file_path=path)
            pathes.append(path)
            print("Done")
    data = [run_preprocessing(p) for p in pathes]
    fig, axs = plt.subplots(3, 3)
    fig.set_size_inches(4, 4)
    fig.set_dpi(dpi)
    centrality_scale = 18
    options_edges = {
        "edge_color": "#808080",
        "width": 0.5
    }

    pos = -1
    for i in range(len(data)):
        d = data[i]
        ax = axs[int(int(i) / 3), int(int(i) % 3)]
        sizes = (d[1]) * centrality_scale
        colors = d[2]
        min_uncertainty = d[3]
        max_uncertainty = d[4]

        if min_uncertainty != max_uncertainty:
            min_uncertainty = min_uncertainty - 0.2 * (max_uncertainty - min_uncertainty)
        else:
            min_uncertainty = min_uncertainty * 0.5
            max_uncertainty = max_uncertainty * 1.5
        norm = mpl.colors.Normalize(vmin=min_uncertainty, vmax=max_uncertainty)
        cmap = mpl.cm.ScalarMappable(norm=norm, cmap=default_cmap)
        if int(i) % 3 == 0:
            pos = nx.spring_layout(d[0])
        nx.draw(d[0], pos=pos, ax=ax, node_size=sizes, node_color=colors, vmin=min_uncertainty,
                vmax=max_uncertainty, cmap=cmap.get_cmap(), **options_edges)
    for ax in axs.reshape(-1):
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set(adjustable='box', aspect='equal')
    plt.tight_layout(pad=0.0)
    save_and_or_show_plot(plot_dir, "mismodeling_correl.png", False)


if __name__=="__main__":
    plot_dir = "plots/single_runs"
    timesteps = 1
    dpi = 1000

    n_agents = 10
    make_indepdent_mismodeling_illustration()

    n_agents = 25
    make_correlated_midemodelling_illustration()