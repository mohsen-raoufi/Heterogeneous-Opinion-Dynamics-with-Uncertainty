import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from plotting_and_analysis.network_checks.postpro_centrality_certainty import post_pro_network_centrality_certainty
from util import save_and_or_show_plot
from experiments.experiment import ExperimentParameters, run_experiment, NetworkParams
from util import dump_config
import networkx as nx
import matplotlib as mpl
# from palettable.matplotlib import Viridis_3, Plasma_20, Inferno_20
from palettable.cartocolors.sequential import DarkMint_7



default_cmap = DarkMint_7.get_mpl_colormap()
fontsize = 10


def make_subplots(G, timesteps, centrality, uncertainty, min_uncertainty, max_uncertainty, min_centrality, max_centrality):
    widths = [1 / float(timesteps + 1) for _ in range(1, timesteps)]
    widths.extend([0.3 / float(timesteps + 1), 0.5 / float(timesteps + 1), 0.5 / float(timesteps + 1), 0.075 / float(timesteps + 1),  0.3 / float(timesteps + 1)])
    print(np.sum(widths))
    fig, axs = plt.subplots(1, timesteps+4, gridspec_kw={"width_ratios": widths})
    norm = mpl.colors.Normalize(vmin=min_uncertainty, vmax=max_uncertainty)
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=default_cmap)
    centrality_scale = 18
    options_edges = {
        "edge_color": "#808080",
        "width": 0.2, # 0.05
    }

    for i in range(1, timesteps):
        sizes = (centrality[i+1] + 0.01) * centrality_scale / 1.8
        colors = np.log(uncertainty[i])
        # draw edges with curvature
        
        pos = nx.circular_layout(G, scale=1.0)

        # make the edge matrix symmetric
        A = nx.adjacency_matrix(G).todense()
        A = np.maximum(A, A.T)
        G = nx.from_numpy_array(A)

        

        nx.draw_networkx_edges(G, pos, ax=axs[i-1], arrows=True, arrowstyle=None, node_size=0*sizes, connectionstyle='arc3, rad=-.8', **options_edges)
        nx.draw_networkx_nodes(G, pos, ax=axs[i-1], node_size=1.0*sizes, node_color=colors, vmin=min_uncertainty, vmax=max_uncertainty, cmap=cmap.get_cmap())

        axs[i-1].axis('square')
        axs[i-1].set_title("$t="+str(i)+"$", fontsize=fontsize)
        sns.despine(ax=axs[i - 1], top=True, right=True, bottom=True, left=True, trim=True)

    norm = mpl.colors.LogNorm(vmin=np.exp(min_uncertainty), vmax=np.exp(max_uncertainty))
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=default_cmap)
    mpl.colorbar.ColorbarBase(axs[-2], cmap=cmap.get_cmap(),orientation='vertical', norm=norm)
    sns.despine(ax=axs[-1], top=True, right=True, bottom=True, left=True, trim=True)
    axs[-1].get_xaxis().set_visible(False)
    axs[-1].get_yaxis().set_visible(False)
    axs[-2].set_title("Uncertainty", fontsize=fontsize)
    axs[-2].yaxis.set_ticks_position('left')
    axs[-2].yaxis.set_label_position("left")
    sns.despine(ax=axs[-2], top=True, right=True, bottom=True, left=True, trim=True)
    sns.despine(ax=axs[-3], top=True, right=True, bottom=True, left=True, trim=True)
    axs[-3].get_xaxis().set_visible(False)
    axs[-3].get_yaxis().set_visible(False)
    axs[-4].set_title("Centrality", fontsize=fontsize)
    centralities = np.linspace(1, 10, 5)
    axs[-4].scatter(np.zeros_like(centralities), centralities, s=(centralities) * centrality_scale, color="white", edgecolors='black')
    axs[-4].set_ylim(0, 11)
    sns.despine(ax=axs[-4], top=True, right=True, bottom=True, trim=False)
    axs[-4].get_xaxis().set_visible(False)
    sns.despine(ax=axs[-5], top=True, right=True, bottom=True, left=True, trim=True)
    axs[-5].get_xaxis().set_visible(False)
    axs[-5].get_yaxis().set_visible(False)
    fig.set_size_inches(10, 1.75)
    plt.tight_layout(w_pad=-3)
    save_and_or_show_plot(plot_dir, "circular_cent_cert.pdf", False)


if __name__=="__main__":
    plot_dir = "/home/mohsen/Project/colab/collective-decison-making-with-direl/results/test/K_reg/plots"
    n_agents = 26
    timesteps = 7
    params = ExperimentParameters(agent_type="BayesCircularPlot",
                                  steps=timesteps,
                                  environment_type="GaussianExternalSTD",
                                  seed=7,
                                  hack_for_circular_plot=True,
                                  network_params=NetworkParams(network_type="k_regular",
                                                               n_agents=n_agents,
                                                               scalar_param1=6,  # 0.4,
                                                               correlation_network_information=-5.0))
    path = "/home/mohsen/Project/colab/collective-decison-making-with-direl/results/test/K_reg"
    os.makedirs(path, exist_ok=True)
    dump_config(params, os.path.join(path, "config"))
    run_experiment(params=params, file_path=path)

    df = post_pro_network_centrality_certainty(path)
    A = df.adjc_time[0]
    A[np.eye(n_agents).astype(bool)] = 0
    G = nx.from_numpy_array(A)
    centrality = df.centrality_time
    uncertainty = df.uncertainty_time
    min_uncertainty = np.min(np.log(uncertainty[timesteps]))
    max_uncertainty = np.max(np.log(uncertainty[1]))
    min_centrality = 1000
    max_centrality = -1
    for i in range(1, timesteps):
        min_centrality = min(min_centrality, np.min(centrality[i]))
        max_centrality = max(max_centrality, np.max(centrality[i]))
    make_subplots(G, timesteps, centrality, uncertainty, min_uncertainty, max_uncertainty, min_centrality, max_centrality)
