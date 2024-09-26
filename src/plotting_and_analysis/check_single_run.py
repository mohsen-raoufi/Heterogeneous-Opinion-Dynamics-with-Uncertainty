import glob
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

import networkx as nx

from networkStudy_parallel import extract_relevant_data_from_files


def give_exp_options(base_path):
    paths = glob.glob(os.path.join(base_path, "*"))
    return paths


def get_random_exp(base_path):
    paths = give_exp_options(base_path)
    p = random.choice(paths)
    return p


def get_data_single_run(single_run_path):
    return extract_relevant_data_from_files(single_run_path)


def trueness_error_wrt_to_most_central(zArr, adjc, avgZArr):
    A_mat = adjc[0].cpu().detach().numpy()
    out_deg = A_mat.sum(axis=0)
    idx_highest_out = np.argmax(out_deg)
    belief_most_central = zArr[idx_highest_out, :]
    E_t_most_central = (avgZArr - belief_most_central) ** 2
    return E_t_most_central


def visualize_beliefs_over_time(zArr, pArr, z_gt, avgZArr, adjc):
    time_steps = zArr.shape[1]
    n_agents = zArr.shape[0]
    best_split = int(np.ceil(np.sqrt(time_steps)))
    fig, axs = plt.subplots(best_split, best_split, sharex="all", sharey="none")
    low_bar = np.min(zArr[:, :] - 3 * pArr[:, :])
    high_bar = np.max(zArr[:, :] + 3 * pArr[:, :])
    A_mat = adjc[0].cpu().detach().numpy()
    out_deg = A_mat.sum(axis=0)
    idx_highest_out = np.argmax(out_deg)
    for i in range(1, time_steps):
        ax = axs[int((i-1) / best_split)][int((i-1) % best_split)]
        max_val = show_beliefs_for_timestep(ax, high_bar, i, low_bar, n_agents, pArr, zArr)
        ax.set_title("$t=" + str(i) + "$")
        ax.vlines(z_gt, -0.1 * max_val, max_val * 1.1, colors="red", linestyles="--", label="True Value")
        ax.vlines(zArr[idx_highest_out, i], -0.1 * max_val, max_val * 1.1, colors="green", linestyles="-",
                  label="Most Central Node Belief")
        ax.vlines(avgZArr[i], -0.1 * max_val, max_val * 1.1, colors="blue", linestyles=":",
                  label="Collective Average")
    h, l = axs[0][0].get_legend_handles_labels()
    fig.legend(h, l)
    fig.suptitle('Belief over State in the Collective over full Sequence', fontsize=12)
    fig.set_size_inches(12, 8, forward=True)
    plt.tight_layout(pad=2.0, w_pad=0.5, h_pad=1.0)


def show_beliefs_for_timestep(ax, high_bar, timestep, low_bar, n_agents, pArr, zArr):
    x = np.concatenate([np.linspace(low_bar, high_bar, 10000), zArr[:, timestep]])
    x.sort()
    max_val = -10000.0
    for j in range(n_agents):
        gauss_vals = stats.norm.pdf(x, zArr[j, timestep], pArr[j, timestep])
        ax.plot(x, gauss_vals, linewidth=1.0)
        max_val = np.maximum(np.max(gauss_vals), max_val)
    ax.set_ylim(-0.1 * max_val, max_val * 1.1)
    ax.set_xlim(low_bar, high_bar)
    return max_val


def visualize_errors_over_time(E_a, E_p, E_t, E_t_most_central):
    time_steps = E_a.shape[0]
    fig, axs = plt.subplots(4, 1, sharex="all")
    ax = axs[0]
    ax.plot(np.arange(1, time_steps), E_p[1:])
    ax.set_ylabel("Precision Error")
    ax = axs[1]
    ax.plot(np.arange(1, time_steps), E_t[1:])
    ax.set_ylabel("Trueness Error")
    ax = axs[2]
    ax.plot(np.arange(1, time_steps), E_a[1:])
    ax.set_ylabel("Accuracy Error")
    ax = axs[3]
    ax.plot(np.arange(1, time_steps), E_t_most_central[1:], color="r")
    ax.set_ylabel("Trueness Error wrt Most Central Node")

    axs[-1].set_xlabel("Time $t$")
    fig.suptitle('Estimation Errors of the Collective over full Sequence', fontsize=12)
    fig.set_size_inches(6, 6, forward=True)
    plt.tight_layout(pad=2.0, w_pad=0.5, h_pad=1.0)


def visualize_weights_over_time(adjc, time_steps):

    # n_agents = zArr.shape[0]
    best_split = int(np.ceil(np.sqrt(time_steps)))
    fig, axs = plt.subplots(best_split, best_split)  # , sharex="all")#, sharey="all")
    fig.set_figheight(9)
    fig.set_figwidth(15)
    mean_degree_in = []
    std_degree_in = []
    mean_degree_out = []
    std_degree_out = []
    i = 0
    for i in range(time_steps):
        ax = axs[int(i / best_split)][int(i % best_split)]
        A_mat = adjc[i].cpu().detach().numpy()
        # A_graph = nx.from_numpy_array(A_mat)
        in_deg = A_mat.sum(axis=1)
        out_deg = A_mat.sum(axis=0)
        ax.plot(in_deg, 'b', label="in degree")

        m_deg_in = np.mean(in_deg)
        std_deg_in = np.std(in_deg)
        m_deg_out = np.mean(out_deg)
        std_deg_out = np.std(out_deg)
        # ax_2 = ax.twiny()
        # ax_2.hist(tmp)
        mean_degree_in.append(m_deg_in)
        std_degree_in.append(std_deg_in)

        mean_degree_out.append(m_deg_out)
        std_degree_out.append(std_deg_out)

        # ax.plot(ax.get_xlim(),np.array([m_deg_in, m_deg_in]),'b')
        ax2 = ax  # .twinx()
        ax2.tick_params(colors='red')
        ax2.plot(out_deg, '--r', label="out degree")
        # ax2.plot(ax.get_xlim(),np.array([m_deg_out, m_deg_out]),'r')
        ax.set_title("$sum\ weights\ at\ t=" + str(i) + "$")

    mean_degree_in = np.array(mean_degree_in)
    std_degree_in = np.array(std_degree_in)
    mean_degree_out = np.array(mean_degree_out)
    std_degree_out = np.array(std_degree_out)

    A_mat_0 = adjc[0].cpu().detach().numpy()
    G = nx.from_numpy_array(A_mat_0)
    in_degree = np.array(G.degree())[:, 1]

    i = i + 1
    ax = axs[int(i / best_split)][int(i % best_split)]
    nx.draw(G, ax=ax, node_size=100 * (in_degree / 100) ** 4, alpha=0.1, width=0.1)
    ax.set_title('Graph')

    t_start_plot = 0
    i = i + 1
    ax = axs[int(i / best_split)][int(i % best_split)]
    ax.plot(np.arange(t_start_plot, time_steps - 1), mean_degree_in[t_start_plot:-1], 'black', label="m_deg_in")
    ax.fill_between(np.arange(t_start_plot, time_steps - 1), mean_degree_in[t_start_plot:-1] - std_degree_in[t_start_plot:-1], mean_degree_in[t_start_plot:-1] + std_degree_in[t_start_plot:-1], alpha=.25, color="b")
    ax.set_title("Average In Weight")

    i = i + 1
    ax = axs[int(i / best_split)][int(i % best_split)]
    ax.plot(np.arange(t_start_plot, time_steps - 1), mean_degree_out[t_start_plot:-1], 'black', label="m_deg_out")
    ax.fill_between(np.arange(t_start_plot, time_steps - 1), mean_degree_out[t_start_plot:-1] - std_degree_out[t_start_plot:-1], mean_degree_out[t_start_plot:-1] + std_degree_out[t_start_plot:-1], alpha=.25, color="r")
    ax.set_title("Average Out Weight")

    i = i + 1
    ax = axs[int(i / best_split)][int(i % best_split)]
    A_mat_5 = adjc[5].cpu().detach().numpy()
    imsh_AMat_5 = ax.imshow(A_mat_5, cmap='Blues')
    fig.colorbar(imsh_AMat_5, ax=ax)
    ax.set_title("Weights at t=5")

    h, l = axs[0][0].get_legend_handles_labels()
    fig.legend(h, l)
    fig.suptitle('In and Out Degrees in the Collective', fontsize=12)
    plt.tight_layout(pad=2.0, w_pad=0.5, h_pad=1.0)


def visualize_adjc_over_time(adjc, time_steps):

    best_split = int(np.ceil(np.sqrt(time_steps)))
    fig, axs = plt.subplots(best_split, best_split, sharex="all", sharey="all")
    fig.set_figheight(9)
    fig.set_figwidth(15)
    for i in range(time_steps):
        ax = axs[int(i / best_split)][int(i % best_split)]
        A_mat = adjc[i].cpu().detach().numpy()
        G = nx.from_numpy_array(A_mat)
        in_degree = np.array(G.degree())[:, 1]
        nx.draw(G, ax=ax, node_size=100 * (in_degree / 100) ** 4, alpha=0.1, width=0.1)
        ax.set_title("Graph at t=" + str(i))
    plt.tight_layout(pad=2.0, w_pad=0.5, h_pad=1.0)


def visualize_weight_mat_over_time(adjc, time_steps):
    best_split = int(np.ceil(np.sqrt(time_steps)))
    fig, axs = plt.subplots(best_split, best_split)  # , sharex="all")#, sharey="all")
    fig.set_figheight(9)
    fig.set_figwidth(15)

    i = 0
    ax = axs[int(i / best_split)][int(i % best_split)]
    A_mat = adjc[i].cpu().detach().numpy()
    imsh_AMat = ax.imshow(A_mat, cmap='Blues')
    fig.colorbar(imsh_AMat, ax=ax)
    ax.set_title("Weights at t=" + str(i))

    for i in range(1, time_steps):
        ax = axs[int(i / best_split)][int(i % best_split)]
        A_mat = adjc[i].cpu().detach().numpy()  # + adjc[i-1].cpu().detach().numpy()
        imsh_AMat = ax.imshow(A_mat, cmap='Blues')
        fig.colorbar(imsh_AMat, ax=ax)
        ax.set_title("Weights t=" + str(i) + "-" + str(i - 1))
    plt.tight_layout(pad=2.0, w_pad=0.5, h_pad=1.0)


def visualize_change_weight_mat_over_time(adjc, time_steps):
    best_split = int(np.ceil(np.sqrt(time_steps)))
    fig, axs = plt.subplots(best_split, best_split)  # , sharex="all")#, sharey="all")
    fig.set_figheight(9)
    fig.set_figwidth(15)

    i = 0
    ax = axs[int(i / best_split)][int(i % best_split)]
    A_mat = adjc[i].cpu().detach().numpy()
    imsh_AMat = ax.imshow(A_mat, cmap='Blues')
    fig.colorbar(imsh_AMat, ax=ax)
    ax.set_title("Weights at t=" + str(i))

    for i in range(1, time_steps):
        ax = axs[int(i / best_split)][int(i % best_split)]
        A_mat = - adjc[i].cpu().detach().numpy() + adjc[i - 1].cpu().detach().numpy()
        imsh_AMat = ax.imshow(A_mat, cmap='bwr')  # coolwarm')
        fig.colorbar(imsh_AMat, ax=ax)
        ax.set_title("Change of Weights t=" + str(i) + "-" + str(i - 1))
    plt.tight_layout(pad=2.0, w_pad=0.5, h_pad=1.0)


def visualize_data_from_single_run_data(data_tuple):
    E_a, E_p, E_t, adjc, G, avgZArr, clustCoef, degreeDist, eigVal, eigVec, env_noise_std,\
        mean_agent_measurement_noise, n_agents, range_agent_measurement_noise, scalar_param1, steps,\
        weight_own_belief, z_gt, zArr, pArr = data_tuple
    E_t_most_central = trueness_error_wrt_to_most_central(zArr, adjc, avgZArr)

    visualize_weights_over_time(adjc, E_a.shape[0])
    plt.show()


    visualize_adjc_over_time(adjc, E_a.shape[0])
    plt.show()

    visualize_weight_mat_over_time(adjc, E_a.shape[0])
    plt.show()

    visualize_change_weight_mat_over_time(adjc, E_a.shape[0])
    plt.show()

    visualize_beliefs_over_time(zArr, pArr, z_gt, avgZArr, adjc)
    plt.show()

    visualize_errors_over_time(E_a, E_p, E_t, E_t_most_central)
    plt.show()


if __name__ == "__main__":
    path = "/home/vito/ws_domip/src/collective-decison-making-with-direl/results/hpc_mount/" \
           "N100_2023-05-02-14-58-42_test_grid_search_Bayes_centralized_random/results" \
           "/envstd_1_mnint_0_mnmin_0_np1_0_ow_0_a_n_0_run_99"
    visualize_data_from_single_run_data(get_data_single_run(path))
