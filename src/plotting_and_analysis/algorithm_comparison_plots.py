import os
import matplotlib.pyplot as plt
import pandas
import seaborn as sns

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from util import dump_python_object, load_python_object, save_and_or_show_plot
import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm
import matplotlib as mpl

# Setting default colormap settings
from palettable.scientific.diverging import Berlin_3
from palettable.matplotlib import Viridis_3, Plasma_20, Inferno_20


default_cmap = Viridis_3.get_mpl_colormap()
default_result_path = "data/processed_files"
default_plots_path = "plots"
default_cache_path = "cached_dfs"

def get_colorMaps(nIds):
    CMap = default_cmap
    weights = np.arange(0, nIds)

    norm = mpl.colors.Normalize(vmin=0, vmax=nIds)
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=CMap)

    return cmap


def plot_for_parameter(ax, dataframe, x_parameter, y_parameter, c_parameter=None, fixed_param_settings=None, cmap=None):
    if fixed_param_settings is not None:
        for p, setting in fixed_param_settings.items():
            dataframe = dataframe[dataframe[p] == setting]
    if c_parameter is not None:
        c_param_options = np.unique(dataframe[c_parameter])
    else:
        c_param_options = [0]
    if cmap is None:
        cmap = get_colorMaps(len(c_param_options))

    for i, c_param_value in enumerate(c_param_options):
        if c_parameter is not None:
            tmp_dataframe = dataframe[dataframe[c_parameter] == c_param_value]
        else:
            tmp_dataframe = dataframe
        ax.plot(tmp_dataframe.groupby(x_parameter).mean()[y_parameter], color=cmap.to_rgba(i))
    return cmap


def plot_colorbar(ax, values, cmap):
    mpl.colorbar.ColorbarBase(ax, cmap=cmap, values=values, orientation='horizontal')


def set_share_axes(axs, target=None, sharex=False, sharey=False):
    # from https://stackoverflow.com/questions/23528477/share-axes-in-matplotlib-for-only-part-of-the-subplots
    if target is None:
        target = axs.flat[0]
    # Manage share using grouper objects
    for ax in axs.flat:
        if sharex:
            target.get_shared_x_axes().join(target, ax)
        if sharey:
            target.get_shared_y_axes().join(target, ax)
    # Turn off x tick labels and offset text for all but the bottom row
    if sharex and axs.ndim > 1:
        for ax in axs[:-1,:].flat:
            ax.xaxis.set_tick_params(which='both', labelbottom=False, labeltop=False)
            ax.xaxis.offsetText.set_visible(False)
    # Turn off y tick labels and offset text for all but the left most column
    if sharey and axs.ndim > 1:
        for ax in axs[:,1:].flat:
            ax.yaxis.set_tick_params(which='both', labelleft=False, labelright=False)
            ax.yaxis.offsetText.set_visible(False)

def compare_average_errors_over_network_param(dataframe, fixed_param_settings=None,
                                              c_parameter="Centrality Homogeneity", logspace=None):
    if fixed_param_settings is None:
        fixed_param_settings = dict()
    dataframe = dataframe[dataframe["Timestep"] > 0]
    fig = plt.figure(tight_layout=True)
    gs = mpl.gridspec.GridSpec(4, 4, height_ratios=[0.3, 0.3, 0.3, 0.1])
    fig.set_size_inches(10, 10)
    axs = np.array([[fig.add_subplot(gs[i, j]) for j in range(4)] for i in range(3)])
    for j, c in enumerate(np.unique(dataframe["Configuration"])):
        axs[0, j].set_title(c)
        axs[-1, j].set_xlabel("Timestep")
        for i, errortype in enumerate(["Trueness Error", "Precision Error", "Accuracy Error"]):
            fixed_param_settings_tmp = fixed_param_settings.copy()
            fixed_param_settings_tmp["Configuration"] = c
            cmap = plot_for_parameter(axs[i, j], dataframe, "Timestep", errortype, c_parameter=c_parameter,
                                      fixed_param_settings=fixed_param_settings_tmp)
            axs[i, 0].set_ylabel(errortype)
            if i != 0:
                axs[i, j].get_shared_x_axes().join(axs[i, j], axs[0, j])
            if j != 0:
                axs[i, j].get_shared_y_axes().join(axs[i, j], axs[i, 0])
            if logspace is not None:
                if logspace == "x" or logspace == "both":
                    axs[i, j].set_xscale('log')
                if logspace == "y" or logspace == "both":
                    axs[i, j].set_yscale('log')
    if c_parameter is not None:
        c_param_options = np.unique(dataframe[c_parameter])
        special_ax = fig.add_subplot(gs[3, :])
        plot_colorbar(special_ax, c_param_options, cmap=cmap.get_cmap())
        special_ax.set_xlabel(c_parameter)


def compare_average_errors_over_two_param(dataframe, fixed_param_settings=None, timestep=9,
                                              x_param="Centrality Homogeneity", y_param="", logspace=False):
    if fixed_param_settings is None:
        fixed_param_settings = dict()
    dataframe = dataframe[dataframe["Timestep"] == timestep]
    fig = plt.figure(tight_layout=True)
    gs = mpl.gridspec.GridSpec(3, 4, height_ratios=[0.33, 0.33, 0.33])
    fig.set_size_inches(10, 10)
    axs = np.array([[fig.add_subplot(gs[i, j]) for j in range(4)] for i in range(3)])
    for j, c in enumerate(np.unique(dataframe["Configuration"])):
        axs[0, j].set_title(c)
        axs[-1, j].set_xlabel(x_param)
        for i, errortype in enumerate(["Trueness Error", "Precision Error", "Accuracy Error"]):
            # tmp_table = pd.pivot_table(dataframe.round(4), values=errortype, index=[y_param], columns=[x_param],
            #                        fill_value=0).to_numpy()
            # map_max = tmp_table.max()
            # map_min = tmp_table.min()
            map_max = None
            map_min = None
            fixed_param_settings_tmp = fixed_param_settings.copy()
            fixed_param_settings_tmp["Configuration"] = c
            cmap = plot_heatmap_for_columns(dataframe, axs[i, j], mapped_value=errortype, x_col=x_param, y_col=y_param, fixed_param_settings=fixed_param_settings_tmp, map_min=map_min, map_max=map_max, logspace=logspace)
            # axs[i, 0].set_ylabel(errortype)
            if i != 0:
                axs[i, j].get_shared_x_axes().join(axs[i, j], axs[0, j])
            if j != 0:
                axs[i, j].get_shared_y_axes().join(axs[i, j], axs[i, 0])
    return axs


def filter_for_optimal_weights_in_naive(dataframe, groupby=None):
    optimized_df, optim_idxs = find_optimal_param_per_setting(dataframe, groupby=groupby, minimize="Accuracy Error",
                                                              parameter="Weight on own Belief (Naive)",
                                                              fixed_param_settings={"Configuration": "Naive"})
    other_configs_df = dataframe[dataframe["Configuration"] != "Naive"]
    filtered_naive = dataframe.iloc[optim_idxs]
    new_df = pd.concat([other_configs_df, filtered_naive], ignore_index=True)
    new_df = new_df.replace("Naive", "Naive (Optimal Weight)")
    return new_df


def find_optimal_param_per_setting(dataframe, groupby=None, minimize="Accuracy Error",
                                   parameter="Weight on own Belief (Naive)", fixed_param_settings=None):
    if groupby is None:
        groupby = ["Standard Deviation of Measurements", "Centrality Homogeneity"]
    if fixed_param_settings is not None:
        for p, setting in fixed_param_settings.items():
            dataframe = dataframe[dataframe[p] == setting]
    grouped = dataframe[[*groupby, parameter, minimize]].groupby([*groupby, parameter], sort=False)
    group_values = np.array(list(grouped.groups.keys()))
    print(np.sum([x.shape[1] != 1100 for x in list([g.values.reshape(1, -1) for g in grouped.groups.values()])]))
    print(group_values[[x.shape[1] != 1100 for x in list([g.values.reshape(1, -1) for g in grouped.groups.values()])]])
    group_indices = np.concatenate(list([g.values.reshape(1, -1) for g in grouped.groups.values()]))
    param_values = np.unique(dataframe[parameter])
    group_indices_table = group_indices.reshape((-1, len(param_values), group_indices.shape[-1]))
    param_table = group_values[:, -1].reshape((-1, len(param_values)))
    grouped_mean = grouped.mean().to_numpy().reshape((-1, len(param_values))).astype(np.float64)
    idxs = np.argmin(grouped_mean, axis=-1)
    flat_idx = np.arange(param_table.size, step=param_table.shape[-1]) + idxs.ravel()
    best_weights = param_table.ravel()[flat_idx].reshape(*param_table.shape[:-1])
    # determing right idxs for all original entries with these settings
    base_idxs = np.arange(group_indices_table.shape[0] * group_indices_table.shape[1],
                          step=group_indices_table.shape[-2])
    raveled = (base_idxs + idxs.ravel()) * group_indices_table.shape[-1]
    repeat = np.repeat(raveled, group_indices_table.shape[-1])
    basis_for_big_array = np.tile(np.arange(group_indices_table.shape[-1]), (group_indices_table.shape[0], 1)).reshape(
        -1)
    flat_idx = repeat + basis_for_big_array
    all_fitting_orig_indices = group_indices_table.ravel()[flat_idx].reshape(
        (group_indices_table.shape[0], group_indices_table.shape[-1]))

    other_values = group_values[[i * len(param_values) for i in range(best_weights.shape[0])], :-1]
    value_dict = {parameter: best_weights,
                  minimize: np.min(grouped_mean, axis=-1),
                  }
    for i, n in enumerate(groupby):
        value_dict[n] = other_values[:, i]
    new_df = pandas.DataFrame.from_dict(value_dict)
    return new_df, all_fitting_orig_indices.reshape(-1)


def plot_heatmap_for_columns(dataframe, ax, mapped_value, x_col, y_col, fixed_param_settings=None, cmap=None, map_min=None, map_max=None, logspace=False, colorbar=True, special_plot=False):
    if fixed_param_settings is not None:
        for p, setting in fixed_param_settings.items():
            dataframe = dataframe[dataframe[p] == setting]
    if logspace:
        dataframe["Log("+mapped_value+")"] = np.log(dataframe[mapped_value].to_numpy())
        mapped_value = "Log("+mapped_value+")"
    if cmap is None:
        cmap = get_colorMaps(len(np.unique(dataframe[mapped_value])))
    piv = pd.pivot_table(dataframe.round(4), values=mapped_value, index=[y_col], columns=[x_col], fill_value=dataframe[mapped_value].mean())
    if map_max is None:
        map_max = piv.to_numpy().max()
    if map_min is None:
        map_min = piv.to_numpy().min()
    if not special_plot:
        sns.heatmap(piv, cmap=cmap.get_cmap(), ax=ax, vmin=map_min, vmax=map_max, cbar=colorbar, cbar_kws={'label': mapped_value})
    else:
        xs = piv.columns
        x_bounds = (xs[:-1] + xs[1:]) / 2
        x_bounds = np.concatenate([[2 * x_bounds[0] - x_bounds[1]], x_bounds, [2 * x_bounds[-1] - x_bounds[-2]]])
        ys = piv.index
        y_bounds = (ys[:-1] + ys[1:]) / 2
        y_bounds = np.concatenate([[2 * y_bounds[0] - y_bounds[1]], y_bounds, [2 * y_bounds[-1] - y_bounds[-2]]])
        values = piv.values
        ax.pcolormesh(x_bounds, y_bounds, values, cmap=cmap.get_cmap(), vmin=map_min, vmax=map_max)
    return cmap


def plot_optimal_weight_comparison_for_errors(dataframe, groupby=None):
    if groupby is None:
        groupby = ["Standard Deviation of Measurements", "Centrality Homogeneity"]
    fig = plt.figure(tight_layout=True)
    gs = mpl.gridspec.GridSpec(2, 3, height_ratios=[0.9, 0.1])
    fig.set_size_inches(15, 6)
    axs = [fig.add_subplot(gs[0, j]) for j in range(3)]
    cmap = None
    for i, error in enumerate(["Trueness Error", "Precision Error", "Accuracy Error"]):
        opimized_own_weightdf, _ = find_optimal_param_per_setting(dataframe,
                                                                  groupby=groupby, minimize=error,
                                                                  parameter="Weight on own Belief (Naive)",
                                                                  fixed_param_settings={"Configuration": "Naive"})
        opimized_own_weightdf = opimized_own_weightdf.rename(columns={
            "Weight on own Belief (Naive)": "Optimal Weight on own Belief"
        })
        cmap = plot_heatmap_for_columns(opimized_own_weightdf, axs[i], "Optimal Weight on own Belief",
                                        groupby[0], groupby[1], cmap=cmap, map_min=0.0, map_max=1.0)
        axs[i].set_title("Optimal Weight for " + str(error))
    c_param_options = np.unique(dataframe["Weight on own Belief (Naive)"])
    special_ax = fig.add_subplot(gs[1, :])
    plot_colorbar(special_ax, c_param_options, cmap=cmap.get_cmap())
    special_ax.set_xlabel("Optimal Weight on own Belief")


def load_all_data(data_pathes):
    data = []
    for k, v in tqdm(data_pathes.items()):
        exp_data = load_dataframe(v)
        exp_data["Configuration"] = k
        data.append(exp_data)
    return pd.concat(data, ignore_index=True)


def load_dataframe(path):
    pickle_path = os.path.join(path, "processed_results.pickle")
    with open(pickle_path, "rb") as f:
        df = pickle.load(f)
    return df


def plot_for_overunderconfident_data():
    global default_cmap
    results_plots_path = os.path.join(default_plots_path, "underoverconfident")
    os.makedirs(results_plots_path, exist_ok=True)
    combined_df = load_overunderconfident_data()
    print("Done Loading Under/OverConfident Data!")
    combined_df = combined_df.rename(columns={
        "Trueness_Error": "Trueness Error",
        "Precision_Error": "Precision Error",
        "Accuracy_Error": "Accuracy Error",
        "scalar_param1": "Centrality Homogeneity",
        "std_environment_noise": "Standard Deviation of Measurements",
        "weight_own_belief": "Weight on own Belief (Naive)",
        "range_agent_measurement_noise": "Interval-Width of Agent Measurement Noise Standard Deviation Parameter Distribution",
        "mean_agent_measurement_noise": "Interval-Center of Agent Measurement Noise Standard Deviation Parameter Distribution"
    })

    combined_df[
        "Difference of Interval-Center of Agent Measurement Noise Standard Deviation Parameter Distribution to True Standard Deviation"] = \
    combined_df["Interval-Center of Agent Measurement Noise Standard Deviation Parameter Distribution"] - combined_df[
        "Standard Deviation of Measurements"]

    pandas.set_option("display.max_rows", None, "display.max_columns", None)
    print(combined_df.groupby("Configuration").mean())
    print(combined_df.groupby("Configuration").std())

    combined_df[combined_df["Configuration"] == "Bayes"].plot.hexbin("Interval-Width of Agent Measurement Noise Standard Deviation Parameter Distribution",
                                                                     "Interval-Center of Agent Measurement Noise Standard Deviation Parameter Distribution",
                                                                     gridsize=60)
    save_and_or_show_plot(results_plots_path, "bayes_distr_width_vs_meas_distr_center.png", show_plots)
    combined_df[combined_df["Configuration"] == "BayesCI"].plot.hexbin("Interval-Width of Agent Measurement Noise Standard Deviation Parameter Distribution",
                                                                       "Interval-Center of Agent Measurement Noise Standard Deviation Parameter Distribution",
                                                                       gridsize=60)
    save_and_or_show_plot(results_plots_path, "bayesci_distr_width_vs_meas_distr_center.png", show_plots)

    compare_average_errors_over_network_param(combined_df,
                                              c_parameter="Interval-Width of Agent Measurement Noise Standard Deviation Parameter Distribution")
    save_and_or_show_plot(results_plots_path, "error_cmp_over_uncertainty_distribution_width.png", show_plots)
    compare_average_errors_over_network_param(combined_df, logspace="y",
                                              c_parameter="Interval-Width of Agent Measurement Noise Standard Deviation Parameter Distribution")
    save_and_or_show_plot(results_plots_path, "error_cmp_over_uncertainty_distribution_width_logspace.png", show_plots)
    default_cmap = Berlin_3.get_mpl_colormap()
    compare_average_errors_over_network_param(combined_df,
                                              c_parameter="Difference of Interval-Center of Agent Measurement Noise Standard Deviation Parameter Distribution to True Standard Deviation")
    save_and_or_show_plot(results_plots_path, "error_cmp_over_uncertainty_distribution_center.png", show_plots)
    compare_average_errors_over_network_param(combined_df, logspace="y",
                                              c_parameter="Difference of Interval-Center of Agent Measurement Noise Standard Deviation Parameter Distribution to True Standard Deviation")
    save_and_or_show_plot(results_plots_path, "error_cmp_over_uncertainty_distribution_center_logspace.png", show_plots)

    filtered_df = combined_df[combined_df["Interval-Width of Agent Measurement Noise Standard Deviation Parameter Distribution"] == combined_df["Interval-Width of Agent Measurement Noise Standard Deviation Parameter Distribution"].max()]
    compare_average_errors_over_network_param(filtered_df,
                                              c_parameter="Difference of Interval-Center of Agent Measurement Noise Standard Deviation Parameter Distribution to True Standard Deviation")
    fig = plt.gcf()
    fig.suptitle('For Max Intervalwidth only')
    save_and_or_show_plot(results_plots_path, "max_width_error_cmp_over_uncertainty_distribution_center.png", show_plots)
    compare_average_errors_over_network_param(filtered_df, logspace="y",
                                              c_parameter="Difference of Interval-Center of Agent Measurement Noise Standard Deviation Parameter Distribution to True Standard Deviation")
    fig = plt.gcf()
    fig.suptitle('For Max Intervalwidth only')
    save_and_or_show_plot(results_plots_path, "max_width_error_cmp_over_uncertainty_distribution_center_logspace.png", show_plots)

    filtered_df = combined_df[combined_df["Interval-Width of Agent Measurement Noise Standard Deviation Parameter Distribution"] == combined_df["Interval-Width of Agent Measurement Noise Standard Deviation Parameter Distribution"].min()]
    compare_average_errors_over_network_param(filtered_df,
                                              c_parameter="Difference of Interval-Center of Agent Measurement Noise Standard Deviation Parameter Distribution to True Standard Deviation")
    fig = plt.gcf()
    fig.suptitle('For Min Intervalwidth only')
    save_and_or_show_plot(results_plots_path, "min_width_error_cmp_over_uncertainty_distribution_center.png", show_plots)
    compare_average_errors_over_network_param(filtered_df, logspace="y",
                                              c_parameter="Difference of Interval-Center of Agent Measurement Noise Standard Deviation Parameter Distribution to True Standard Deviation")
    fig = plt.gcf()
    fig.suptitle('For Min Intervalwidth only')
    save_and_or_show_plot(results_plots_path, "min_width_error_cmp_over_uncertainty_distribution_center_logspace.png", show_plots)

    default_cmap = Viridis_3.get_mpl_colormap()
    filtered_df = combined_df[combined_df["Difference of Interval-Center of Agent Measurement Noise Standard Deviation Parameter Distribution to True Standard Deviation"] == combined_df["Difference of Interval-Center of Agent Measurement Noise Standard Deviation Parameter Distribution to True Standard Deviation"].max()]
    compare_average_errors_over_network_param(filtered_df,
                                              c_parameter="Interval-Width of Agent Measurement Noise Standard Deviation Parameter Distribution")
    fig = plt.gcf()
    fig.suptitle('For Max Center only')
    save_and_or_show_plot(results_plots_path, "max_center_error_cmp_over_uncertainty_distribution_width.png", show_plots)
    compare_average_errors_over_network_param(filtered_df, logspace="y",
                                              c_parameter="Interval-Width of Agent Measurement Noise Standard Deviation Parameter Distribution")
    fig = plt.gcf()
    fig.suptitle('For Max Center only')
    save_and_or_show_plot(results_plots_path, "max_center_error_cmp_over_uncertainty_distribution_width_logspace.png", show_plots)

    filtered_df = combined_df[combined_df["Difference of Interval-Center of Agent Measurement Noise Standard Deviation Parameter Distribution to True Standard Deviation"] == combined_df["Difference of Interval-Center of Agent Measurement Noise Standard Deviation Parameter Distribution to True Standard Deviation"].min()]
    compare_average_errors_over_network_param(filtered_df,
                                              c_parameter="Interval-Width of Agent Measurement Noise Standard Deviation Parameter Distribution")
    fig = plt.gcf()
    fig.suptitle('For Min Center only')
    save_and_or_show_plot(results_plots_path, "min_center_error_cmp_over_uncertainty_distribution_width.png", show_plots)
    compare_average_errors_over_network_param(filtered_df, logspace="y",
                                              c_parameter="Interval-Width of Agent Measurement Noise Standard Deviation Parameter Distribution")
    fig = plt.gcf()
    fig.suptitle('For Min Center only')
    save_and_or_show_plot(results_plots_path, "min_center_error_cmp_over_uncertainty_distribution_width_logspace.png", show_plots)

    print("Done Plotting for Under/OverConfident Data")
    del combined_df


def load_overunderconfident_data():
    cache_path = os.path.join(default_cache_path, "combined_dataframe_cached_with_underoverconfident")
    try:
        print("Trying to find cached dataframe for Under/OverConfident data...")
        combined_df = load_python_object(cache_path)
        print("Using Cached Dataframe!")
    except Exception:
        print("Did not find cached dataframe.")
        print("Loading Under/OverConfident Data!")
        data_pathes = {
            "Bayes": os.path.join(default_result_path,
                                  "N100_2023-09-25-16-01-39_overunderconfident_search_Bayes_centralized_random_fixed_mdeg"),
            "BayesCI": os.path.join(default_result_path,
                                    "N100_2023-09-25-16-24-36_overunderconfident_search_BayesCI_centralized_random_fixed_mdeg"),
        }
        combined_df = load_all_data(data_pathes)
        # cache python object
        dump_python_object(combined_df, cache_path)
    return combined_df


def load_and_plot_combined_data():
    results_plots_path = os.path.join(default_plots_path, "new_combined")
    os.makedirs(results_plots_path, exist_ok=True)
    combined_df = load_combined_data()
    print("Done Loading Combined Data!")
    print(combined_df.columns)
    # please_plot_this(combined_df)
    combined_df = rename_columns(combined_df)
    plot_combined_dataframe(combined_df, results_plots_path)

    print("Done Plotting for Combined Data!")
    del combined_df


def load_combined_data():
    cache_path = os.path.join(default_cache_path, "new_combined_dataframe_cached_combined")
    try:
        print("Trying to find cached dataframe for combined data...")
        combined_df = load_python_object(cache_path)
        print("Using Cached Dataframe!")
    except Exception:
        print("Did not find cached dataframe.")
        print("Loading Combined Data!")
        data_pathes = {
            "Bayes": os.path.join(default_result_path,
                                  "N100_2023-09-25-13-42-52_certainty_search_Bayes_centralized_random_fixed_mdeg"),
            "BayesCI": os.path.join(default_result_path,
                                    "N100_2023-09-25-14-21-29_certainty_search_BayesCI_centralized_random_fixed_mdeg"),
            "Naive": os.path.join(default_result_path,
                                  "N100_2023-09-15-16-19-36_certainty_search_Naive_centralized_random_fixed_mdeg"),
            "Naive (Locally Optimal Weighting)": os.path.join(default_result_path,
                                                              "N100_2023-09-25-15-09-30_certainty_search_NaiveLO_centralized_random_fixed_mdeg"),
        }
        combined_df = load_all_data(data_pathes)
        # cache python object
        dump_python_object(combined_df, cache_path)
    return combined_df


def plot_for_correl_combined_data():
    results_plots_path = os.path.join(default_plots_path, "correl_meas_combined")
    os.makedirs(results_plots_path, exist_ok=True)
    combined_df = load_correl_data()
    print("Done Loading correl Data!")
    print(combined_df.columns)
    # please_plot_this(combined_df)
    combined_df = rename_columns(combined_df)
    only_optimal_weights_df = filter_for_optimal_weights_in_naive(combined_df, ["Centrality Homogeneity",
                                                                                "Interval-Width of Agent Measurement Noise Standard Deviation Parameter Distribution", ])
    plot_optimal_weight_comparison_for_errors(combined_df, groupby=["Centrality Homogeneity",
                                                                    "Interval-Width of Agent Measurement Noise Standard Deviation Parameter Distribution", ])
    save_and_or_show_plot(results_plots_path, "optimal_own_weight_for_different_errors.png", show_plots)

    compare_average_errors_over_network_param(combined_df, c_parameter="Centrality Homogeneity")
    save_and_or_show_plot(results_plots_path, "error_cmp_over_network.png", show_plots)
    compare_average_errors_over_network_param(combined_df, logspace="y", c_parameter="Centrality Homogeneity")
    save_and_or_show_plot(results_plots_path, "error_cmp_over_network_logspace.png", show_plots)
    save_and_or_show_plot(results_plots_path, "error_cmp_over_env_noise_logspace.png", show_plots)
    compare_average_errors_over_network_param(combined_df, c_parameter="Weight on own Belief (Naive)")
    save_and_or_show_plot(results_plots_path, "error_cmp_over_own_weight.png", show_plots)
    compare_average_errors_over_network_param(combined_df, logspace="y", c_parameter="Weight on own Belief (Naive)")
    save_and_or_show_plot(results_plots_path, "error_cmp_over_own_weight_logspace.png", show_plots)
    compare_average_errors_over_network_param(only_optimal_weights_df, c_parameter="Centrality Homogeneity")
    save_and_or_show_plot(results_plots_path, "optimal_weight_error_cmp_over_network.png", show_plots)
    compare_average_errors_over_network_param(only_optimal_weights_df, logspace="y",
                                              c_parameter="Centrality Homogeneity")
    save_and_or_show_plot(results_plots_path, "optimal_weight_error_cmp_over_network_logspace.png", show_plots)
    compare_average_errors_over_network_param(combined_df,
                                              c_parameter="correlation_network_information")
    save_and_or_show_plot(results_plots_path, "error_cmp_over_correlation.png", show_plots)
    compare_average_errors_over_network_param(combined_df, logspace="y",
                                              c_parameter="correlation_network_information")
    save_and_or_show_plot(results_plots_path, "error_cmp_over_correlation_logspace.png", show_plots)
    compare_average_errors_over_network_param(combined_df, c_parameter="Weight on own Belief (Naive)")
    save_and_or_show_plot(results_plots_path, "error_cmp_over_own_weight.png", show_plots)
    compare_average_errors_over_network_param(combined_df, logspace="y", c_parameter="Weight on own Belief (Naive)")
    save_and_or_show_plot(results_plots_path, "error_cmp_over_own_weight_logspace.png", show_plots)
    compare_average_errors_over_network_param(only_optimal_weights_df,
                                              c_parameter="correlation_network_information")
    save_and_or_show_plot(results_plots_path, "optimal_own_weight_error_cmp_over_correlation_network_information.png",
                          show_plots)
    compare_average_errors_over_network_param(only_optimal_weights_df, logspace="y",
                                              c_parameter="correlation_network_information")
    save_and_or_show_plot(results_plots_path,
                          "optimal_own_weight_error_cmp_over_correlation_network_information_logspace.png", show_plots)
    filtered_df = combined_df
    filtered_optimal_df = only_optimal_weights_df
    intial_timestep_only = filtered_df[filtered_df["Timestep"] == 1]
    optimal_intial_timestep_only = filtered_optimal_df[filtered_optimal_df["Timestep"] == 1]
    for i in range(1, 10):
        compare_average_errors_over_two_param(filtered_df, x_param="Centrality Homogeneity",
                                              y_param="correlation_network_information",
                                              timestep=i)
        save_and_or_show_plot(results_plots_path, "correlation_heatmaps_t" + str(i) + ".png", show_plots)
        compare_average_errors_over_two_param(filtered_optimal_df, x_param="Centrality Homogeneity",
                                              y_param="correlation_network_information",
                                              timestep=i)
        save_and_or_show_plot(results_plots_path, "optimal_correlation_heatmaps_t" + str(i) + ".png",
                              show_plots)
        compare_average_errors_over_two_param(filtered_df, x_param="Centrality Homogeneity",
                                              y_param="correlation_network_information",
                                              timestep=i, logspace=True)
        save_and_or_show_plot(results_plots_path, "correlation_heatmaps_logspace_t" + str(i) + ".png", show_plots)
        compare_average_errors_over_two_param(filtered_optimal_df, x_param="Centrality Homogeneity",
                                              y_param="correlation_network_information",
                                              timestep=i, logspace=True)
        save_and_or_show_plot(results_plots_path, "optimal_correlation_heatmaps_logspace_t" + str(i) + ".png",
                              show_plots)

        this_timestep_only = filtered_df[filtered_df["Timestep"] == i]
        this_timestep_only["Accuracy Error"] = this_timestep_only["Accuracy Error"].to_numpy() - (
            intial_timestep_only["Accuracy Error"]).to_numpy()
        this_timestep_only["Trueness Error"] = this_timestep_only["Trueness Error"].to_numpy() - (
            intial_timestep_only["Trueness Error"]).to_numpy()
        this_timestep_only["Precision Error"] = this_timestep_only["Precision Error"].to_numpy() - intial_timestep_only[
            "Precision Error"].to_numpy()

        optimal_this_timestep_only = filtered_optimal_df[filtered_optimal_df["Timestep"] == i]
        optimal_this_timestep_only["Accuracy Error"] = optimal_this_timestep_only["Accuracy Error"].to_numpy() - \
                                                       optimal_intial_timestep_only["Accuracy Error"].to_numpy()
        optimal_this_timestep_only["Trueness Error"] = optimal_this_timestep_only["Trueness Error"].to_numpy() - \
                                                       optimal_intial_timestep_only["Trueness Error"].to_numpy()
        optimal_this_timestep_only["Precision Error"] = optimal_this_timestep_only["Precision Error"].to_numpy() - \
                                                        optimal_intial_timestep_only["Precision Error"].to_numpy()

        compare_average_errors_over_two_param(this_timestep_only, x_param="Centrality Homogeneity",
                                              y_param="correlation_network_information",
                                              timestep=i)
        save_and_or_show_plot(results_plots_path, "rate_of_error_correlation_heatmaps_t" + str(i) + ".png",
                              show_plots)
        compare_average_errors_over_two_param(optimal_this_timestep_only, x_param="Centrality Homogeneity",
                                              y_param="correlation_network_information",
                                              timestep=i)
        save_and_or_show_plot(results_plots_path,
                              "optimal_rate_of_error_correlation_heatmaps_t" + str(i) + ".png", show_plots)

    print("Done Plotting for correl Data!")
    del combined_df


def load_correl_data():
    cache_path = os.path.join(default_cache_path, "correl_combined_dataframe_cached_combined")
    try:
        print("Trying to find cached dataframe for correl meas data...")
        combined_df = load_python_object(cache_path)
        print("Using Cached Dataframe!")
    except Exception:
        print("Did not find cached dataframe.")
        print("Loading Combined Data!")
        data_pathes = {
            "Bayes": os.path.join(default_result_path,
                                  "N100_2023-09-30-01-13-40_correlation_search_Bayes_centralized_random_fixed_mdeg"),
            "BayesCI": os.path.join(default_result_path,
                                    "N100_2023-09-30-01-18-18_correlation_search_BayesCI_centralized_random_fixed_mdeg"),
            "Naive": os.path.join(default_result_path,
                                  "N100_2023-09-29-23-36-25_correlation_search_Naive_centralized_random_fixed_mdeg"),
            "Naive (Locally Optimal Weighting)": os.path.join(default_result_path,
                                                              "N100_2023-09-30-01-22-49_correlation_search_NaiveLO_centralized_random_fixed_mdeg"),
        }
        combined_df = load_all_data(data_pathes)
        # cache python object
        dump_python_object(combined_df, cache_path)
    return combined_df


def plot_combined_dataframe(combined_df, results_plots_path):
    only_optimal_weights_df = filter_for_optimal_weights_in_naive(combined_df, ["Centrality Homogeneity",
                                                                                "Interval-Width of Agent Measurement Noise Standard Deviation Parameter Distribution", ])
    plot_optimal_weight_comparison_for_errors(combined_df, groupby=["Centrality Homogeneity",
                                                                    "Interval-Width of Agent Measurement Noise Standard Deviation Parameter Distribution", ])
    save_and_or_show_plot(results_plots_path, "optimal_own_weight_for_different_errors.png", show_plots)

    compare_average_errors_over_network_param(combined_df, c_parameter="Centrality Homogeneity")
    save_and_or_show_plot(results_plots_path, "error_cmp_over_network.png", show_plots)
    compare_average_errors_over_network_param(combined_df, logspace="y", c_parameter="Centrality Homogeneity")
    save_and_or_show_plot(results_plots_path, "error_cmp_over_network_logspace.png", show_plots)
    save_and_or_show_plot(results_plots_path, "error_cmp_over_env_noise_logspace.png", show_plots)
    compare_average_errors_over_network_param(combined_df, c_parameter="Weight on own Belief (Naive)")
    save_and_or_show_plot(results_plots_path, "error_cmp_over_own_weight.png", show_plots)
    compare_average_errors_over_network_param(combined_df, logspace="y", c_parameter="Weight on own Belief (Naive)")
    save_and_or_show_plot(results_plots_path, "error_cmp_over_own_weight_logspace.png", show_plots)
    compare_average_errors_over_network_param(only_optimal_weights_df, c_parameter="Centrality Homogeneity")
    save_and_or_show_plot(results_plots_path, "optimal_weight_error_cmp_over_network.png", show_plots)
    compare_average_errors_over_network_param(only_optimal_weights_df, logspace="y",
                                              c_parameter="Centrality Homogeneity")
    save_and_or_show_plot(results_plots_path, "optimal_weight_error_cmp_over_network_logspace.png", show_plots)
    compare_average_errors_over_network_param(combined_df,
                                              c_parameter="Interval-Width of Agent Measurement Noise Standard Deviation Parameter Distribution")
    save_and_or_show_plot(results_plots_path, "error_cmp_over_uncertainty_distribution_width.png", show_plots)
    compare_average_errors_over_network_param(combined_df, logspace="y",
                                              c_parameter="Interval-Width of Agent Measurement Noise Standard Deviation Parameter Distribution")
    save_and_or_show_plot(results_plots_path, "error_cmp_over_uncertainty_distribution_width_logspace.png", show_plots)
    compare_average_errors_over_network_param(combined_df, c_parameter="Weight on own Belief (Naive)")
    save_and_or_show_plot(results_plots_path, "error_cmp_over_own_weight.png", show_plots)
    compare_average_errors_over_network_param(combined_df, logspace="y", c_parameter="Weight on own Belief (Naive)")
    save_and_or_show_plot(results_plots_path, "error_cmp_over_own_weight_logspace.png", show_plots)
    compare_average_errors_over_network_param(only_optimal_weights_df,
                                              c_parameter="Interval-Width of Agent Measurement Noise Standard Deviation Parameter Distribution")
    save_and_or_show_plot(results_plots_path, "optimal_own_weight_error_cmp_over_uncertainty_distribution_width.png",
                          show_plots)
    compare_average_errors_over_network_param(only_optimal_weights_df, logspace="y",
                                              c_parameter="Interval-Width of Agent Measurement Noise Standard Deviation Parameter Distribution")
    save_and_or_show_plot(results_plots_path,
                          "optimal_own_weight_error_cmp_over_uncertainty_distribution_width_logspace.png", show_plots)
    filtered_df = combined_df
    filtered_optimal_df = only_optimal_weights_df
    intial_timestep_only = filtered_df[filtered_df["Timestep"] == 1]
    optimal_intial_timestep_only = filtered_optimal_df[filtered_optimal_df["Timestep"] == 1]
    for i in range(1, 10):
        compare_average_errors_over_two_param(filtered_df, x_param="Centrality Homogeneity",
                                              y_param="Interval-Width of Agent Measurement Noise Standard Deviation Parameter Distribution",
                                              timestep=i)
        save_and_or_show_plot(results_plots_path, "heterogeneity_heatmaps_t" + str(i) + ".png", show_plots)
        compare_average_errors_over_two_param(filtered_optimal_df, x_param="Centrality Homogeneity",
                                              y_param="Interval-Width of Agent Measurement Noise Standard Deviation Parameter Distribution",
                                              timestep=i)
        save_and_or_show_plot(results_plots_path, "optimal_naive_heterogeneity_heatmaps_t" + str(i) + ".png",
                              show_plots)
        compare_average_errors_over_two_param(filtered_df, x_param="Centrality Homogeneity",
                                              y_param="Interval-Width of Agent Measurement Noise Standard Deviation Parameter Distribution",
                                              timestep=i, logspace=True)
        save_and_or_show_plot(results_plots_path, "heterogeneity_heatmaps_logspace_t" + str(i) + ".png", show_plots)
        compare_average_errors_over_two_param(filtered_optimal_df, x_param="Centrality Homogeneity",
                                              y_param="Interval-Width of Agent Measurement Noise Standard Deviation Parameter Distribution",
                                              timestep=i, logspace=True)
        save_and_or_show_plot(results_plots_path, "optimal_naive_heterogeneity_heatmaps_logspace_t" + str(i) + ".png",
                              show_plots)

        this_timestep_only = filtered_df[filtered_df["Timestep"] == i]
        this_timestep_only["Accuracy Error"] = this_timestep_only["Accuracy Error"].to_numpy() - (
        intial_timestep_only["Accuracy Error"]).to_numpy()
        this_timestep_only["Trueness Error"] = this_timestep_only["Trueness Error"].to_numpy() - (
        intial_timestep_only["Trueness Error"]).to_numpy()
        this_timestep_only["Precision Error"] = this_timestep_only["Precision Error"].to_numpy() - intial_timestep_only[
            "Precision Error"].to_numpy()

        optimal_this_timestep_only = filtered_optimal_df[filtered_optimal_df["Timestep"] == i]
        optimal_this_timestep_only["Accuracy Error"] = optimal_this_timestep_only["Accuracy Error"].to_numpy() - \
                                                       optimal_intial_timestep_only["Accuracy Error"].to_numpy()
        optimal_this_timestep_only["Trueness Error"] = optimal_this_timestep_only["Trueness Error"].to_numpy() - \
                                                       optimal_intial_timestep_only["Trueness Error"].to_numpy()
        optimal_this_timestep_only["Precision Error"] = optimal_this_timestep_only["Precision Error"].to_numpy() - \
                                                        optimal_intial_timestep_only["Precision Error"].to_numpy()

        compare_average_errors_over_two_param(this_timestep_only, x_param="Centrality Homogeneity",
                                              y_param="Interval-Width of Agent Measurement Noise Standard Deviation Parameter Distribution",
                                              timestep=i)
        save_and_or_show_plot(results_plots_path, "diff_in_error_heterogeneity_heatmaps_t" + str(i) + ".png",
                              show_plots)
        compare_average_errors_over_two_param(optimal_this_timestep_only, x_param="Centrality Homogeneity",
                                              y_param="Interval-Width of Agent Measurement Noise Standard Deviation Parameter Distribution",
                                              timestep=i)
        save_and_or_show_plot(results_plots_path,
                              "optimal_naive_diff_in_error_heterogeneity_heatmaps_t" + str(i) + ".png", show_plots)


def rename_columns(combined_df):
    combined_df = combined_df.rename(columns={
        "Trueness_Error": "Trueness Error",
        "Precision_Error": "Precision Error",
        "Accuracy_Error": "Accuracy Error",
        "scalar_param1": "Centrality Homogeneity",
        "std_environment_noise": "Standard Deviation of Measurements",
        "weight_own_belief": "Weight on own Belief (Naive)",
        "range_agent_measurement_noise": "Interval-Width of Agent Measurement Noise Standard Deviation Parameter Distribution",
        "mean_agent_measurement_noise": "Interval-Center of Agent Measurement Noise Standard Deviation Parameter Distribution"
    })
    return combined_df


def paper_plots():
    results_plots_path = os.path.join(default_plots_path, "paper_plots")
    os.makedirs(results_plots_path, exist_ok=True)
    cache_path = os.path.join(default_cache_path, "combined_dataframe_cached_with_underoverconfident")
    combined_df = load_python_object(cache_path)
    combined_df = rename_columns(combined_df)
    combined_df.replace(2.5000999999999998, 2.5, inplace=True)
    combined_df = combined_df[combined_df["Timestep"] == 9]
    combined_df[
        "Difference of Interval-Center of Agent Measurement Noise Standard Deviation Parameter Distribution to True Standard Deviation"] = \
    combined_df["Interval-Center of Agent Measurement Noise Standard Deviation Parameter Distribution"] - combined_df[
        "Standard Deviation of Measurements"]
    fig, axs = plt.subplots(1, 3, gridspec_kw={"width_ratios": [1.0, 1.0, 0.1]})
    error_max1, error_min1 = get_limits(combined_df[combined_df["Configuration"] == "Bayes"], col1="Interval-Width of Agent Measurement Noise Standard Deviation Parameter Distribution", col2="Difference of Interval-Center of Agent Measurement Noise Standard Deviation Parameter Distribution to True Standard Deviation")
    error_max2, error_min2 = get_limits(combined_df[combined_df["Configuration"] == "BayesCI"], col1="Interval-Width of Agent Measurement Noise Standard Deviation Parameter Distribution", col2="Difference of Interval-Center of Agent Measurement Noise Standard Deviation Parameter Distribution to True Standard Deviation")
    error_max = np.log(max([error_max1, error_max2,]))
    error_min = np.log(0.1)
    CMap = Inferno_20.get_mpl_colormap()
    norm = mpl.colors.Normalize(vmin=error_min, vmax=error_max)
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=CMap)
    plot_heatmap_for_columns(combined_df, axs[0],
                             mapped_value="Accuracy Error",
                             x_col="Difference of Interval-Center of Agent Measurement Noise Standard Deviation Parameter Distribution to True Standard Deviation",
                             y_col="Interval-Width of Agent Measurement Noise Standard Deviation Parameter Distribution",
                             cmap=cmap, colorbar=False, logspace=True,
                             # map_min=error_min, map_max=error_max,
                             fixed_param_settings={"Configuration": "Bayes"})
    plot_heatmap_for_columns(combined_df, axs[1],
                             mapped_value="Accuracy Error",
                             x_col="Difference of Interval-Center of Agent Measurement Noise Standard Deviation Parameter Distribution to True Standard Deviation",
                             y_col="Interval-Width of Agent Measurement Noise Standard Deviation Parameter Distribution",
                             cmap=cmap, colorbar=False, logspace=True,
                             # map_min=error_min, map_max=error_max,
                             fixed_param_settings={"Configuration": "BayesCI"})
    mpl.colorbar.ColorbarBase(axs[2], cmap=cmap.get_cmap(), values=np.linspace(error_min, error_max, 1000), orientation='vertical')
    axs[0].set_xlabel("Average Offset of Initial Certainty")
    axs[1].set_xlabel("Average Offset of Initial Certainty")
    axs[0].set_title("Bayes")
    axs[1].set_title("BayesCI")
    axs[0].set_ylabel("Heterogeneity of Initial Information")
    axs[1].get_yaxis().set_visible(False)
    axs[2].set_ylabel("Log(Average Final Accuracy Error)")
    axs[2].yaxis.set_ticks_position('left')
    axs[2].yaxis.set_label_position("left")
    fig.set_size_inches(10, 5)
    plt.tight_layout(w_pad=1)
    save_and_or_show_plot(results_plots_path,
                          "uncertainty_model.png", show_plots)


def get_limits(local_df,
               col1="Interval-Width of Agent Measurement Noise Standard Deviation Parameter Distribution",
               col2="Centrality Homogeneity", mapped_col="Accuracy Error"):
    piv = pd.pivot_table(local_df.round(4), values=mapped_col,
                         index=[col1],
                         columns=[col2], fill_value=local_df[mapped_col].mean())
    error_min = piv.to_numpy().min()
    error_max = piv.to_numpy().max()
    return error_max, error_min


def please_plot_this(df):
    tmp_val = df[df.scalar_param1==0].Netw_mean_degree
    plt.plot(tmp_val,'.',)
    plt.show()


if __name__ == "__main__":
    default_result_path = "data/processed_files"
    default_plots_path = "plots"
    default_cache_path = "cached_dfs"
    show_plots = False

    plot_for_overunderconfident_data()
    load_and_plot_combined_data()
    plot_for_correl_combined_data()
    paper_plots()
