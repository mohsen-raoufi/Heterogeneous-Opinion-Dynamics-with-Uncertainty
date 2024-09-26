from algorithm_comparison_plots import set_share_axes, plot_heatmap_for_columns, get_limits, filter_for_optimal_weights_in_naive
from util import dump_python_object, load_python_object, save_and_or_show_plot
import pandas
import matplotlib.pyplot as plt
import numpy as np
from palettable.matplotlib import Viridis_3, Plasma_20, Inferno_20
from palettable.cartocolors.sequential import Sunset_7
import matplotlib as mpl
import os
import pandas as pd
import seaborn as sns


def het_het_precision_plot(local_df, results_plots_path, log=False):
    fontsize_title = 11
    fig, axs = plt.subplots(1, 5, gridspec_kw={"width_ratios": [1.0, 1.0, 1.0, 1.0, 0.1]})
    set_share_axes(axs[:3],sharey=True, sharex=True)
    last_t = local_df[local_df["Timestep"] == 10]
    rate_of_acc_err = last_t["Precision Error"].to_numpy() / local_df[local_df["Timestep"] == 1]["Precision Error"].to_numpy()
    local_df = last_t
    local_df["Precision Error"] = rate_of_acc_err
    mapping = local_df[["Netw_std_degree", "Centrality Homogeneity"]].groupby("Centrality Homogeneity").mean()
    local_df["HetCent"] = [mapping.values[mapping.index == i][0][0] for i in local_df["Centrality Homogeneity"]]
    error_max1, error_min1 = get_limits(local_df[local_df["Configuration"] == "Bayes"], mapped_col="Precision Error")
    error_max2, error_min2 = get_limits(local_df[local_df["Configuration"] == "BayesCI"], mapped_col="Precision Error")
    error_max3, error_min3 = 0.0000000001, 1000.0
    error_max4, error_min4 = get_limits(local_df[local_df["Configuration"] == "Naive (Optimal Weight)"], mapped_col="Precision Error")
    error_max = max([error_max1, error_max2, error_max3, error_max4])
    error_min = min([error_min1, error_min2, error_min3, error_min4])
    if log:
        error_max = np.log(error_max)
        error_min = np.log(error_min)
    CMap = Sunset_7.get_mpl_colormap()
    norm = mpl.colors.Normalize(vmin=error_min, vmax=error_max)
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=CMap)
    plot_heatmap_for_columns(local_df, axs[0],
                             mapped_value="Precision Error",
                             y_col="HetCent",
                             x_col="Interval-Width of Agent Measurement Noise Standard Deviation Parameter Distribution",
                             cmap=cmap, colorbar=False, logspace=log, special_plot=True,
                             map_min=error_min, map_max=error_max,
                             fixed_param_settings={"Configuration": "Naive (Optimal Weight)"})
    plot_heatmap_for_columns(local_df, axs[1],
                             mapped_value="Precision Error",
                             y_col="HetCent",
                             x_col="Interval-Width of Agent Measurement Noise Standard Deviation Parameter Distribution",
                             cmap=cmap, colorbar=False, logspace=log, special_plot=True,
                             map_min=error_min, map_max=error_max,
                             fixed_param_settings={"Configuration": "Naive (Locally Optimal Weighting)"})
    plot_heatmap_for_columns(local_df, axs[2],
                             mapped_value="Precision Error",
                             y_col="HetCent",
                             x_col="Interval-Width of Agent Measurement Noise Standard Deviation Parameter Distribution",
                             cmap=cmap, colorbar=False, logspace=log, special_plot=True,
                             map_min=error_min, map_max=error_max,
                             fixed_param_settings={"Configuration": "BayesCI"})
    plot_heatmap_for_columns(local_df, axs[3],
                             mapped_value="Precision Error",
                             y_col="HetCent",
                             x_col="Interval-Width of Agent Measurement Noise Standard Deviation Parameter Distribution",
                             cmap=cmap, colorbar=False, logspace=log, special_plot=True,
                             map_min=error_min, map_max=error_max,
                             fixed_param_settings={"Configuration": "Bayes"})
    mpl.colorbar.ColorbarBase(axs[4], cmap=cmap.get_cmap(), values=np.linspace(error_min, error_max, 1000), orientation='vertical')
    axs[0].set_ylabel("Centrality heterogeneity\n$\sigma_\mathrm{net}$")
    axs[0].set_title("Naive Averaging\nNA-OUW", fontsize=fontsize_title)
    axs[1].set_title("Naive Averaging\nNA-LEW", fontsize=fontsize_title)
    axs[2].set_title("Bayesian Inference\nBI-UD", fontsize=fontsize_title)
    axs[3].set_title("Bayesian Inference\nBI-AI", fontsize=fontsize_title)
    axs[0].set_xlabel("Information heterogeneity\n$p_\mathrm{inf}$")
    axs[1].set_xlabel("Information heterogeneity\n$p_\mathrm{inf}$")
    axs[2].set_xlabel("Information heterogeneity\n$p_\mathrm{inf}$")
    axs[3].set_xlabel("Information heterogeneity\n$p_\mathrm{inf}$")
    axs[1].get_yaxis().set_visible(False)
    axs[2].get_yaxis().set_visible(False)
    axs[3].get_yaxis().set_visible(False)
    axs[4].set_ylabel("Normalized final precision error\n${\\overline{E}^{P}_{t=T}}$  /  ${\\overline{E}^{P}_{t=0}}$")
    fig.set_size_inches(10.2, 3)
    plt.tight_layout(w_pad=1)
    save_and_or_show_plot(results_plots_path,
                          "het_het_prec.pdf", show_plots)


def bar_plot(local_df):
    fig, axs = plt.subplots(2, 2, gridspec_kw={"width_ratios": [1.0, 1.0]}, sharex="all", sharey="all")
    local_df = local_df.replace("Naive", "NA")
    local_df = local_df.replace("Bayes", "BI-AI")
    local_df = local_df.replace("BayesCI", "BI-UD")
    local_df = local_df.replace("Naive (Optimal Weight)", "NA-OUW")
    local_df = local_df.replace("Naive (Locally Optimal Weighting)", "NA-LEW")


    last_t = local_df[local_df["Timestep"] == 10]
    rate_of_prec_err = last_t["Precision Error"].to_numpy() / local_df[local_df["Timestep"] == 1][
        "Accuracy Error"].to_numpy()
    rate_of_true_err = last_t["Trueness Error"].to_numpy() / local_df[local_df["Timestep"] == 1][
        "Accuracy Error"].to_numpy()
    local_df = last_t
    local_df["Precision Error"] = rate_of_prec_err
    local_df["Trueness Error"] = rate_of_true_err

    local_df= local_df.rename(columns={
        "Trueness Error": "Normalized final trueness error ${\\overline{E}^{T}_{t=T}}$  /  ${\\overline{E}^{A}_{t=0}}$",
        "Precision Error": "Normalized final precision error ${\\overline{E}^{P}_{t=T}}$  /  ${\\overline{E}^{A}_{t=0}}$",
    })

    no_net_het_value = local_df["Centrality Homogeneity"].max()
    net_het_vals = np.unique(local_df["Centrality Homogeneity"])
    net_het_vals.sort()
    high_net_het_value = net_het_vals[3]
    high_inf_het_val = local_df[
        "Interval-Width of Agent Measurement Noise Standard Deviation Parameter Distribution"].max()
    no_inf_het_val = local_df[
        "Interval-Width of Agent Measurement Noise Standard Deviation Parameter Distribution"].min()


    high_net_het = local_df[(local_df["Centrality Homogeneity"] == high_net_het_value) &
                            (local_df["Interval-Width of Agent Measurement Noise Standard Deviation Parameter Distribution"] == no_inf_het_val)]

    high_inf_het = local_df[(local_df["Centrality Homogeneity"] == no_net_het_value) &
                            (local_df[
                                 "Interval-Width of Agent Measurement Noise Standard Deviation Parameter Distribution"] ==
                             high_inf_het_val)]
    no_het = local_df[(local_df["Centrality Homogeneity"] == no_net_het_value) &
                      (local_df["Interval-Width of Agent Measurement Noise Standard Deviation Parameter Distribution"] == no_inf_het_val)]
    both_het = local_df[(local_df["Centrality Homogeneity"] == high_net_het_value) &
                        (local_df["Interval-Width of Agent Measurement Noise Standard Deviation Parameter Distribution"] ==
                         high_inf_het_val)]

    rot = 0
    clm_multiplier = 1.96   # 95% clm
    capsize = 6

    het_inf_ax = axs[1, 1]
    het_both_ax = axs[0, 1]
    homo_ax = axs[1, 0]
    het_cent_ax = axs[0, 0]

    het_inf_ax.set_title("High information heterogeneity\n$p_\mathrm{inf}="+str(high_inf_het_val)+"$    $\sigma_\mathrm{net}="+str(np.round(no_het["Netw_std_degree"].median()))+"$")
    colors = ["#3781B7", "#ED6D49"]
    high_inf_het.groupby("Configuration").mean().plot.bar(ax=het_inf_ax, y=["Normalized final trueness error ${\\overline{E}^{T}_{t=T}}$  /  ${\\overline{E}^{A}_{t=0}}$", "Normalized final precision error ${\\overline{E}^{P}_{t=T}}$  /  ${\\overline{E}^{A}_{t=0}}$"],
                                                          stacked=True, legend=False, rot=rot, capsize=capsize,
                                                          yerr=clm_multiplier*high_inf_het.groupby("Configuration").sem(), color=colors)

    het_both_ax.set_title("High information and centrality heterogeneity\n$p_\mathrm{inf}="+str(high_inf_het_val)+"$    $\sigma_\mathrm{net}="+str(np.round(both_het["Netw_std_degree"].median()))+"$")
    both_het.groupby("Configuration").mean().plot.bar(ax=het_both_ax, y=["Normalized final trueness error ${\\overline{E}^{T}_{t=T}}$  /  ${\\overline{E}^{A}_{t=0}}$", "Normalized final precision error ${\\overline{E}^{P}_{t=T}}$  /  ${\\overline{E}^{A}_{t=0}}$"],
                                                      stacked=True, legend=False, rot=rot, capsize=capsize,
                                                      yerr=clm_multiplier*both_het.groupby("Configuration").sem(), color=colors)

    homo_ax.set_title("Information and centrality homogeneity\n$p_\mathrm{inf}="+str(no_inf_het_val)+"$    $\sigma_\mathrm{net}="+str(np.round(no_het["Netw_std_degree"].median()))+"$")
    no_het.groupby("Configuration").mean().plot.bar(ax=homo_ax, y=["Normalized final trueness error ${\\overline{E}^{T}_{t=T}}$  /  ${\\overline{E}^{A}_{t=0}}$", "Normalized final precision error ${\\overline{E}^{P}_{t=T}}$  /  ${\\overline{E}^{A}_{t=0}}$"],
                                                    stacked=True, legend=False, rot=rot, capsize=capsize,
                                                    yerr=clm_multiplier*no_het.groupby("Configuration").sem(), color=colors)

    het_cent_ax.set_title("High centrality heterogeneity\n$p_\mathrm{inf}="+str(no_inf_het_val)+"$    $\sigma_\mathrm{net}="+str(np.round(high_net_het["Netw_std_degree"].median()))+"$")
    high_net_het.groupby("Configuration").mean().plot.bar(ax=het_cent_ax, y=["Normalized final trueness error ${\\overline{E}^{T}_{t=T}}$  /  ${\\overline{E}^{A}_{t=0}}$", "Normalized final precision error ${\\overline{E}^{P}_{t=T}}$  /  ${\\overline{E}^{A}_{t=0}}$"],
                                                          stacked=True, legend=False, rot=rot, capsize=capsize,
                                                          yerr=clm_multiplier*high_net_het.groupby("Configuration").sem(), color=colors)

    axs[1, 0].yaxis.grid(True)
    axs[1, 1].yaxis.grid(True)
    axs[0, 0].yaxis.grid(True)
    axs[0, 1].yaxis.grid(True)
    axs[0, 0].set_ylabel("Mean error with 95% CLM")
    axs[1, 0].set_ylabel("Mean error with 95% CLM")
    axs[1, 0].set(xlabel=None)
    axs[1, 1].set(xlabel=None)
    fig.set_size_inches(11.5, 5.5)
    plt.tight_layout()
    save_and_or_show_plot(results_plots_path,
                          "hets_method_comp.pdf", show_plots)
    legend = axs[1, 0].legend(ncol=2, loc=9, bbox_to_anchor=(1.0,-0.5))
    plt.tight_layout()
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array([-5,-5,5,5])))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(os.path.join(results_plots_path, "bar_plot_legend.pdf"), dpi="figure", bbox_inches=bbox)



def rename_columns(combined_df):
    combined_df = combined_df.rename(columns={
        "Trueness_Error": "Trueness Error",
        "Precision_Error": "Precision Error",
        "Accuracy_Error": "Accuracy Error",
        "scalar_param1": "Centrality Homogeneity",
        "std_environment_noise": "Standard Deviation of Measurements",
        "weight_own_belief": "Weight on own Belief (Naive)",
        "range_agent_measurement_noise": "Interval-Width of Agent Measurement Noise Standard Deviation Parameter Distribution",
        "mean_agent_measurement_noise": "Interval-Center of Agent Measurement Noise Standard Deviation Parameter Distribution",
        "correlation_network_information": "Correlation Coefficient",
    })
    return combined_df


def underover_heatmap_plot(dataframe, ax, mapped_value, x_col, y_col, fixed_param_settings=None, cmap=None, map_min=None, map_max=None):
    if fixed_param_settings is not None:
        for p, setting in fixed_param_settings.items():
            dataframe = dataframe[dataframe[p] == setting]
    piv = pd.pivot_table(dataframe.round(4), values=mapped_value, index=[y_col], columns=[x_col], fill_value=dataframe[mapped_value].mean())
    if map_max is None:
        map_max = piv.to_numpy().max()
    if map_min is None:
        map_min = piv.to_numpy().min()
    xs = piv.columns
    x_bounds = (xs[:-1] + xs[1:]) / 2
    x_bounds = np.concatenate([[2 * x_bounds[0] - x_bounds[1]], x_bounds, [2 * x_bounds[-1] - x_bounds[-2]]])
    ys = piv.index
    y_bounds = (ys[:-1] + ys[1:]) / 2
    y_bounds = np.concatenate([[2 * y_bounds[0] - y_bounds[1]], y_bounds, [2 * y_bounds[-1] - y_bounds[-2]]])
    values = piv.values
    ax.pcolormesh(x_bounds, y_bounds, values, cmap=cmap.get_cmap(), vmin=map_min, vmax=map_max)

def correl_heatmap_plot(dataframe, ax, mapped_value, x_col, y_col, fixed_param_settings=None, cmap=None, map_min=None, map_max=None):
    if fixed_param_settings is not None:
        for p, setting in fixed_param_settings.items():
            dataframe = dataframe[dataframe[p] == setting]
    piv = pd.pivot_table(dataframe.round(4), values=mapped_value, index=[y_col], columns=[x_col], fill_value=dataframe[mapped_value].mean())
    if map_max is None:
        map_max = piv.to_numpy().max()
    if map_min is None:
        map_min = piv.to_numpy().min()
    xs = piv.columns
    x_bounds = (xs[:-1] + xs[1:]) / 2
    x_bounds = np.concatenate([[2 * x_bounds[0] - x_bounds[1]], x_bounds, [2 * x_bounds[-1] - x_bounds[-2]]])
    ys = piv.index
    y_bounds = (ys[:-1] + ys[1:]) / 2
    print(ys)
    print(y_bounds)
    # need to rescale to make it a square plot
    y_bounds = np.concatenate([[2 * y_bounds[0] - y_bounds[1]], y_bounds, [2 * y_bounds[-1] - y_bounds[-2]]])
    x_bounds_diff = np.max(x_bounds) - np.min(x_bounds)
    y_bounds_diff = np.max(y_bounds) - np.min(y_bounds)
    y_bounds_mod = (y_bounds - np.min(y_bounds)) * x_bounds_diff / y_bounds_diff
    values = piv.values
    ax.pcolormesh(x_bounds, y_bounds_mod, values, cmap=cmap.get_cmap(), vmin=map_min, vmax=map_max)
    print(y_bounds_mod)
    y_ticks_shown = np.array([2,4,6,8,10])
    # retransforming the y ticks
    y_ticks_positions = (y_ticks_shown - np.min(y_bounds)) * x_bounds_diff / y_bounds_diff
    ax.set_yticks(y_ticks_positions)
    ax.set_yticklabels(y_ticks_shown, rotation=0)


def underover_plot(log=False):
    cache_path = os.path.join(default_cache_path, "combined_dataframe_cached_with_underoverconfident")
    combined_df = load_python_object(cache_path)
    combined_df = rename_columns(combined_df)
    combined_df.replace(2.5000999999999998, 2.5, inplace=True)
    combined_df = combined_df[combined_df["Timestep"] == 9]
    combined_df[
        "Difference of Interval-Center of Agent Measurement Noise Standard Deviation Parameter Distribution to True Standard Deviation"] = \
        combined_df["Interval-Center of Agent Measurement Noise Standard Deviation Parameter Distribution"] - \
        combined_df[
            "Standard Deviation of Measurements"]
    fig, axs = plt.subplots(1, 4, gridspec_kw={"width_ratios": [1.0, 1.0, 1.0, 0.1]})
    empty_ax = axs[0]
    empty_ax.get_xaxis().set_visible(False)
    empty_ax.get_yaxis().set_visible(False)
    empty_ax.set(adjustable='box', aspect='equal')
    axs = axs[1:]
    error_max1, error_min1 = get_limits(combined_df[combined_df["Configuration"] == "Bayes"],
                                        col1="Interval-Width of Agent Measurement Noise Standard Deviation Parameter Distribution",
                                        col2="Difference of Interval-Center of Agent Measurement Noise Standard Deviation Parameter Distribution to True Standard Deviation")
    error_max2, error_min2 = get_limits(combined_df[combined_df["Configuration"] == "BayesCI"],
                                        col1="Interval-Width of Agent Measurement Noise Standard Deviation Parameter Distribution",
                                        col2="Difference of Interval-Center of Agent Measurement Noise Standard Deviation Parameter Distribution to True Standard Deviation")
    if log:
        error_max = np.log(max([error_max1, error_max2, ]))
        error_min = np.log(min([error_min1, error_min2, ]))
    else:
        error_max = max([error_max1, error_max2,])
        error_min = 0.0


    CMap = Sunset_7.get_mpl_colormap()
    norm = mpl.colors.Normalize(vmin=error_min, vmax=error_max)
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=CMap)
    underover_heatmap_plot(combined_df, axs[0],
                             mapped_value="Accuracy Error",
                             x_col="Difference of Interval-Center of Agent Measurement Noise Standard Deviation Parameter Distribution to True Standard Deviation",
                             y_col="Interval-Width of Agent Measurement Noise Standard Deviation Parameter Distribution",
                             cmap=cmap,
                             map_min=error_min, map_max=error_max,
                             fixed_param_settings={"Configuration": "Bayes"})
    underover_heatmap_plot(combined_df, axs[1],
                             mapped_value="Accuracy Error",
                             x_col="Difference of Interval-Center of Agent Measurement Noise Standard Deviation Parameter Distribution to True Standard Deviation",
                             y_col="Interval-Width of Agent Measurement Noise Standard Deviation Parameter Distribution",
                             cmap=cmap,
                             map_min=error_min, map_max=error_max,
                             fixed_param_settings={"Configuration": "BayesCI"})
    mpl.colorbar.ColorbarBase(axs[2], cmap=cmap.get_cmap(), values=np.linspace(error_min, error_max, 1000),
                              orientation='vertical')
    axs[0].set_xlabel("Initial Certainty Offset\n$c_\mathrm{inf}^\mathrm{model} - c_\mathrm{inf}^\mathrm{true}$")
    axs[1].set_xlabel("Initial Certainty Offset\n$c_\mathrm{inf}^\mathrm{model} - c_\mathrm{inf}^\mathrm{true}$")
    axs[1].set_title("Bayesian Inference\nBI-UD")
    axs[0].set_title("Bayesian Inference\nBI-AI")
    axs[0].set_ylabel("Modeled Information Heterogeneity\n$p_\mathrm{inf}^\mathrm{model}$")
    axs[1].get_yaxis().set_visible(False)
    axs[2].set_ylabel("Final Accuracy Error $\\overline{E}^{A}_{t=T}$")
    axs[0].set(adjustable='box', aspect='equal')
    axs[1].set(adjustable='box', aspect='equal')
    fig.set_size_inches(9, 3.3)
    plt.tight_layout(w_pad=1.0)
    save_and_or_show_plot(results_plots_path,
                          "uncertainty_model.pdf", show_plots)


def correl_modelling_plot(log=False):
    cache_path = os.path.join(default_cache_path, "correl_combined_dataframe_cached_combined")
    combined_df = load_python_object(cache_path)
    combined_df = rename_columns(combined_df)
    combined_df = combined_df[combined_df["Timestep"] == 9]
    mapping = combined_df[["Netw_std_degree", "Centrality Homogeneity"]].groupby("Centrality Homogeneity").mean()
    combined_df["HetCent"] = [mapping.values[mapping.index == i][0][0] for i in combined_df["Centrality Homogeneity"]]
    fig, axs = plt.subplots(1, 4, gridspec_kw={"width_ratios": [1.0, 1.0, 1.0, 0.1]})
    empty_ax = axs[0]
    empty_ax.get_xaxis().set_visible(False)
    empty_ax.get_yaxis().set_visible(False)
    empty_ax.set(adjustable='box', aspect='equal')
    axs = axs[1:]
    error_max1, error_min1 = get_limits(combined_df[combined_df["Configuration"] == "Bayes"],
                                        col1="HetCent",
                                        col2="Correlation Coefficient")
    error_max2, error_min2 = get_limits(combined_df[combined_df["Configuration"] == "BayesCI"],
                                        col1="HetCent",
                                        col2="Correlation Coefficient")
    if log:
        error_max = np.log(max([error_max1, error_max2, ]))
        error_min = np.log(min([error_min1, error_min2, ]))
    else:
        error_max = max([error_max1, error_max2,])
        error_min = 0.0


    CMap = Sunset_7.get_mpl_colormap()
    norm = mpl.colors.Normalize(vmin=error_min, vmax=error_max)
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=CMap)
    correl_heatmap_plot(combined_df, axs[0],
                             mapped_value="Accuracy Error",
                             x_col="Correlation Coefficient",
                             y_col="HetCent",
                             cmap=cmap,
                             map_min=error_min, map_max=error_max,
                             fixed_param_settings={"Configuration": "Bayes"})
    correl_heatmap_plot(combined_df, axs[1],
                             mapped_value="Accuracy Error",
                             x_col="Correlation Coefficient",
                             y_col="HetCent",
                             cmap=cmap,
                             map_min=error_min, map_max=error_max,
                             fixed_param_settings={"Configuration": "BayesCI"})
    mpl.colorbar.ColorbarBase(axs[2], cmap=cmap.get_cmap(), values=np.linspace(error_min, error_max, 1000),
                              orientation='vertical')
    axs[0].set_xlabel("Centrality-Certainty Correlation\n$\\rho^\mathrm{model}$")
    axs[1].set_xlabel("Centrality-Certainty Correlation\n$\\rho^\mathrm{model}$")
    axs[1].set_title("Bayesian Inference\nBI-UD")
    axs[0].set_title("Bayesian Inference\nBI-AI")
    axs[0].set_ylabel("Centrality Heterogeneity\n$\sigma_\mathrm{net}$")
    axs[1].get_yaxis().set_visible(False)
    axs[2].set_ylabel("Final Accuracy Error $\\overline{E}^{A}_{t=T}$")
    axs[0].set(adjustable='box', aspect='equal')
    axs[1].set(adjustable='box', aspect='equal')
    fig.set_size_inches(9, 3.3)
    plt.tight_layout(w_pad=1.0)
    save_and_or_show_plot(results_plots_path,
                          "correl_uncertainty_model.pdf", show_plots)


def combined_modelling_plot(log=False):
    cache_path = os.path.join(default_cache_path, "combined_dataframe_cached_with_underoverconfident")
    Uncorrelated_combined_df = load_python_object(cache_path)
    Uncorrelated_combined_df = rename_columns(Uncorrelated_combined_df)
    Uncorrelated_combined_df.replace(2.5000999999999998, 2.5, inplace=True)
    Uncorrelated_combined_df = Uncorrelated_combined_df[Uncorrelated_combined_df["Timestep"] == 9]
    Uncorrelated_combined_df[
        "Difference of Interval-Center of Agent Measurement Noise Standard Deviation Parameter Distribution to True Standard Deviation"] = \
        Uncorrelated_combined_df["Interval-Center of Agent Measurement Noise Standard Deviation Parameter Distribution"] - \
        Uncorrelated_combined_df[
            "Standard Deviation of Measurements"]
    fig, all_axs = plt.subplots(2, 4, gridspec_kw={"width_ratios": [1.8, 1.0, 1.0, 0.1]})


    empty_ax = all_axs[0,0]
    empty_ax.axes.get_xaxis().set_ticks([])
    empty_ax.axes.get_yaxis().set_ticks([])
    img = mpl.image.imread(os.path.join(default_plots_path,"single_runs", "mismodeling_independent.png"))
    empty_ax.imshow(img)
    empty_ax.set_xlabel("Illustration of\nmodeled initial uncertainties")
    empty_ax.set_title("Uncorrelated errors", rotation='vertical', x=-.18, y=0.1)
    empty_ax.set(adjustable='box', aspect='equal')
    empty_ax = all_axs[1,0]
    img = mpl.image.imread(os.path.join(default_plots_path,"single_runs", "mismodeling_correl.png"))
    empty_ax.imshow(img)
    empty_ax.axes.get_xaxis().set_ticks([])
    empty_ax.axes.get_yaxis().set_ticks([])
    empty_ax.set_xlabel("Illustration of modeled correlation\nof uncertainty (color intensity)\nand centrality (size)")
    empty_ax.set_title("Centrality-correlated errors", rotation='vertical', x=-.18, y=-0.05)
    empty_ax.set(adjustable='box', aspect='equal')


    cache_path = os.path.join(default_cache_path, "correl_combined_dataframe_cached_combined")
    correlated_combined_df = load_python_object(cache_path)
    correlated_combined_df = rename_columns(correlated_combined_df)
    correlated_combined_df = correlated_combined_df[correlated_combined_df["Timestep"] == 9]
    print(correlated_combined_df["Netw_std_degree"].max())
    mapping = correlated_combined_df[["Netw_std_degree", "Centrality Homogeneity"]].groupby("Centrality Homogeneity").mean()
    print(mapping)
    correlated_combined_df["HetCent"] = [mapping.values[mapping.index == i][0][0] for i in correlated_combined_df["Centrality Homogeneity"]]
    print(correlated_combined_df["HetCent"].max())

    error_max1, error_min1 = get_limits(correlated_combined_df[correlated_combined_df["Configuration"] == "Bayes"],
                                        col1="HetCent",
                                        col2="Correlation Coefficient")
    error_max2, error_min2 = get_limits(correlated_combined_df[correlated_combined_df["Configuration"] == "BayesCI"],
                                        col1="HetCent",
                                        col2="Correlation Coefficient")

    error_max3, error_min3 = get_limits(Uncorrelated_combined_df[Uncorrelated_combined_df["Configuration"] == "Bayes"],
                                        col1="Interval-Width of Agent Measurement Noise Standard Deviation Parameter Distribution",
                                        col2="Difference of Interval-Center of Agent Measurement Noise Standard Deviation Parameter Distribution to True Standard Deviation")
    error_max4, error_min4 = get_limits(
        Uncorrelated_combined_df[Uncorrelated_combined_df["Configuration"] == "BayesCI"],
        col1="Interval-Width of Agent Measurement Noise Standard Deviation Parameter Distribution",
        col2="Difference of Interval-Center of Agent Measurement Noise Standard Deviation Parameter Distribution to True Standard Deviation")

    if log:
        error_max = np.log(max([error_max1, error_max2, error_max3, error_max4]))
        error_min = np.log(min([error_min1, error_min2, ]))
        # error_min = np.log(0.1)
    else:
        error_max = max([error_max1, error_max2, error_max3, error_max4])
        error_min = min([error_min1, error_min2, error_min3, error_min4])
        error_min = 0.0

    CMap = Sunset_7.get_mpl_colormap()
    norm = mpl.colors.Normalize(vmin=error_min, vmax=error_max)
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=CMap)

    axs = all_axs[0,1:]
    underover_heatmap_plot(Uncorrelated_combined_df, axs[0],
                           mapped_value="Accuracy Error",
                           x_col="Difference of Interval-Center of Agent Measurement Noise Standard Deviation Parameter Distribution to True Standard Deviation",
                           y_col="Interval-Width of Agent Measurement Noise Standard Deviation Parameter Distribution",
                           cmap=cmap,
                           map_min=error_min, map_max=error_max,
                           fixed_param_settings={"Configuration": "Bayes"})
    underover_heatmap_plot(Uncorrelated_combined_df, axs[1],
                           mapped_value="Accuracy Error",
                           x_col="Difference of Interval-Center of Agent Measurement Noise Standard Deviation Parameter Distribution to True Standard Deviation",
                           y_col="Interval-Width of Agent Measurement Noise Standard Deviation Parameter Distribution",
                           cmap=cmap,
                           map_min=error_min, map_max=error_max,
                           fixed_param_settings={"Configuration": "BayesCI"})
    axs[0].set_xlabel("Initial certainty offset\n$c_\mathrm{inf}^\mathrm{model} - c_\mathrm{inf}^\mathrm{true}$")
    axs[1].set_xlabel("Initial certainty offset\n$c_\mathrm{inf}^\mathrm{model} - c_\mathrm{inf}^\mathrm{true}$")
    axs[1].set_title("Bayesian Inference\nBI-UD")
    axs[0].set_title("Bayesian Inference\nBI-AI")
    axs[0].set_ylabel("Modeled information heterogeneity\n$p_\mathrm{inf}^\mathrm{model}$")
    axs[1].get_yaxis().set_visible(False)
    # axs[2].set_ylabel("Final Accuracy Error $\\overline{E}^{A}_{t=T}$")
    # axs[2].yaxis.set_ticks_position('left')
    # axs[2].yaxis.set_label_position("left")
    axs[0].set(adjustable='box', aspect='equal')
    axs[1].set(adjustable='box', aspect='equal')

    axs = all_axs[1, 1:]
    correl_heatmap_plot(correlated_combined_df, axs[0],
                        mapped_value="Accuracy Error",
                        x_col="Correlation Coefficient",
                        y_col="HetCent",
                        cmap=cmap,
                        map_min=error_min, map_max=error_max,
                        fixed_param_settings={"Configuration": "Bayes"})
    correl_heatmap_plot(correlated_combined_df, axs[1],
                        mapped_value="Accuracy Error",
                        x_col="Correlation Coefficient",
                        y_col="HetCent",
                        cmap=cmap,
                        map_min=error_min, map_max=error_max,
                        fixed_param_settings={"Configuration": "BayesCI"})



    axs[0].set_xlabel("Centrality-certainty correlation\n$\\rho^\mathrm{model}$")
    axs[1].set_xlabel("Centrality-certainty correlation\n$\\rho^\mathrm{model}$")
    # axs[1].set_title("Bayesian Inference\nBI-UD")
    # axs[0].set_title("Bayesian Inference\nBI-AI")
    axs[0].set_ylabel("Centrality heterogeneity\n$\sigma_\mathrm{net}$")
    axs[1].get_yaxis().set_visible(False)
    # axs[2].yaxis.set_ticks_position('left')
    # axs[2].yaxis.set_label_position("left")
    axs[0].set(adjustable='box', aspect='equal')
    axs[1].set(adjustable='box', aspect='equal')

    gs = all_axs[0, -1].get_gridspec()
    # remove the underlying axes
    for ax in all_axs[:, -1]:
        ax.remove()
    axbig = fig.add_subplot(gs[:, -1])
    mpl.colorbar.ColorbarBase(axbig, cmap=cmap.get_cmap(), values=np.linspace(error_min, error_max, 1000),
                              orientation='vertical')
    axbig.set_ylabel("Final accuracy error $\\overline{E}^{A}_{t=T}$")

    fig.set_size_inches(9.75, 6)
    plt.tight_layout(w_pad=1.0)
    save_and_or_show_plot(results_plots_path,
                          "combined_model.pdf", show_plots)

def error_error_plot(df):
    df = df[df["Timestep"] == 10]
    df = df[df["Configuration"] == "Naive"]

    fig, ax = plt.subplots(1, 1)
    tmp = df.groupby("Weight on own Belief (Naive)").mean()
    sns.scatterplot(ax=ax, data=tmp, x="Trueness Error", y="Precision Error", hue="Weight on own Belief (Naive)")
    ax.set_yscale("log")
    plt.show()

if __name__ == "__main__":
    default_result_path = "/home/vito/ws_domip/src/collective-decison-making-with-direl/results/hpc_mount"
    default_plots_path = "/home/vito/ws_domip/src/collective-decison-making-with-direl/plots"
    default_cache_path = "/media/vito/TOSHIBA EXT/cached_dfs"
    show_plots = False
    results_plots_path = os.path.join(default_plots_path, "paper_plots")
    os.makedirs(results_plots_path, exist_ok=True)
    cache_path = os.path.join(default_cache_path, "new_combined_dataframe_cached_combined")
    combined_df = load_python_object(cache_path)
    combined_df = rename_columns(combined_df)
    only_optimal_weights_df = filter_for_optimal_weights_in_naive(combined_df, ["Centrality Homogeneity",
                                                                                "Interval-Width of Agent Measurement Noise Standard Deviation Parameter Distribution",])

    combined_df = combined_df[combined_df["Timestep"] > 0]
    only_optimal_weights_df = only_optimal_weights_df[only_optimal_weights_df["Timestep"] > 0]
    local_df = pandas.concat(
        [combined_df, only_optimal_weights_df[only_optimal_weights_df["Configuration"] == "Naive (Optimal Weight)"]],
        ignore_index=True)
    het_het_precision_plot(local_df, results_plots_path)
    bar_plot(local_df)
    underover_plot()
    correl_modelling_plot()
    combined_modelling_plot()