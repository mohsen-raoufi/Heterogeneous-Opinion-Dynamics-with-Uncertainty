""""
This script is used to post process the raw data and create a pandas dataframe from it.
If you already downloaded the processed data, you can skip this step.
detail: it also saves the eigenvalues of the network at each time-step.
"""

import os
import numpy as np
import sys
import pickle
import dcargs
import pandas as pd

from experiments.experiment import ExperimentParameters
import glob
from tqdm import tqdm

path = "../../results/"
sys.path.append(path)

from multiprocessing import Pool
import networkx as nx

def parse_experiment_data_par(file_path):
    try:

        pickleFileName = file_path+"/data.pickle5"
        yamlFileName = file_path+"/config.yaml"

        with open(pickleFileName, "rb") as f:
            resFile = pickle.load(f)

        with open(yamlFileName, 'r') as f:
            configData = dcargs.from_yaml(ExperimentParameters, f)

        n_agents = configData.network_params.n_agents
        correlation_network_information = configData.network_params.correlation_network_information

        steps = configData.steps + 1 
        z_gt = configData.true_value
        

        # not checked yet
        range_agent_measurement_noise = configData.network_params.max_agent_measurement_noise - configData.network_params.min_agent_measurement_noise
        mean_agent_measurement_noise = 0.5*(configData.network_params.max_agent_measurement_noise + configData.network_params.min_agent_measurement_noise)
        env_noise_std = configData.env_noise_std
        scalar_param1 = configData.network_params.scalar_param1
        weight_own_belief = configData.network_params.weight_own_belief

        colData = resFile['collective_data']
        adjc = colData['connectivity']
        agntData = colData['agent_data']

        zArr = np.zeros((n_agents,steps))
        pArr = zArr.copy()

        for i in range(n_agents):
            z = agntData[i].belief_mean
            std = agntData[i].belief_std
            zArr[i,:] = z
            pArr[i,:] = std

        avgZArr = np.mean(zArr, axis=0)
        E_t = (avgZArr-z_gt)**2
        E_p2 = (zArr - avgZArr)**2
        E_p = np.mean(E_p2, axis=0)
        E_a = E_t + E_p



        G = nx.from_numpy_array(adjc[0].cpu().detach().numpy())
        degreeDist = np.array(G.degree())[:, 1]
        clustCoef = np.array(list(nx.clustering(G).values()))
        eigVec = np.array(list(nx.eigenvector_centrality_numpy(G).values()))

        GG = nx.adjacency_matrix(G).toarray()
        row_sums = GG.sum(axis=1)
        row_sums[row_sums==0] = 1
        row_sums[np.isinf(row_sums)] = 1
        G2 = GG / row_sums[:, np.newaxis]
        eigVal = np.linalg.eigvals(G2)
        eigVal[np.isinf(eigVal)] = np.iinfo(np.int16).max

        eigvals_time = []
        for t in range(steps):
            G = nx.from_numpy_array(adjc[t].cpu().detach().numpy())
            GG = nx.adjacency_matrix(G).toarray()
            row_sums = GG.sum(axis=1)
            row_sums[row_sums==0] = 1
            row_sums[np.isinf(row_sums)] = 1
            G2 = GG / row_sums[:, np.newaxis]
            eigVal = np.linalg.eigvals(G2)
            eigVal[np.isinf(eigVal)] = np.iinfo(np.int16).max
            eigvals_time.append(eigVal)


        d = {"Trueness_Error": E_t, "Precision_Error": E_p, "Accuracy_Error": E_a, "Collective_Mean": avgZArr,
            "Number_of_Agents": [n_agents for _ in range(steps)], "True_Value": [z_gt for _ in range(steps)],
            "Timestep": [i for i in range(steps)],
            "mean_agent_measurement_noise": [mean_agent_measurement_noise for _ in range(steps)],
            "range_agent_measurement_noise": [range_agent_measurement_noise for _ in range(steps)],
            "std_environment_noise": [env_noise_std for _ in range(steps)],
            "scalar_param1": [scalar_param1 for _ in range(steps)],
            "weight_own_belief": [weight_own_belief for _ in range(steps)],
             # ToDo: Calculate Network properties for each time-step, so that we can see the evolution of these properties over time
            "Netw_num_of_Edges" : G.number_of_edges(),
            "Netw_std_degree" : np.std(degreeDist),
            "Netw_mean_degree" : np.mean(degreeDist),
            "Netw_std_eigVec" : np.std(eigVec),
            "Netw_mean_eigVec" : np.mean(eigVec),
            "Netw_max_eigVec" : np.max(eigVec),
            "Netw_std_CC" : np.std(clustCoef),
            "Netw_mean_CC" : np.mean(clustCoef),
            "Netw_max_CC" : np.max(clustCoef),
            "Netw_std_eigVal" : np.std(eigVal),
            "Netw_mean_eigVal" : np.mean(eigVal),
            "Netw_max_eigVal" : np.max(eigVal),#[-1],
            "Netw_eigVals_vs_time" : eigvals_time, # }
            "correlation_network_information": correlation_network_information}

        return pd.DataFrame.from_dict(d)
    except Exception as e:
    # else:
        print(e, flush=True)
        print("Something Went Wrong!",flush=True)


def parse_experiments(base_file_path):
    pathes = glob.glob(os.path.join(base_file_path, "*"))
    pool = Pool(20)
    dataframes = pool.imap(parse_experiment_data_par, tqdm(pathes))
    return pd.concat(dataframes, ignore_index=True)



def parse_experiments_check(base_file_path):
    pathes = glob.glob(os.path.join(base_file_path, "*"))
    dataframes = parse_experiment_data_par(pathes[0])




for experiment_name in experiment_name_list:
    print(experiment_name)
    experiment_folder_path = path + "/" + experiment_name
    dataFrame_path = experiment_folder_path[:-8] + "/processed_results.pickle"
    print("DataFrame Path: ", dataFrame_path)
    df = parse_experiments(experiment_folder_path)
    df.to_pickle(dataFrame_path)
