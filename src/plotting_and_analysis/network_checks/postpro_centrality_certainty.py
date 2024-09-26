import pickle
import dcargs
import numpy as np
import pandas as pd
import networkx as nx
<<<<<<< HEAD
import matplotlib as mpl
=======
>>>>>>> e0aa1dbd6bfefdcc00da2d900e960dd883428476
import matplotlib.pyplot as plt

from experiments.experiment import ExperimentParameters


def post_pro_network_centrality_certainty(file_path):
    # print("File Path is: " + file_path)
    # try:
    if(True):
        # print("Dummy Check!", flush=True)

        # remove "data.pickle5" from file_path
        file_path = file_path.replace("data.pickle5", "")
        file_path = file_path.replace("config.yaml", "")

        pickleFileName = file_path+"/data.pickle5"
        yamlFileName = file_path+"/config.yaml"

        with open(pickleFileName, "rb") as f:
            resFile = pickle.load(f)

        with open(yamlFileName, 'r') as f:
            configData = dcargs.from_yaml(ExperimentParameters, f)

        n_agents = configData.network_params.n_agents
        steps = configData.steps + 1 
        z_gt = configData.true_value

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

<<<<<<< HEAD
        # just in this postprocessing, for the sake of visualization of opinions!
        zArr = zArr - z_gt

=======
>>>>>>> e0aa1dbd6bfefdcc00da2d900e960dd883428476
        G = nx.from_numpy_array(adjc[0].cpu().detach().numpy())
        degreeDist = np.array(G.degree())[:, 1]
        clustCoef = np.array(list(nx.clustering(G).values()))
        

        # calculate eigen value
        # eigVal = get_eigvals_from_nx(G)

        # calculate the eigenvector centrality
        G_normed_np = get_row_normalized_adj_mat_from_nx(G)
        G_normed = nx.from_numpy_array(G_normed_np)
<<<<<<< HEAD
        centrality_0 = np.array(list(nx.eigenvector_centrality(G_normed, max_iter=10000).values()))
=======
        centrality_0 = np.array(list(nx.eigenvector_centrality(G_normed).values()))
>>>>>>> e0aa1dbd6bfefdcc00da2d900e960dd883428476

        init_central_node = np.argmax(centrality_0)
        ind_central = init_central_node # initial values
        ind_peripheral = np.argmin(centrality_0) # initial values
        init_peripheral_node = ind_peripheral

        uncertainty_time = []
        uncertainty_central_node = []
        uncertainty_peripheral_node = []
        uncertainty_avg_nodes = []       
        uncertainty_init_central_node = []
        uncertainty_init_peripheral_node = []

        centrality_central_node = []
        centrality_peripheral_node = [] 
        centrality_most_cert_node = []
        centrality_least_cert_node = []
        centrality_init_central_node = []
        centrality_init_most_cert_node = []
        centrality_init_least_cert_node = []

        ind_most_cert_node_arr = []
        ind_least_cert_node_arr = []
        ind_central_node_arr = []
        ind_peripheral_node_arr = []

        uncertainty_init = pArr[:,1].T
        ind_init_most_cert_node = np.argmin(uncertainty_init)
        ind_init_least_cert_node = np.argmax(uncertainty_init)

        # dist_to_center_all_nodes = []
        # dist_to_center_central_node = []
        # dist_to_center_peripheral_node = []

        z_arr_time = []        
        centrality_time = []
        adjc_time = []
        

        

        # print("Start Testing Adjc Over Time! for steps: ", steps, flush=True)
        
        for t in range(steps):
            # print(t)
            adjc_t = adjc[t].cpu().detach().numpy()
            G = nx.from_numpy_array(adjc_t)
            degreeDist = np.array(G.degree())[:, 1]

            # calculate eigen value
            # eigVal = get_eigvals_from_nx(G)

            # calculate the eigenvector centrality
            G_normed_np = get_row_normalized_adj_mat_from_nx(G)
<<<<<<< HEAD
            # G_normed_np = nx.adjacency_matrix(G).toarray()
=======
>>>>>>> e0aa1dbd6bfefdcc00da2d900e960dd883428476
            G_normed = nx.from_numpy_array(G_normed_np)

            centrality = np.sum(adjc_t, axis=0)

<<<<<<< HEAD
            # centrality = np.array(list(nx.eigenvector_centrality(G_normed, weight='weight', max_iter=100000).values()))
            # print(centrality, flush=True)
            centrality = centrality / np.max(centrality)
=======
            # centrality = np.array(list(nx.eigenvector_centrality(G_normed, weight='weight', max_iter=10000).values()))
            # centrality = centrality / np.max(centrality)
>>>>>>> e0aa1dbd6bfefdcc00da2d900e960dd883428476

            ind_central = np.argmax(centrality)
            ind_peripheral = np.argmin(centrality)
            centrality_central_node.append(centrality[ind_central])
            centrality_peripheral_node.append(centrality[ind_peripheral])

            centrality_time.append(centrality)
            adjc_time.append(adjc_t)

            
            uncertainty_all_tmp = pArr[:,t].T
            uncertainty_time.append(uncertainty_all_tmp)
            uncertainty_central_node.append(uncertainty_all_tmp[ind_central])
            uncertainty_peripheral_node.append(uncertainty_all_tmp[ind_peripheral])
            uncertainty_avg_nodes.append(np.mean(uncertainty_all_tmp))

            ind_most_cert_node = np.argmin(uncertainty_all_tmp)
            ind_least_cert_node = np.argmax(uncertainty_all_tmp)
            centrality_most_cert_node.append(centrality[ind_most_cert_node])
            centrality_least_cert_node.append(centrality[ind_least_cert_node])
            centrality_init_most_cert_node.append(centrality[ind_init_most_cert_node])
            centrality_init_least_cert_node.append(centrality[ind_init_least_cert_node])

            uncertainty_init_central_node.append(uncertainty_all_tmp[init_central_node])
            uncertainty_init_peripheral_node.append(uncertainty_all_tmp[init_peripheral_node])
            centrality_init_central_node.append(centrality[init_central_node])

            ind_most_cert_node_arr.append(ind_most_cert_node)
            ind_least_cert_node_arr.append(ind_least_cert_node)
            ind_central_node_arr.append(ind_central)
            ind_peripheral_node_arr.append(ind_peripheral)

            # mean_zArr = np.mean(zArr[:,i])

            # dist_to_center_all_nodes_tmp = (zArr[:,i].T - mean_zArr)**2;
            # dist_to_center_central_node.append(dist_to_center_all_nodes_tmp[ind_central])
            # dist_to_center_peripheral_node.append(dist_to_center_all_nodes_tmp[ind_peripheral])
            # dist_to_center_all_nodes.append(dist_to_center_all_nodes_tmp)

            z_arr_time.append(zArr[:,t])
            

        # for the movement comparison: not final yet
        diff_z_tf = np.abs(z_arr_time[steps - 1][:] - z_arr_time[0][:])
        diff_z_tf_central_node = diff_z_tf[ind_central]


        # for the movement vector dot product
        moved_vector_all = z_arr_time[steps - 1] - z_arr_time[0]
        center_moved_vec = np.mean(z_arr_time[steps - 1]) - np.mean(z_arr_time[0])

        dot_prod_all = np.dot(moved_vector_all, center_moved_vec)
        sign_dot_prod_all = np.sign(dot_prod_all)
        sign_dot_prod_central = sign_dot_prod_all[ind_central]


        d = {"adjc_time" : adjc_time,
            "Timestep": [i for i in range(steps)],
<<<<<<< HEAD
            "True_Value": [z_gt for _ in range(steps)],
=======
>>>>>>> e0aa1dbd6bfefdcc00da2d900e960dd883428476
            "z_arr_time" : z_arr_time,
            "scalar_param1": [scalar_param1 for _ in range(steps)],
            "weight_own_belief": [weight_own_belief for _ in range(steps)],
            #
            "uncertainty_time" : uncertainty_time,
            "uncertainty_central_node" : uncertainty_central_node,
            "uncertainty_peripheral_node" : uncertainty_peripheral_node,
            "uncertainty_avg_nodes" : uncertainty_avg_nodes,
            "uncertainty_init_central_node" : uncertainty_init_central_node,
            "uncertainty_init_peripheral_node" : uncertainty_init_peripheral_node,
            #
            "Netw_num_of_Edges" : G.number_of_edges(),
            "Netw_std_degree" : np.std(degreeDist),
            "Netw_mean_degree" : np.mean(degreeDist),
            "Netw_std_centrality" : np.std(centrality_0),
            "Netw_mean_centrality" : np.mean(centrality_0),
            "Netw_max_centrality" : np.max(centrality_0),
            "Netw_std_CC" : np.std(clustCoef),
            "Netw_mean_CC" : np.mean(clustCoef),
            "Netw_max_CC" : np.max(clustCoef),
            #
            "diff_z_tf_avg" : np.mean(diff_z_tf),
            "diff_z_tf_central_node" : diff_z_tf_central_node,
            "sign_dot_prod_avg" : np.mean(sign_dot_prod_all),
            "sign_dot_prod_central" : sign_dot_prod_central,#
            #
            "centrality_time" : centrality_time, 
            "centrality_central_node" : centrality_central_node,
            "centrality_peripheral_node" : centrality_peripheral_node,
            "centrality_most_cert_node" : centrality_most_cert_node,
            "centrality_least_cert_node" : centrality_least_cert_node,
            "centrality_init_central_node" : centrality_init_central_node,
            "centrality_init_most_cert_node" : centrality_init_most_cert_node,
            "centrality_init_least_cert_node" : centrality_init_least_cert_node,
            #
            "ind_most_cert_node_arr" : ind_most_cert_node_arr,
            "ind_least_cert_node_arr" : ind_least_cert_node_arr,
            "ind_central_node_arr" : ind_central_node_arr,
            "ind_peripheral_node_arr" : ind_peripheral_node_arr,
            }
        return pd.DataFrame.from_dict(d)
    # except Exception as e:
    # # else:
    #     print(e, flush=True)
    #     print("Something Went Wrong!",flush=True)


def get_eigvals_from_nx(G):
    G2 = get_row_normalized_adj_mat_from_nx(G)
    eigVal = np.linalg.eigvals(G2)
    eigVal[np.isinf(eigVal)] = np.iinfo(np.int16).max
    eigVal = np.abs(eigVal)
    return eigVal


def get_row_normalized_adj_mat_from_nx(G):
    G_np = nx.adjacency_matrix(G).toarray()
    row_sums = G_np.sum(axis=1)
    row_sums[row_sums==0] = 1
    row_sums[np.isinf(row_sums)] = 1
    G2 = G_np / row_sums[:, np.newaxis]
    return G2


<<<<<<< HEAD
def draw_network_color_nodes_on_degree(G, ax=None, edge_alpha = 0.5, edge_width = 1.5, node_size_scale = 1.0):
    # # pos = nx.spring_layout(G, seed=100)  # Seed layout for reproducibility
    # pos = nx.spring_layout(G, k=0.01, iterations=20)

    # # change the layout based on the node centrality + spring layout so that they don't stick together
    # # pos = nx.spectral_layout(G)
    # # pos = nx.spring_layout(G, k=0.01, iterations=2, pos=pos)
    # # 
    # # pos = nx.spiral_layout(G)

    # # pos = nx.shell_layout(G)
    # # pos = nx.random_layout(G)
    # # pos = nx.kamada_kawai_layout(G)
    # # pos = nx.circular_layout(G)
    # # pos = nx.fruchterman_reingold_layout(G)
    # # pos = nx.planar_layout(G)
    # # pos = nx.rescale_layout(G)
    # # pos = nx.bipartite_layout(G)
    # # pos = nx.multipartite_layout(G)
    
    # node_color = []
    # degree_dist = np.array(G.degree())[:,1]
    # for i_node in G.nodes():
    #     node_color.append((degree_dist[i_node]/degree_dist.max(),0,0))
    # options = {
    #     "node_color": node_color, #"#A0CBE2",
    #     "edge_color": "#6BCBFF",
    #     "width": 1,
    #     "edge_cmap": plt.cm.Blues,
    #     "with_labels": False,
    #     "node_size": 1000/G.number_of_nodes(),
    # }
    # if(ax is None):
    #     nx.draw(G, pos, **options)
    # else:
    #     nx.draw(G, pos, ax=ax, **options)

    # pos = nx.circular_layout(G)
    # pos = nx.spring_layout(G, seed=63)  # Seed layout for reproducibility
    # change the layout based on the node centrality
    # pos = nx.spring_layout(G, k=1.0, iterations=100)

    pos = nx.random_layout(G)
    # pos = nx.spring_layout(G, k=100.1, iterations=1000,fixed=[0],pos=pos) # for the star network

    pos = nx.spiral_layout(G)
    # pos = nx.spring_layout(G, k=7.5, iterations=200, pos=pos, center=pos[0])
# 
    pos = nx.spring_layout(G, k=.5, iterations=200, pos=pos, center=pos[0])

    # pos = nx.spectral_layout(G)

    # pos = nx.graphviz_layout(G, prog="neato")
    # pos = nx.pydot_layout(G, prog="neato")

    # use different networkx layouts
    # 
    # pos = nx.spectral_layout(G)
    # pos = nx.kamada_kawai_layout(G)
    # pos = nx.fruchterman_reingold_layout(G)
    # pos = nx.spiral_layout(G)

    # import hot colormap from matplotlib
    cmap = mpl.cm.hot

    node_color = []
    degree_dist = np.array(G.degree())[:,1]

    max_deg = degree_dist.max()
    min_deg = degree_dist.min()
    diff_deg = max_deg - min_deg

    for i_node in G.nodes():
        # define color for each node based on its degree
        tmp = (degree_dist[i_node]-min_deg)/(diff_deg)
        # node_color.append(cmap(tmp))
        node_color.append((tmp**.5,0,0))
        
    options_nodes = {
        "node_color": node_color, 
        "node_size": node_size_scale*3000/G.number_of_nodes(),
    }
    options_edges = {
        "edge_color": "#33cccc", # "#6BCBFF",
        "width": edge_width,
        "edge_cmap": plt.cm.Blues,
        #"with_labels": False,
    }
    if(ax is None):
        nx.draw_networkx_nodes(G,pos=pos,**options_nodes)#draw nodes
        nx.draw_networkx_edges(G,pos=pos, alpha=edge_alpha, **options_edges) #  for key,value in cent.items()] #loop through edges and draw them

        # nx.draw(G, pos, **options)
    else:
        nx.draw_networkx_nodes(G,pos=pos,**options_nodes,ax=ax)#draw nodes
        nx.draw_networkx_edges(G,pos=pos, alpha=edge_alpha, **options_edges, ax=ax)
        ax.axis('off')
        # nx.draw(G, pos, ax=ax, **options)

    ax.set_facecolor('white')
=======
def draw_network_color_nodes_on_degree(G, ax=None):
    # pos = nx.spring_layout(G, seed=100)  # Seed layout for reproducibility
    pos = nx.spring_layout(G, k=0.01, iterations=20)

    # change the layout based on the node centrality + spring layout so that they don't stick together
    # pos = nx.spectral_layout(G)
    # pos = nx.spring_layout(G, k=0.01, iterations=2, pos=pos)
    # 
    # pos = nx.spiral_layout(G)

    # pos = nx.shell_layout(G)
    # pos = nx.random_layout(G)
    # pos = nx.kamada_kawai_layout(G)
    # pos = nx.circular_layout(G)
    # pos = nx.fruchterman_reingold_layout(G)
    # pos = nx.planar_layout(G)
    # pos = nx.rescale_layout(G)
    # pos = nx.bipartite_layout(G)
    # pos = nx.multipartite_layout(G)
    
    node_color = []
    degree_dist = np.array(G.degree())[:,1]
    for i_node in G.nodes():
        node_color.append((degree_dist[i_node]/degree_dist.max(),0,0))
    options = {
        "node_color": node_color, #"#A0CBE2",
        "edge_color": "#6BCBFF",
        "width": 1,
        "edge_cmap": plt.cm.Blues,
        "with_labels": False,
        "node_size": 1000/G.number_of_nodes(),
    }
    if(ax is None):
        nx.draw(G, pos, **options)
    else:
        nx.draw(G, pos, ax=ax, **options)
>>>>>>> e0aa1dbd6bfefdcc00da2d900e960dd883428476



####### THESE ARE JUST FOR BACKUP! USE THEM CAREFULLY #######

def parse_experiment_data_par(file_path):
    # print("File Path is: " + file_path)
    try:
    # if(True):
        # E_a, E_p, E_t, G, avgZArr, clustCoef, degreeDist, eigVal, eigVec, env_noise_std, mean_agent_measurement_noise, n_agents, range_agent_measurement_noise, scalar_param1, steps, weight_own_belief, z_gt, _, _ = extract_relevant_data_from_files(
            # file_path)

        E_a, E_p, E_t, adjc, G, avgZArr, clustCoef, degreeDist, eigVal, eigVec, env_noise_std, mean_agent_measurement_noise, n_agents, range_agent_measurement_noise, scalar_param1, steps, weight_own_belief, z_gt, zArr, pArr = extract_relevant_data_from_files(
            file_path)

    # print("eigVals list lenght: ", len(eigvals_time), flush=True)

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
            "Netw_max_eigVal" : np.max(eigVal), #}##,#[-1],
            "Netw_2ndmax_eigVal" : eigVal[-2]}#,
            # "Netw_Adjc" : adjc_time,
            # "Netw_eigVals_vs_time" : eigvals_time}
            # "Netw_eigVals_vs_time" : [get_eigVals_of_adj_mat(nx.from_numpy_array(adjc[i].cpu().detach().numpy())) for i in range(steps)]}

        return pd.DataFrame.from_dict(d)
    except Exception as e:
    # else:
        print(e, flush=True)
        print("Something Went Wrong!",flush=True)


def parse_experiment_data_par_bayes(file_path):
    # print("File Path is: " + file_path)
    try:
    # if(True):
        # print("Dummy Check!", flush=True)

        # remove "data.pickle5" from file_path
        file_path = file_path.replace("data.pickle5", "")
        file_path = file_path.replace("config.yaml", "")

        pickleFileName = file_path+"/data.pickle5"
        yamlFileName = file_path+"/config.yaml"

        with open(pickleFileName, "rb") as f:
            resFile = pickle.load(f)

        with open(yamlFileName, 'r') as f:
            configData = dcargs.from_yaml(ExperimentParameters, f)


        n_agents = configData.network_params.n_agents
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

        # print("nAgents: " + str(n_agents) + ", steps: " + str(steps));
        # print("agnt data: ", len(agntData))

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
        # print(G, flush=True)
        degreeDist = np.array(G.degree())[:, 1]
        clustCoef = np.array(list(nx.clustering(G).values()))
        eigVec = np.array(list(nx.eigenvector_centrality_numpy(G).values()))

        ind_central = np.argmax(eigVec)
        ind_peripheral = np.argmin(eigVec)

        confidence_arr = []
        confident_central_node = []
        confident_peripheral_node = []
        confident_avg_nodes = []        
        E_T_central_node = []
        E_T_peripheral_node = []
        E_T_all_nodes = []    

        dist_to_center_all_nodes = []
        dist_to_center_central_node = []
        dist_to_center_peripheral_node = []

        z_arr_time = []

        for i in range(steps):
            confidence_all_tmp = pArr[:,i].T
            confidence_arr.append(confidence_all_tmp)
            confident_central_node.append(confidence_all_tmp[ind_central])
            confident_peripheral_node.append(confidence_all_tmp[ind_peripheral])
            confident_avg_nodes.append(np.mean(confidence_all_tmp))

            E_T_all_tmp = (zArr[:,i].T - z_gt)**2
            E_T_all_nodes.append(E_T_all_tmp)
            E_T_central_node.append(E_T_all_tmp[ind_central])
            E_T_peripheral_node.append(E_T_all_tmp[ind_peripheral])

            mean_zArr = np.mean(zArr[:,i])

            dist_to_center_all_nodes_tmp = (zArr[:,i].T - mean_zArr)**2;
            dist_to_center_central_node.append(dist_to_center_all_nodes_tmp[ind_central])
            dist_to_center_peripheral_node.append(dist_to_center_all_nodes_tmp[ind_peripheral])
            dist_to_center_all_nodes.append(dist_to_center_all_nodes_tmp)

            z_arr_time.append(zArr[:,i])



        # print("Dummy Check3 !", flush=True)

        GG = nx.adjacency_matrix(G).toarray()
        row_sums = GG.sum(axis=1)
        row_sums[row_sums==0] = 1
        row_sums[np.isinf(row_sums)] = 1
        G2 = GG / row_sums[:, np.newaxis]
        eigVal = np.linalg.eigvals(G2)
        eigVal[np.isinf(eigVal)] = np.iinfo(np.int16).max
        # eigVal[np.isinf(eigVal) + np.isnan(eigVal)] = -1.01
        # eigVal = np.sort(np.abs(eigVal))

        # print("Start Testing Adjc Over Time! for steps: ", steps, flush=True)
        eigvals_time = []
        adjc_time = []
        for t in range(steps):
            # print(t)
            G = nx.from_numpy_array(adjc[t].cpu().detach().numpy())
            GG = nx.adjacency_matrix(G).toarray()
            row_sums = GG.sum(axis=1)
            row_sums[row_sums==0] = 1
            row_sums[np.isinf(row_sums)] = 1
            G2 = GG / row_sums[:, np.newaxis]
            eigVal = np.linalg.eigvals(G2)
            # eigVal[np.isinf(eigVal) + np.isnan(eigVal)] = -1.01
            eigVal[np.isinf(eigVal)] = np.iinfo(np.int16).max
            # L = nx.normalized_laplacian_matrix(G)
            # eigVal = np.array(list(nx.eigenvector_centrality(G).values()))
            # print("eigVals lenght: ", eigVal.shape, flush=True)
            eigvals_time.append(eigVal)
            adjc_time.append(adjc[t].cpu().detach().numpy())

        # print("eigVals list lenght: ", len(eigvals_time), flush=True)

        d = {"Trueness_Error": E_t, "Precision_Error": E_p, "Accuracy_Error": E_a, "Collective_Mean": avgZArr,
             "confidence_arr" : confidence_arr,
             "z_arr" : z_arr_time,
             "E_T_all_nodes" : E_T_all_nodes,
             "dist_to_center_all_nodes" : dist_to_center_all_nodes,
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
            # "Netw_2ndmax_eigVal" : eigVal[-2],
            # "Netw_Adjc" : adjc_time,
            "Netw_eigVals_vs_time" : eigvals_time,
            "adjc_time" : adjc_time}
            # "Netw_eigVals_vs_time" : [get_eigVals_of_adj_mat(nx.from_numpy_array(adjc[i].cpu().detach().numpy())) for i in range(steps)]}

        return pd.DataFrame.from_dict(d)
    except Exception as e:
    # else:
        print(e, flush=True)
        print("Something Went Wrong!",flush=True)

def extract_relevant_data_from_files(file_path):
    pickleFileName = file_path + "/data.pickle5"
    yamlFileName = file_path + "/config.yaml"
    with open(pickleFileName, "rb") as f:
        resFile = pickle.load(f)
    with open(yamlFileName, 'r') as f:
        configData = dcargs.from_yaml(ExperimentParameters, f)
    n_agents = configData.network_params.n_agents

    steps = configData.steps + 1
    z_gt = configData.true_value
    # not checked yet
    range_agent_measurement_noise = configData.network_params.max_agent_measurement_noise - configData.network_params.min_agent_measurement_noise
    mean_agent_measurement_noise = 0.5 * (
                configData.network_params.max_agent_measurement_noise + configData.network_params.min_agent_measurement_noise)
    env_noise_std = configData.env_noise_std
    scalar_param1 = configData.network_params.scalar_param1
    weight_own_belief = configData.network_params.weight_own_belief
    colData = resFile['collective_data']
    adjc = colData['connectivity']
    agntData = colData['agent_data']
    # print("nAgents: " + str(n_agents) + ", steps: " + str(steps));
    # print("agnt data: ", len(agntData))
    zArr = np.zeros((n_agents, steps))
    pArr = zArr.copy()
    for i in range(n_agents):
        z = agntData[i].belief_mean
        std = agntData[i].belief_std
        zArr[i, :] = z
        pArr[i, :] = std
    avgZArr = np.mean(zArr, axis=0)
    E_t = (avgZArr - z_gt) ** 2
    E_p2 = (zArr - avgZArr) ** 2
    E_p = np.mean(E_p2, axis=0)
    E_a = E_t + E_p
    G = nx.from_numpy_array(adjc[0].cpu().detach().numpy())
    degreeDist = np.array(G.degree())[:, 1]
    clustCoef = np.array(list(nx.clustering(G).values()))
    eigVec = np.array(list(nx.eigenvector_centrality_numpy(G).values()))
    GG = nx.adjacency_matrix(G).toarray()
    row_sums = GG.sum(axis=1)
    row_sums[row_sums == 0] = 1
    row_sums[np.isinf(row_sums)] = 1
    G2 = GG / row_sums[:, np.newaxis]
    eigVal = np.linalg.eigvals(G2)
    eigVal[np.isinf(eigVal)] = np.iinfo(np.int16).max
    # eigVal[np.isinf(eigVal) + np.isnan(eigVal)] = -1.01
    # eigVal = np.sort(np.abs(eigVal))
    # # print("Start Testing Adjc Over Time! for steps: ", steps)
    # eigvals_time = []
    # # adjc_time = []
    # for t in range(steps):
    #     # print(t)
    #     G = nx.from_numpy_array(adjc[t].cpu().detach().numpy())
    #     GG = nx.adjacency_matrix(G).toarray()
    #     row_sums = GG.sum(axis=1)
    #     row_sums[row_sums==0] = 1
    #     G2 = GG / row_sums[:, np.newaxis]
    #     eigVal = np.linalg.eigvals(G2)
    #     # eigVal[np.isinf(eigVal) + np.isnan(eigVal)] = -1.01
    #     # L = nx.normalized_laplacian_matrix(G)
    #     # eigVal = np.array(list(nx.eigenvector_centrality(G).values()))
    #     # print("eigVals lenght: ", eigVal.shape, flush=True)
    #     eigvals_time.append(eigVal)
    #     # adjc_time.append(adjc[t].cpu().detach().numpy())
    return E_a, E_p, E_t, adjc, G, avgZArr, clustCoef, degreeDist, eigVal, eigVec, env_noise_std, mean_agent_measurement_noise, n_agents, range_agent_measurement_noise, scalar_param1, steps, weight_own_belief, z_gt, zArr, pArr



# def process_file(base_file_path):
def parse_experiments(base_file_path):
    pathes = glob.glob(os.path.join(base_file_path, "*"))
    pool = Pool(10)
    dataframes = pool.imap(parse_experiment_data_par, tqdm(pathes))#[479400:-1]))#[0:460000]))
    #   dataframes.append(parse_experiment_data_par(p))
    #   dataframes = [parse_experiment_data(p) for p in tqdm(pathes)]
    return pd.concat(dataframes, ignore_index=True)

