import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import torch
from abc import ABC, abstractmethod
# from src.util import save_and_or_show_plot
from util import save_and_or_show_plot
from dataclasses import dataclass

import networkx as nx


class AbstractCollective(ABC):

    def __init__(self, n_agents, environment, hack_for_circular_plot=False):
        self.n_agents = n_agents
        self.environment = environment
        self.agents = []
        self.cmap = None
        self.hack_for_circular_plot = hack_for_circular_plot
        self.current_state = dict()
        self.current_state["time"] = []
        self.current_state["connectivity"] = []
        self.timeStep = 0
        self.E_true_0 = 0
        self.std_agent_measurement_noise = 0.5
        self.mean_agent_measurement_noise = 0.2

    @abstractmethod
    def generate_agents(self):
        ...

    @abstractmethod
    def get_available_neighbors(self, agent):
        ...

    @abstractmethod
    def let_agents_take_one_action(self):
        ...

    def create_plots(self, directory, show_plots, true_value):
        pass

    def set_belief_cmap(self, cmap):
        self.cmap = cmap

    @abstractmethod
    def get_current_weighted_connectivity_map(self):
        ...

    def record_current_state(self, t):
        for a in self.agents:
            a.record_current_state(t)
        cm = self.get_current_weighted_connectivity_map()
        self.current_state["time"].append(t)
        self.current_state["connectivity"].append(cm)

    def get_recorded_states(self):
        agent_data = {}
        for i, a in enumerate(self.agents):
            agent_data[i] = a.get_recorded_states()
        d = self.current_state.copy()
        d["agent_data"] = agent_data
        return d


@dataclass
class NetworkParams:
    network_type: str = "spatial"
    scalar_param1: float = 0.5
    n_agents: int = 10
    # These are old
    mean_agent_measurement_noise: float = 0.2
    std_agent_measurement_noise: float = 0.5
    # New ones
    min_agent_measurement_noise: float = 0.01
    max_agent_measurement_noise: float = 0.5

    weight_own_belief: float = 0.5
    weight_others_belief: float = 0.5

    correlation_network_information: float = 0.0


class Network(AbstractCollective):

    def __init__(self, environment, agent_class, network_params: NetworkParams, network_seed, hack_for_circular_plot=False):
        self._agent_class = agent_class
        self.synchronous = True

        super(Network, self).__init__(network_params.n_agents, environment, hack_for_circular_plot=hack_for_circular_plot)

        # These are old
        self.mean_agent_measurement_noise = network_params.mean_agent_measurement_noise
        self.std_agent_measurement_noise = network_params.std_agent_measurement_noise

        # New ones
        self.min_agent_measurement_noise = network_params.min_agent_measurement_noise
        self.max_agent_measurement_noise = network_params.max_agent_measurement_noise

        if self.max_agent_measurement_noise - self.min_agent_measurement_noise > 0:
            self.randomized_agent_configs = True
        else:
            self.randomized_agent_configs = False

        self.correlation_network_information = network_params.correlation_network_information

        # For Naiive
        self.weight_own_belief = network_params.weight_own_belief
        self.weight_others_belief = network_params.weight_others_belief

        if network_params.network_type == "spatial":
            param = network_params.scalar_param1

            r_link = param

            # Use seed when creating the graph for reproducibility
            G = nx.random_geometric_graph(self.n_agents, r_link, seed=int(network_seed))
            # position is stored as node attribute data for random_geometric_graph

        elif network_params.network_type == "spatial_uniform":
            x = np.random.uniform(-1, 1, self.n_agents).T  # TODO: we can modify this later, to get pos from outside
            y = np.random.uniform(-1, 1, self.n_agents).T
            G = nx.empty_graph(n=self.n_agents)
            r_link = network_params.scalar_param1

            for node_i in range(self.n_agents):
                G.nodes[node_i]['pos'] = [x[node_i], y[node_i]]
                for node_j in range(node_i, self.n_agents):
                    d = (x[node_i] - x[node_j]) ** 2 + (y[node_i] - y[node_j]) ** 2
                    if d < r_link:
                        G.add_edge(node_i, node_j)

        elif network_params.network_type == "spatial_normal":
            mean = network_params.scalar_param_mean
            cov = network_params.scalar_param_cov
            x, y = np.random.multivariate_normal(mean, cov, self.n_agents).T
            G = nx.empty_graph(n=self.n_agents)
            r_link = network_params.scalar_param1

            for node_i in range(self.n_agents):
                G.nodes[node_i]['pos'] = [x[node_i], y[node_i]]
                for node_j in range(node_i, self.n_agents):
                    d = (x[node_i] - x[node_j]) ** 2 + (y[node_i] - y[node_j]) ** 2
                    if d < r_link:
                        G.add_edge(node_i, node_j)

        elif network_params.network_type == "centralized_random":
            x = np.random.uniform(-1, 1, self.n_agents).T  # TODO: we can modify this later, to get pos from outside
            y = np.random.uniform(-1, 1, self.n_agents).T
            # TODO: we may need to add a new parameter named "p_add_wire" instead of scalar_param1
            G = self.get_updated_central_graph_with_random_edges(n_nodes=network_params.n_agents, p_add_wire=network_params.scalar_param1,
                                                                 p_cut_wire=0.0)

            # TODO: the following loop can be integrated with the graph generation one in the function above.
            for node_i in range(self.n_agents):
                G.nodes[node_i]['pos'] = [x[node_i], y[node_i]]

        elif network_params.network_type == "centralized_random_fixed_mdeg":
            #
            G = self.get_updated_central_graph_with_random_edges_fixed_mdeg_connected(n_nodes=network_params.n_agents,
                                                                                      p_change=network_params.scalar_param1)

            x = np.random.uniform(-1, 1, self.n_agents).T  # TODO: we can modify this later, to get pos from outside
            y = np.random.uniform(-1, 1, self.n_agents).T
            # TODO: the following loop can be integrated with the graph generation one in the function above.
            for node_i in range(self.n_agents):
                G.nodes[node_i]['pos'] = [x[node_i], y[node_i]]


        elif network_params.network_type == "k_regular":

            G = self.gen_even_k_reg_network(n_nodes=network_params.n_agents, k_reg=network_params.scalar_param1)

            x = np.random.uniform(-1, 1, self.n_agents).T  # TODO: we can modify this later, to get pos from outside
            y = np.random.uniform(-1, 1, self.n_agents).T
            # TODO: the following loop can be integrated with the graph generation one in the function above.
            for node_i in range(self.n_agents):
                G.nodes[node_i]['pos'] = [x[node_i], y[node_i]]

        else:
            raise ValueError("Unkown network type: " + str(network_params.network_type))

        self.G = G
        self.positions = nx.get_node_attributes(G, "pos")
        self.connectivity_map = torch.from_numpy(nx.adjacency_matrix(G).toarray())
        self.reinit_weighted_connectivity_map()

        self.generate_agents()

    def get_updated_central_graph_with_random_edges(self, n_nodes=100, p_add_wire=0.0, p_cut_wire=0.0):
        G_star = nx.star_graph(n_nodes - 1)
        for node_i in G_star.nodes:
            for node_j in G_star.nodes:
                if node_i == node_j:
                    continue
                if (p_add_wire > 0):
                    if (np.random.rand() < p_add_wire):
                        G_star.add_edge(node_i, node_j)
                        G_star.add_edge(node_j, node_i)
                        # print("Edge added between " + str(node_i) + ", ", str(node_j))
                if (p_cut_wire > 0):
                    if (G_star.has_edge(node_i, node_j) & (np.random.rand() < p_cut_wire) & (
                            (G_star.degree[node_i] + G_star.degree[node_j]) > 2)):
                        G_star.remove_edge(node_i, node_j)
                        # print("Edge removed between " + str(node_i) + ", ", str(node_j))

        return G_star

    # Add edges based on the mat, remove the edges based on a the number of edges needed to be removed
    # # Issue: the giant component size is not fixed
    def get_updated_central_graph_with_random_edges_fixed_mdeg_disconnected(self, n_nodes=100, p_change=0.0):
        # make a warning if p_change is greater than 1
        if (p_change > 1):
            print("WARNING!! p_change is greater than 1. It is set to 1.")
            p_change = 1

        adjc = np.zeros((n_nodes, n_nodes))
        adjc[0, 1:] = 1  # making it a star graph
        adjc = adjc + adjc.T  # making it symmetric

        # create a random matrix for adding edges
        mat_pot_add_edge = np.random.rand(n_nodes, n_nodes)

        # make the lower triangle of mat_potential_add_edge zero
        mat_pot_add_edge = np.triu(mat_pot_add_edge, 1)

        # make sure that the diagonal elements are not selected to be added => the value is zero
        mat_pot_add_edge = mat_pot_add_edge + np.zeros(n_nodes)

        # the edges with random variable greater than (1-p_change) will be added
        adding_link = mat_pot_add_edge > (1 - p_change)
        adding_link = adding_link + adding_link.T

        # compare the adding_link with adjc, and see how many edges should be added
        # make a difference of the two matrices
        new_links_mat = np.logical_and(adding_link, np.logical_not(adjc))

        # calculate number of edges to be removed, based on the adding links and the number of mutual links
        # it should be divided by 2, because the removing_link will be forced to be symmetric
        n_edges_to_be_removed = int(np.sum(new_links_mat) / 2)

        # update the adjc by adding the new links
        adjc = adjc + new_links_mat

        # pick n_edges_to_be_removed edges randomly from the adjc and remove them
        # find the indices of the edges
        edge_indices = np.where(np.triu(adjc, 1))
        # pick n edges_to_be_removed edges randomly without repetition
        edge_indices_to_be_removed = np.random.choice(len(edge_indices[0]), n_edges_to_be_removed, replace=False)

        # remove the edges
        for i_edge in edge_indices_to_be_removed:
            if (adjc[edge_indices[0][i_edge], edge_indices[1][i_edge]] == 0):
                print("ERROR: the edge is already removed")
            adjc[edge_indices[0][i_edge], edge_indices[1][i_edge]] = 0
            adjc[edge_indices[1][i_edge], edge_indices[0][i_edge]] = 0

        G = nx.from_numpy_array(adjc)

        return G

    # using matrices. Add edges based on the mat, remove the edges based on a the number of edges needed to be removed
    # # fixed issue of type4: add links so that the network becomes connected
    def get_updated_central_graph_with_random_edges_fixed_mdeg_connected(self, n_nodes=100, p_change=0.0):

        G = self.get_updated_central_graph_with_random_edges_fixed_mdeg_disconnected(n_nodes=n_nodes, p_change=p_change)

        # check if the network is connected
        if (nx.is_connected(G)):
            # print("The network is already connected")
            pass
        else:
            # add links from random components to another random component until the network becomes connected
            while (not nx.is_connected(G)):
                # find the components
                components = list(nx.connected_components(G))
                # pick two random components without repetition at the same time
                component_picked = np.random.choice(len(components), 2, replace=False)
                component_1 = component_picked[0]
                component_2 = component_picked[1]
                # pick a random node from each component
                node_1 = np.random.choice(list(components[component_1]))
                node_2 = np.random.choice(list(components[component_2]))
                # add an edge between the two nodes
                G.add_edge(node_1, node_2)
        return G
    
    def gen_even_k_reg_network(self, n_nodes=10, k_reg = 4):
        # assert if k_reg is even, show a warning if it is odd
        if(k_reg%2 == 1):
            print("WARNING: k_reg is odd. It should be even.")
        assert k_reg%2 == 0, "K_reg is odd. It should be even" # k_reg should be even

        G_null = nx.empty_graph(n_nodes)
        # make a loop over the nodes and connect them with the next k_reg nodes
        for i_node in range(n_nodes):
            # add edges between the node and the next k_reg nodes
            for j_node in range(i_node-int(np.ceil(k_reg/2)), i_node+int(np.ceil(k_reg/2))+1):
                if (j_node == i_node):
                    continue
                # make a link between i_node and mod(i_node,n_nodes)
                G_null.add_edge(i_node, j_node%n_nodes)

        G_k_reg = G_null.copy()
        return G_k_reg

    def generate_agents(self):
        if self.hack_for_circular_plot:
            # TODO : THIS IS A HECK! we just changed n_neighbors to i so that we create weird correlation
            self.agents = [self._agent_class(self.environment, self, torch.tensor(self.positions[i]),
                                             randomize_parameter_config=self.randomized_agent_configs,
                                             n_neighbors=i) # torch.sum(self.connectivity_map[i]))
                           for i in range(self.n_agents)]
            print("hack")
        else:
            self.agents = [self._agent_class(self.environment, self, torch.tensor(self.positions[i]),
                                             randomize_parameter_config=self.randomized_agent_configs,
                                             n_neighbors=torch.sum(self.connectivity_map[i]))
                           for i in range(self.n_agents)]

    def get_available_neighbors(self, agent):
        idx = self.agents.index(agent)
        connection_mask = self.connectivity_map[idx].bool()
        connection_mask[idx] = False
        connection_idxs = torch.arange(self.connectivity_map.shape[0])[connection_mask]
        n = [self.agents[int(connection_idxs[i].item())] for i in range(connection_idxs.shape[0])]
        return n

    def report_back_weights(self, agent, weight_array):
        idx = self.agents.index(agent)
        connection_mask = self.connectivity_map[idx]
        connection_idxs = torch.arange(self.connectivity_map.shape[0])[connection_mask.bool()]
        adjacency_row = torch.zeros_like(connection_mask).double()
        for i, agent_idx in enumerate(connection_idxs):
            adjacency_row[agent_idx] = weight_array[i + 1]
        adjacency_row[idx] = weight_array[0]
        self.weighted_connectivity_map[idx] = adjacency_row

    def let_agents_take_one_action(self, order_idxs=None):
        self.reinit_weighted_connectivity_map()
        if order_idxs is None:
            order_idxs = torch.randperm(len(self.agents), dtype=torch.int).detach().cpu().numpy()
        if self.synchronous:
            for i in order_idxs:
                self.agents[i].take_action()
            for a in self.agents:
                a.synchronize_actions()
        else:
            for i in order_idxs:
                self.agents[i].take_asynchronous_action()

    def reinit_weighted_connectivity_map(self):
        self.weighted_connectivity_map = torch.clone(self.connectivity_map).double()
        normalizer = torch.sum(self.weighted_connectivity_map, dim=1, keepdim=True) + self.weight_own_belief
        self.weighted_connectivity_map = self.weighted_connectivity_map / normalizer
        self.weighted_connectivity_map = self.weighted_connectivity_map + \
                                         torch.eye(self.weighted_connectivity_map.shape[0]) * self.weight_own_belief

    def get_current_weighted_connectivity_map(self):
        return self.weighted_connectivity_map


class All2AllCollective(AbstractCollective):

    def __init__(self, n_agents, environment, agent_class):
        super(All2AllCollective, self).__init__(n_agents, environment)
        self._agent_class = agent_class
        self.synchronous = True
        self.randomized_agent_configs = True

        # initialize_agents
        self.generate_agents()

    def get_current_weighted_connectivity_map(self):
        return (torch.ones(len(self.agents), len(self.agents)) - torch.eye(len(self.agents))).cpu().numpy()

    def generate_agents(self):
        starting_positions = self.generate_starting_positions()
        self.agents = [self._agent_class(self.environment, self, starting_positions[i],
                                         randomize_parameter_config=self.randomized_agent_configs)
                       for i in range(self.n_agents)]

    def generate_starting_positions(self):
        values_for_dimensions = []
        for dim_max in self.environment.shape:
            values_for_dimensions.append(torch.rand((self.n_agents, 1)) * (dim_max - 1.0))
        positions = torch.cat(values_for_dimensions, dim=1)
        return positions

    def get_available_neighbors(self, agent):
        n = self.agents.copy()
        n.remove(agent)
        return n

    def let_agents_take_one_action(self, order_idxs=None):
        if order_idxs is None:
            order_idxs = torch.randperm(len(self.agents), dtype=torch.int).detach().cpu().numpy()
        if self.synchronous:
            for i in order_idxs:
                self.agents[i].take_action()
            for a in self.agents:
                a.synchronize_actions()
        else:
            for i in order_idxs:
                self.agents[i].take_asynchronous_action()

    def plot_current_connectivity(self, positions):
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                f1 = plt.figure(1)
                # f1.canvas.manager.window.wm_geometry("+1500+0")
                plt.plot([positions[i][0], positions[j][0]],
                         [positions[i][1], positions[j][1]],
                         color="black", linestyle="--")

    def create_plots(self, directory, show_plots, true_value):
        positions = [a.position.detach().cpu() for a in self.agents]
        estimates = [a.get_current_estimate()[0].detach().cpu().numpy() for a in self.agents]
        certainties = [a.get_current_estimate()[1].detach().cpu().numpy() for a in self.agents]
        colors = self.cmap(estimates)
        sizes = [c * 500.0 for c in certainties]
        self.plot_current_connectivity(positions)
        plt.scatter([positions[i][0] for i in range(self.n_agents)],
                    [positions[i][1] for i in range(self.n_agents)],
                    s=sizes,
                    c=colors)

        x = np.concatenate([np.linspace(-2, 2, 1000), np.array(estimates)[:, 0]])
        x.sort()
        max_val = 0.0
        plt.cla()
        for i in range(self.n_agents):
            plt.plot(x, stats.norm.pdf(x, estimates[i], certainties[i])[0], label="Belief Agent " + str(i))
            max_val = np.maximum(np.max(stats.norm.pdf(x, estimates[i], certainties[i])[0]), max_val)
        plt.vlines(true_value, -max_val * 0.1, max_val * 1.1, colors="red", linestyles="--", label="True Value")
        plt.legend()
        plt.ylim(-max_val * 0.1, max_val * 1.1)

        plts = []
        E_prec = self.n_agents ** 2 * np.std(estimates)
        E_true = np.linalg.norm(estimates - true_value * np.ones(estimates.__sizeof__()))
        plts += plt.plot(self.timeStep, E_prec, "*b", label="Precision Error")
        plts += plt.plot(self.timeStep, E_true, ".r", label="Trueness Error")

        if self.timeStep == 0:
            self.E_true_0 = E_true

        plt.plot([0, self.timeStep], [self.E_true_0, self.E_true_0], '--r')
        plt.legend(plts, ['E_Prec', 'E_True'])
        self.timeStep = self.timeStep + 1


class CentralityTreeCollective(All2AllCollective):
    # NOTE: position and connectivity are currently not related

    def __init__(self, n_agents, environment, agent_class):
        self.connectivity_map = torch.zeros(1, 1)
        num_start_children = int(torch.randint(2, 7, (1,)).cpu().item())
        for i in range(num_start_children):
            self.generate_next_agent_connectivity(0, num_start_children)

        n_agents = self.connectivity_map.shape[0]
        super(CentralityTreeCollective, self).__init__(n_agents, environment, agent_class)

    def generate_next_agent_connectivity(self, parent_idx, num_parents_children):
        num_agents_before = self.connectivity_map.shape[0]
        tmp_connectivity_map = torch.zeros((num_agents_before + 1, num_agents_before + 1))
        tmp_connectivity_map[:num_agents_before, :num_agents_before] = self.connectivity_map
        self.connectivity_map = tmp_connectivity_map
        self.connectivity_map[num_agents_before, parent_idx] = 1
        self.connectivity_map[parent_idx, num_agents_before] = 1
        if num_parents_children == 1:
            return
        min_number_of_children = num_parents_children - 4
        if min_number_of_children < 0:
            min_number_of_children = 0
        num_own_children = int(torch.randint(min_number_of_children, num_parents_children - 1, (1,)).cpu().item())
        for i in range(num_own_children):
            self.generate_next_agent_connectivity(num_agents_before, num_own_children)

    def plot_current_connectivity(self, positions):
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                if self.connectivity_map[i, j] == 1:
                    f1 = plt.figure(1)
                    # f1.canvas.manager.window.wm_geometry("+1500+0")
                    plt.plot([positions[i][0], positions[j][0]],
                             [positions[i][1], positions[j][1]],
                             color="black", linestyle="--")

    def get_available_neighbors(self, agent):
        idx = self.agents.index(agent)
        connection_mask = self.connectivity_map[idx].bool()
        connection_mask[idx] = False
        connection_idxs = torch.arange(self.connectivity_map.shape[0])[connection_mask]
        n = [self.agents[int(connection_idxs[i].item())] for i in range(connection_idxs.shape[0])]
        return n

    def get_current_weighted_connectivity_map(self):
        return self.connectivity_map.cpu().numpy()
