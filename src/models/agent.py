import torch
import sys
import os
from .kf import update as kf_update
from .fast_covariance_intersection import fast_covariance_intersection
from abc import ABC, abstractmethod
import pandas as pd

import networkx as nx
import numpy as np

class AbstractAgent(ABC):

    def __init__(self, environment, collective, starting_position):
        self.environment = environment
        self.collective = collective
        self.position = starting_position
        self.current_state = dict()
        self.current_state["time"] = []
        self.current_state["position"] = []

    def take_asynchronous_action(self):
        self.take_action()
        self.synchronize_actions()

    @abstractmethod
    def synchronize_actions(self):
        ...

    @abstractmethod
    def take_action(self):
        ...

    @abstractmethod
    def get_current_estimate(self):
        ...

    def get_current_observation(self):
        obs = self.environment.get_distribution_values(self.position.unsqueeze(0))
        return obs

    def get_available_neighbors(self):
        neighbors = self.collective.get_available_neighbors(self)
        return neighbors

    def record_current_state(self, t):
        self.current_state["time"].append(t)
        self.current_state["position"].append(self.position.detach().cpu().numpy())

    def get_recorded_states(self):
        return pd.DataFrame(self.current_state)


class BasicBayesianAgent(AbstractAgent):

    def __init__(self, environment, collective, starting_position, randomize_parameter_config=False, n_neighbors=-1):
        super(BasicBayesianAgent, self).__init__(environment, collective, starting_position)
        self.action_counter = 0
        self.belief_mean = 0.0
        self.belief_var = torch.ones(1)
        self.tmp_belief_mean = 0.0
        self.tmp_belief_var = 1.0

        self.current_state["belief_mean"] = []
        self.current_state["belief_std"] = []

        if randomize_parameter_config:
            self.measurement_noise_std = torch.distributions.Uniform(collective.min_agent_measurement_noise, collective.max_agent_measurement_noise).sample()
        else:
            # to make the amount of samples taken the same we take a sample
            _ = torch.distributions.Uniform(0, 1).sample()
            self.measurement_noise_std = collective.min_agent_measurement_noise + (collective.max_agent_measurement_noise - collective.min_agent_measurement_noise) * 0.5

    def get_current_observation(self):
        obs = self.environment.get_distribution_values(self.position.unsqueeze(0), torch.ones(1) * self.measurement_noise_std)
        return obs

    def synchronize_actions(self):
        self.belief_mean, self.belief_var = self.tmp_belief_mean, self.tmp_belief_var
        self.action_counter += 1

    def take_action(self):
        if self.action_counter == 0:
            # init belief with local observation
            obs = self.get_current_observation()
            self.tmp_belief_mean = obs
            self.tmp_belief_var = torch.pow(torch.ones(1, 1) * self.measurement_noise_std, 2)
        else:
            neighbors = self.get_available_neighbors()
            H = torch.ones(1, 1)
            order_idxs = torch.randperm(len(neighbors), dtype=torch.int).detach().cpu().numpy()
            weight_array = torch.zeros(len(neighbors)+1)
            self.tmp_belief_mean, self.tmp_belief_var = self.belief_mean, self.belief_var
            for i in order_idxs:
                n_mean, n_var = neighbors[i].get_current_estimate()
                self.tmp_belief_mean, self.tmp_belief_var, weights = self.neighbor_belief_inference(self.tmp_belief_mean, self.tmp_belief_var, H, n_mean, n_var)
                if torch.sum(weight_array) == 0:
                    weight_array[0] = weights[0].squeeze()
                    weight_array[i+1] = weights[1].squeeze()
                else:
                    weight_array = weight_array * weights[0].squeeze()
                    weight_array[i+1] = weights[1].squeeze()
            self.collective.report_back_weights(self, weight_array)

    def neighbor_belief_inference(self, mean, var, H, n_mean, n_var):
        return kf_update(mean, var, n_mean, H, lambda x: n_var)

    def get_current_estimate(self):
        return self.belief_mean, self.belief_var

    def record_current_state(self, t):
        super(BasicBayesianAgent, self).record_current_state(t)
        self.current_state["belief_mean"].append(self.belief_mean)
        self.current_state["belief_std"].append(torch.sqrt(self.belief_var))


class CIBayesianAgent(BasicBayesianAgent):

    def neighbor_belief_inference(self, mean, var, H, n_mean, n_var):
        means = torch.cat([H.mv(mean).unsqueeze(0), n_mean.unsqueeze(0)], dim=0)
        Sigmas = torch.cat([torch.mm(torch.mm(H, var), H.transpose(0, 1)).unsqueeze(0), n_var.unsqueeze(0)], dim=0)
        mean, var, weights = fast_covariance_intersection(means, Sigmas, is_batch=False)
        return mean, var, weights


class NaiveAgent(AbstractAgent):

    def __init__(self, environment, collective, starting_position, randomize_parameter_config=False, n_neighbors=-1):
        super(NaiveAgent, self).__init__(environment, collective, starting_position)
        self.action_counter = 0
        self.belief_mean = 0.0
        self.belief_var = 0.0
        self.tmp_belief_mean = 0.0
        self.tmp_belief_var = 0.0
        self.weight_own_belief = collective.weight_own_belief
        self.weight_others_belief = collective.weight_others_belief
        self.collective = collective

        if randomize_parameter_config:
            self.measurement_noise = torch.distributions.Uniform(collective.min_agent_measurement_noise, collective.max_agent_measurement_noise).sample()
        else:
            # to make the amount of samples taken the same we take a sample
            _ = torch.distributions.Uniform(0, 1).sample()
            self.measurement_noise = collective.min_agent_measurement_noise + (collective.max_agent_measurement_noise - collective.min_agent_measurement_noise) * 0.5

        self.current_state["belief_mean"] = []
        self.current_state["belief_std"] = []

    def synchronize_actions(self):
        self.belief_mean, self.belief_var = self.tmp_belief_mean, self.tmp_belief_var
        self.action_counter += 1

    def get_current_observation(self):
        obs = self.environment.get_distribution_values(self.position.unsqueeze(0), torch.ones(1) * self.measurement_noise)
        return obs

    def take_action(self):
        if self.action_counter == 0:
            # init belief with local observation
            obs = self.get_current_observation()
            self.tmp_belief_mean = obs
            self.tmp_belief_var = torch.zeros(1, 1)
        else:
            neighbors = self.get_available_neighbors()
            if len(neighbors) > 0:
                values = [n.get_current_estimate()[0] for n in neighbors]
                self.tmp_belief_mean = self.neighborhood_inference(values)
                self.report_weights_back(neighbors)

    def report_weights_back(self, neighbors):
        self.collective.report_back_weights(self, torch.tensor(
            [self.weight_own_belief] + [self.weight_own_belief / float(len(neighbors)) for _ in range(len(neighbors))]))

    def neighborhood_inference(self, values):
        return self.belief_mean * self.weight_own_belief + torch.mean(torch.cat(values, dim=0), dim=0,
                                                                      keepdim=True) * self.weight_others_belief

    def get_current_estimate(self):
        return self.belief_mean, self.belief_var

    def record_current_state(self, t):
        super(NaiveAgent, self).record_current_state(t)
        self.current_state["belief_mean"].append(self.belief_mean)
        self.current_state["belief_std"].append(self.belief_var)


class NaiveLocalOptimal(NaiveAgent):

    def report_weights_back(self, neighbors):
        self.collective.report_back_weights(self, torch.tensor(
            [1.0 / float(len(neighbors) + 1) for _ in range(len(neighbors) + 1)]))

    def neighborhood_inference(self, values):
        return torch.mean(torch.tensor(values + [self.belief_mean]))


class BayesCorrel(BasicBayesianAgent):

    def __init__(self, environment, collective, starting_position, randomize_parameter_config=False, n_neighbors=-1):
        super().__init__(environment, collective, starting_position, randomize_parameter_config)
        n_agents = collective.n_agents

        self_value = float(n_neighbors) / n_agents
        self.measurement_noise_std = net_inf_cor_generator(self_value, Gamma=self.collective.correlation_network_information, b=5)


class BayesCorrelCircularPlot(BasicBayesianAgent):

    def __init__(self, environment, collective, starting_position, randomize_parameter_config=False, n_neighbors=-1):
        super().__init__(environment, collective, starting_position, randomize_parameter_config)
        n_agents = collective.n_agents

        if n_neighbors == 0:
            self_value = 1
        else:
            self_value = 0

        self.measurement_noise_std = net_inf_cor_generator(self_value, Gamma=self.collective.correlation_network_information, b=5)
        print(self.measurement_noise_std)


class BayesCICorrel(CIBayesianAgent):

    def __init__(self, environment, collective, starting_position, randomize_parameter_config=False, n_neighbors=-1):
        super().__init__(environment, collective, starting_position, randomize_parameter_config)
        n_agents = collective.n_agents

        np_G = nx.adjacency_matrix(self.collective.G).toarray()
        degree = np_G.sum(axis=0)
        degree_ratio = degree / n_agents
        self_degree = float(n_neighbors) / n_agents
        # normalize degree_ratio so that the minimum is 0 and the maximum is 1
        self_degree_norm = (self_degree - np.min(degree_ratio)) / (np.max(degree_ratio) - np.min(degree_ratio))
        
        self.measurement_noise_std = net_inf_cor_generator(self_degree_norm, Gamma=self.collective.correlation_network_information, b=5)


class BayesCorrelMismodel(BasicBayesianAgent):

    def __init__(self, environment, collective, starting_position, randomize_parameter_config=False, n_neighbors=-1):
        super().__init__(environment, collective, starting_position, randomize_parameter_config)
        n_agents = collective.n_agents
        self.measurement_noise_std = 0.02 - 0.02 * self.collective.correlation_network_information * float(n_neighbors) / float(
            n_agents)
        print(self.measurement_noise_std)


class BayesCICorrelMismodel(CIBayesianAgent):

    def __init__(self, environment, collective, starting_position, randomize_parameter_config=False, n_neighbors=-1):
        super().__init__(environment, collective, starting_position, randomize_parameter_config)
        n_agents = collective.n_agents
        self.measurement_noise_std = 0.02 - 0.02 * self.collective.correlation_network_information * float(n_neighbors) / float(
            n_agents)


class NaiveCorrel(NaiveAgent):

    def __init__(self, environment, collective, starting_position, randomize_parameter_config=False, n_neighbors=-1):
        super().__init__(environment, collective, starting_position, randomize_parameter_config)
        n_agents = collective.n_agents
        
        np_G = nx.adjacency_matrix(self.collective.G).toarray()
        degree = np_G.sum(axis=0)
        degree_ratio = degree / n_agents;
        self_degree = float(n_neighbors) / n_agents
        # normalize degree_ratio so that the minimum is 0 and the maximum is 1
        self_degree_norm = (self_degree - np.min(degree_ratio)) / (np.max(degree_ratio) - np.min(degree_ratio))
        
        self.measurement_noise_std = net_inf_cor_generator(self_degree_norm, Gamma=self.collective.correlation_network_information, b=5)

class NaiveLOCorrel(NaiveLocalOptimal):

    def __init__(self, environment, collective, starting_position, randomize_parameter_config=False, n_neighbors=-1):
        super().__init__(environment, collective, starting_position, randomize_parameter_config)
        n_agents = collective.n_agents
        
        np_G = nx.adjacency_matrix(self.collective.G).toarray()
        degree = np_G.sum(axis=0)
        degree_ratio = degree / n_agents;
        self_degree = float(n_neighbors) / n_agents
        # normalize degree_ratio so that the minimum is 0 and the maximum is 1
        self_degree_norm = (self_degree - np.min(degree_ratio)) / (np.max(degree_ratio) - np.min(degree_ratio))
        
        self.measurement_noise_std = net_inf_cor_generator(self_degree_norm, Gamma=self.collective.correlation_network_information, b=5)


def net_inf_cor_generator(x, Gamma=1, b=5):
    a = Gamma / (np.exp(b) - 1)
    c = -a
    if(Gamma>0):
        y = a * np.exp(b * x) + c
    else:
        y = -a * np.exp(-b * (x - 1)) - c

    y = y + 10**-2
    return y