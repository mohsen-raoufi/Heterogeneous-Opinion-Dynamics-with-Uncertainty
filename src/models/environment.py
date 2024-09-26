import torch
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

# from src.util import save_and_or_show_plot
from util import save_and_or_show_plot


class AbstractEnvironment(ABC):

    def __init__(self, true_value):
        self.true_value = true_value
        self.cmap = None
        self.current_state = dict()
        self.current_state["time"] = []
        self.current_state["true_value"] = []

    @abstractmethod
    def init_value_distribution(self, true_value):
        ...

    @abstractmethod
    def get_distribution_values(self, positions, stds=None):
        ...

    def create_plots(self, directory, show_plots):
        pass

    def set_belief_cmap(self, cmap):
        self.cmap = cmap

    def record_current_state(self, t):
        self.current_state["time"].append(t)
        self.current_state["true_value"].append(self.true_value)

    def get_recorded_states(self):
        return self.current_state.copy()


class BasicGaussianEnvironment(AbstractEnvironment):

    def __init__(self, true_value, std, shape):
        super(BasicGaussianEnvironment, self).__init__(true_value)
        self.shape = shape
        self.std = std
        self.distribution_map = torch.zeros(self.shape)
        self.init_value_distribution(true_value)
        self.current_state["distribution_map"] = []

    def init_value_distribution(self, true_value):
        self.distribution_map = torch.distributions.Normal(self.true_value, self.std).sample(self.shape)

    def get_distribution_values(self, positions, stds=None):
        idxs = torch.round(positions * (self.shape[0] - 1)).long()
        return self.distribution_map[idxs[:, 0], idxs[:, 1]]

    def create_plots(self, directory, show_plots):
        # plt.figure(4)
        # plt.imshow(self.distribution_map.detach().cpu(), cmap=self.cmap)
        name = "environment_value_distribution.png"
        # save_and_or_show_plot(directory, name, show_plots)

    def record_current_state(self, t):
        super(BasicGaussianEnvironment, self).record_current_state(t)
        self.current_state["distribution_map"].append(self.distribution_map.detach().cpu().numpy())


class GaussianEnvironmentFixedSTD(AbstractEnvironment):

    def __init__(self, true_value, std):
        super(GaussianEnvironmentFixedSTD, self).__init__(true_value)
        self.std = std
        self.distribution = torch.distributions.Normal(self.true_value, self.std)

    def init_value_distribution(self, true_value):
        pass

    def get_distribution_values(self, positions, stds=None):
        return self.distribution.sample(positions.shape[0:1])

    def create_plots(self, directory, show_plots):
        pass

    def record_current_state(self, t):
        super(GaussianEnvironmentFixedSTD, self).record_current_state(t)


class GaussianEnvironmentExternalSTD(AbstractEnvironment):

    def init_value_distribution(self, true_value):
        pass

    def __init__(self, true_value):
        super(GaussianEnvironmentExternalSTD, self).__init__(true_value)

    def get_distribution_values(self, positions, stds=None):
        return torch.normal(torch.ones_like(stds) * self.true_value, stds)

    def create_plots(self, directory, show_plots):
        pass

    def record_current_state(self, t):
        super(GaussianEnvironmentExternalSTD, self).record_current_state(t)
