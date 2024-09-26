from agent import BasicBayesianAgent
from collective import All2AllCollective
from environment import BasicGaussianEnvironment
import matplotlib.pyplot as plt
from util import dump_python_object
import dataclasses

class Simulator:

    def __init__(self, environment, collective):
        self.environment = environment
        self.collective = collective

    def step(self):
        self.collective.let_agents_take_one_action()

    def make_plots(self, diectory, show_plots=False):
        self.collective.create_plots(diectory, show_plots, self.environment.true_value)
        self.environment.create_plots(diectory, show_plots)

    def record_current_state(self, t):
        self.collective.record_current_state(t)
        self.environment.record_current_state(t)

    def save_recorded_data(self, filepath):
        d = {
            "collective_data": self.collective.get_recorded_states(),
            #"environment_data": self.environment.get_recorded_states(),
            }
        dump_python_object(d, filepath)



if __name__ == "__main__":
    shape = (10, 10)


    env = BasicGaussianEnvironment(true_value, std, shape)
    collective = All2AllCollective(n_agents, env, BasicBayesianAgent)
    collective.agents[0].measurement_noise_std = 0.0000001
    simulator = Simulator(env, collective)

    # setting a common color palette
    cmap = plt.get_cmap("plasma")
    env.set_belief_cmap(cmap)
    collective.set_belief_cmap(cmap)

    for i in range(steps):
        simulator.step()
        simulator.record_current_state(float(i))

        simulator.make_plots(None, False)
        plt.pause(1)

    plt.pause(2)
