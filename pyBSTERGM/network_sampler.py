import numpy as np
import matplotlib.pyplot as plt

from network import UndirectedNetwork



class NetworkSampler:
    initial_network = 0
    param = np.array(0)
    network_samples = []
    node_num = 0
    random_gen = 0

    def __init__(self, model_fn, param, init_network: UndirectedNetwork, rng_seed=2021):
        self.param = param
        self.initial_network = init_network
        self.network_samples.append(init_network)
        self.model = model_fn
        self.node_num = init_network.node_num
        self.random_gen = np.random.default_rng(seed=rng_seed)

    def choose_edge(self):
        edge_idx = (0, 0)
        while(edge_idx[0]==edge_idx[1]):
            edge_idx = self.random_gen.integers(low=0, high=self.node_num - 1, size=2)
        return edge_idx

    def propose_network(self, last_network: UndirectedNetwork):
        proposed_structure = last_network.structure.copy()
        edge_idx = self.choose_edge()
        proposed_structure[edge_idx[0], edge_idx[1]] = 1 - proposed_structure[edge_idx[0], edge_idx[1]]
        proposed_structure[edge_idx[1], edge_idx[0]] = proposed_structure[edge_idx[0], edge_idx[1]]
        return (UndirectedNetwork(proposed_structure))

    def log_r(self, last_network, proposed_network):
        proposed_network_netStat = self.model(proposed_network)
        last_network_netStat = self.model(last_network)
        return np.dot(proposed_network_netStat - last_network_netStat, self.param)


    def sampler(self):
        last_network = self.network_samples[-1]
        proposed_network = self.propose_network(last_network)
        log_r_val = self.log_r(last_network, proposed_network)
        unif_sample = self.random_gen.random()
        if np.log(unif_sample) < log_r_val:
            self.network_samples.append(proposed_network)
        else:
            self.network_samples.append(last_network)

    def run(self, iter):
        for _ in range(iter):
            self.sampler()

    def netStat_trace(self):
        netStat = []
        for _ in range(len(self.param)):
            netStat.append([])

        for network in self.network_samples:
            each_netStat = self.model(network)
            for i, stat in enumerate(each_netStat):
                netStat[i].append(stat)
        return netStat

    def show_traceplot(self, show=True):
        netStat = self.netStat_trace()
        
        grid_column = 1
        # grid_row = int(len(netStat)/2+0.51)
        grid_row = 2
        plt.figure(figsize=(5*grid_column, 3*grid_row))
        for i, statSeq in enumerate(netStat):
            plt.subplot(grid_row, grid_column, i+1)
            plt.plot(range(len(statSeq)), statSeq)
        
        if show:
            plt.show()

if __name__ == "__main__":

    def model_netStat(network : UndirectedNetwork):
        model = []
        #define model
        model.append(network.statCal_edgeNum())
        model.append(network.statCal_geoWeightedESP(0.5))
        return np.array(model)


    test_structure = np.array(
        [
            [0,1,1,0,0,0,0,0,0,0],
            [1,0,1,1,0,0,0,0,0,0],
            [1,1,0,1,0,0,0,0,0,0],
            [0,1,1,0,1,0,0,0,0,0],
            [0,0,0,1,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0]
        ]
        )
    test_initnet = UndirectedNetwork(test_structure)
    test_netSampler = NetworkSampler(model_netStat, np.array([0, 0]), test_initnet)
    test_netSampler.run(30000)
    test_netSampler.show_traceplot()
