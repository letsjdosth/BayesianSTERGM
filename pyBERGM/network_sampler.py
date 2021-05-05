import numpy as np
import matplotlib.pyplot as plt

from network import UndirectedNetwork, DirectedNetwork



class NetworkSampler:
    def __init__(self, model_fn, param, init_network, is_formation=None, constraint_net=None, rng_seed=2021):
        #variables
        self.initial_network = 0
        self.param = np.array(0)
        self.network_samples = []
        self.node_num = 0
        self.random_gen = 0
        self.mutable_edges = []

        #initialize
        self.param = param
        self.initial_network = init_network
        self.network_samples.append(init_network)
        self.model = model_fn
        self.node_num = init_network.node_num
        self.random_gen = np.random.default_rng(seed=rng_seed)
        self.is_formation = is_formation
        self.constraint_net = constraint_net
        self.make_mutable_edges_list(self.is_formation, constraint_net)


    def make_mutable_edges_list(self, is_formation, constraint_net):
        if is_formation is None: #no constraint
            for i in range(self.node_num):
                for j in range(self.node_num):
                    if i==j:
                        pass
                    else:
                        self.mutable_edges.append((i,j))
        
        elif is_formation:
            constraint_net_structure = self.constraint_net.structure
            # we can change 0s
            for i in range(self.node_num):
                for j in range(self.node_num):
                    if i==j:
                        pass
                    elif constraint_net_structure[i][j]==0:
                        self.mutable_edges.append((i,j))
        else: #dissolution
            constraint_net_structure = self.constraint_net.structure
            # we can change 1s
            for i in range(self.node_num):
                for j in range(self.node_num):
                    if constraint_net_structure[i][j]==1:
                        self.mutable_edges.append((i,j))

    # def choose_edge(self):
    #     edge_idx = (0, 0)
    #     while(edge_idx[0]==edge_idx[1]):
    #         edge_idx = self.random_gen.integers(low=0, high=self.node_num - 1, size=2)
    #     return edge_idx

    def choose_mutable_edge(self):
        rnd_idx = self.random_gen.integers(low=0, high=len(self.mutable_edges))
        edge_idx = self.mutable_edges[rnd_idx]
        return edge_idx


    def propose_network(self, last_network):
        proposed_structure = last_network.structure.copy()
        edge_idx = self.choose_mutable_edge() #choose_edge or choose_mutable_edge
        result_network = 0
        if isinstance(last_network, UndirectedNetwork):
            proposed_structure[edge_idx[0], edge_idx[1]] = 1 - proposed_structure[edge_idx[0], edge_idx[1]]
            proposed_structure[edge_idx[1], edge_idx[0]] = proposed_structure[edge_idx[0], edge_idx[1]]
            result_network = UndirectedNetwork(proposed_structure)
        elif isinstance(last_network, DirectedNetwork):
            proposed_structure[edge_idx[0], edge_idx[1]] = 1 - proposed_structure[edge_idx[0], edge_idx[1]]
            result_network = DirectedNetwork(proposed_structure)
        return result_network

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
        grid_row = int(len(netStat))
        
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

    #1
    
    test_netSampler = NetworkSampler(model_netStat, np.array([2, -1]), test_initnet, is_formation=True)
    # test_netSampler.make_mutable_edges_list(is_formation=False)
    # print(test_netSampler.mutable_edges)
    # print(test_netSampler.choose_mutable_edge())
    

    test_netSampler.run(50)
    test_netSampler.show_traceplot()


    #2

    # test_initnet2 = DirectedNetwork(test_structure)
    # test_netSampler = NetworkSampler(model_netStat, np.array([0, 0]), test_initnet)
    # test_netSampler.run(30000)
    # test_netSampler.show_traceplot()

    # test_newNetSampler = NetworkSampler_integrated(model_netStat, np.array([0]), np.array([0]), test_initnet)
    # print(test_newNetSampler.formation_edges)
    # print(test_newNetSampler.dissolution_edges)
    # test_newNetSampler.run(50000)
    # test_newNetSampler.show_traceplot()