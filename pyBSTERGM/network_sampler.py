import numpy as np
import matplotlib.pyplot as plt

from network import UndirectedNetwork, DirectedNetwork

class NetworkSampler:
    def __init__(self, model_fn, param, init_network, is_formation=None, constraint_net=None, num_joint_blocks=0, rng_seed=2021):
        # containers
        self.network_samples = []
        self.network_samples_netStats = []
        self.mutable_edges = []

        #error
        if is_formation is not None and constraint_net is None:
            raise TypeError("set 'constraint_net'.") #TypeError?
        if constraint_net is not None and is_formation is None:
            raise TypeError("set 'is_formation'.")

        #initialize
        self.param = param
        self.initial_network = init_network
        self.node_num = init_network.node_num
        self.model = model_fn
        self.random_gen = np.random.default_rng(seed=rng_seed)
        self.is_formation = is_formation
        self.constraint_net = constraint_net
        if num_joint_blocks == 0 or num_joint_blocks == 1:
            self.make_mutable_edges_list(self.is_formation, self.constraint_net)
        else:
            self.make_mutable_edges_list_block_diag(self.is_formation, self.constraint_net, num_joint_blocks)

        self.network_samples.append(init_network)
        self.network_samples_netStats.append(self.model(self.initial_network))

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

    def make_mutable_edges_list_block_diag(self, is_formation, constraint_net, num_joint_blocks):
        block_dim = self.node_num//num_joint_blocks
        if is_formation is None: #no constraint
            for left_up in range(0, self.node_num, block_dim):
                print(left_up)
                for i in range(left_up, left_up+block_dim):
                    for j in range(left_up, left_up+block_dim):
                        if i==j:
                            pass
                        else:
                            self.mutable_edges.append((i,j))
        
        elif is_formation:
            constraint_net_structure = self.constraint_net.structure
            # we can change 0s
            for left_up in range(0, self.node_num, block_dim):
                for i in range(left_up, left_up+block_dim):
                    for j in range(left_up, left_up+block_dim):
                        if i==j:
                            pass
                        elif constraint_net_structure[i][j]==0:
                            self.mutable_edges.append((i,j))

        else: #dissolution
            constraint_net_structure = self.constraint_net.structure
            # we can change 1s
            for left_up in range(0, self.node_num, block_dim):
                for i in range(left_up, left_up+block_dim):
                    for j in range(left_up, left_up+block_dim):
                        if i==j:
                            pass
                        elif constraint_net_structure[i][j]==1:
                            self.mutable_edges.append((i,j))

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

    def log_r(self, proposed_network_netStat):
        last_network_netStat = self.network_samples_netStats[-1]
        return np.dot(proposed_network_netStat - last_network_netStat, self.param)

    def sampler(self):
        last_network = self.network_samples[-1]
        proposed_network = self.propose_network(last_network)
        proposed_network_netStat = self.model(proposed_network)

        log_r_val = self.log_r(proposed_network_netStat)
        unif_sample = self.random_gen.random()
        if np.log(unif_sample) < log_r_val:
            self.network_samples.append(proposed_network)
            self.network_samples_netStats.append(proposed_network_netStat)
        else:
            self.network_samples.append(last_network)
            self.network_samples_netStats.append(self.network_samples_netStats[-1])

    def run(self, iter):
        for _ in range(iter):
            self.sampler()

    def netStat_trace(self):
        netStat_trace_result = []
        for _ in range(len(self.param)):
            netStat_trace_result.append([])

        for netStat in self.network_samples_netStats:
            for i, val in enumerate(netStat):
                netStat_trace_result[i].append(val)

        return netStat_trace_result

    def show_traceplot(self, show=True):
        netStat = self.netStat_trace()
        grid_column = 1
        grid_row = int(len(netStat))
        
        constraint_netStat = None
        if self.constraint_net is not None:
            constraint_netStat = self.model(self.constraint_net)

        plt.figure(figsize=(5*grid_column, 3*grid_row))
        for i, statSeq in enumerate(netStat):
            plt.subplot(grid_row, grid_column, i+1)
            plt.plot(range(len(statSeq)), statSeq)
            if self.constraint_net is not None:
                plt.axhline(constraint_netStat[i], color='red', linewidth=1.5)

        if show:
            plt.show()


if __name__ == "__main__":

    def model_netStat(network : UndirectedNetwork):
        model = []
        #define model
        model.append(network.statCal_edgeNum())
        model.append(network.statCal_geoWeightedESP(0.3))
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

    test_constraint_structure = np.array(
        [
            [0,1,0,0,0,0,0,0,0,0],
            [1,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,1],
            [0,0,0,0,0,0,0,0,1,0],
        ]
    )
    test_constraint_net = UndirectedNetwork(test_constraint_structure)
    #1
    
    test_netSampler = NetworkSampler(model_netStat, np.array([0, 0]), test_initnet, is_formation=False, constraint_net=test_constraint_net, num_joint_blocks=2)
    # test_netSampler.make_mutable_edges_list_block_diag(None,None,num_joint_blocks=2)
    print(test_netSampler.mutable_edges)
    # print(test_netSampler.choose_mutable_edge())
    

    # test_netSampler.run(500)
    # test_netSampler.show_traceplot()


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