import numpy as np
import matplotlib.pyplot as plt

from network import UndirectedNetwork, DirectedNetwork
from network_sampler import NetworkSampler
from BSTERGM import BSTERGM

class BSTERGM_GOF:
    def __init__(self, model_function, GOFplot_netstat_function, init_network, posterior_formation, posterior_dissolution, rng_seed=2021):
        self.model = model_function
        self.netstats = GOFplot_netstat_function
        self.formation_samples = posterior_formation
        self.dissolution_samples = posterior_dissolution
        self.init_network = init_network

        self.random_seed = rng_seed
        self.random_gen = np.random.default_rng(seed=rng_seed)
        self.netstat_list = []
        self.isDirected = False
        if isinstance(init_network, DirectedNetwork):
            self.isDirected = True
        else:
            self.isDirected = False

    def choose_samples(self):
        num_posterior_sample = len(self.formation_samples)
        sample_index = self.random_gen.integers(num_posterior_sample)
        return self.formation_samples[sample_index], self.dissolution_samples[sample_index]
    
    def get_next_network(self, last_net, formation_net, dissolution_net):
        dissolution_net_structure = dissolution_net.structure
        y_last_minus_ydis_structure = last_net.structure.copy()
        y_now_structure = formation_net.structure.copy()
        node_num = last_net.node_num

        for row in range(node_num):
            for col in range(node_num):
                if dissolution_net_structure[row,col]==1:
                    y_last_minus_ydis_structure[row,col]=0
        
        for row in range(node_num):
            for col in range(node_num):
                if y_last_minus_ydis_structure[row,col]==1:
                    y_now_structure[row,col] = 0
        
        result_network = 0
        if self.isDirected:
            result_network = DirectedNetwork(y_now_structure)
        else:
            result_network = UndirectedNetwork(y_now_structure)
        return result_network


    def gof_sampler(self, formation_parameter, dissolution_parameter, exchange_iter, rng_seed):
        Net_formation_sampler = NetworkSampler(self.model, formation_parameter, self.init_network, rng_seed)
        Net_dissolution_sampler = NetworkSampler(self.model, dissolution_parameter, self.init_network, rng_seed)
        
        Net_formation_sampler.run(exchange_iter)
        Net_dissolution_sampler.run(exchange_iter)

        return (Net_formation_sampler.network_samples[-1], Net_dissolution_sampler.network_samples[-1])

    def gof_run(self, num_sim, exchange_iter):
        for i in range(num_sim):
            if i%10==9:
                print("gof iter: ",i+1)
            formation_parameter, dissolution_parameter = self.choose_samples()
            formation_net, dissolution_net = self.gof_sampler(formation_parameter, dissolution_parameter, exchange_iter, self.random_seed + i)
            gof_sample_network = self.get_next_network(self.init_network, formation_net, dissolution_net)
            self.netstat_list.append(self.netstats(gof_sample_network))
    
    def netstat_trace(self):
        netstat_array = np.array(self.netstat_list)
        return netstat_array.T.tolist()


    def show_boxplot(self, next_net=None, show=True):
        netstats = self.netstat_trace()
        plt.boxplot(netstats)

        if next_net is not None:
            line_val = self.netstats(next_net)
            plt.plot([i+1 for i in range(len(line_val))], line_val, '-b')

        if show:
            plt.show()






if __name__ == "__main__":


    
    test_structure1 = np.array(
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
    test_structure2 = np.array(
    [
        [0,1,1,0,0,0,0,0,0,0],
        [1,0,1,1,0,0,0,0,0,0],
        [1,1,0,1,0,0,0,0,0,0],
        [0,1,1,0,1,1,0,0,0,0],
        [0,0,0,1,0,0,0,0,0,0],
        [0,0,0,1,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,1,1,0],
        [0,0,0,0,0,0,1,0,0,1],
        [0,0,0,0,0,0,1,0,0,1],
        [0,0,0,0,0,0,0,1,1,0]
    ]
    )
    test_structure3 = np.array(
    [
        [0,0,1,0,0,0,0,0,1,1],
        [0,0,1,1,0,0,0,0,1,0],
        [1,1,0,0,0,0,0,1,0,0],
        [0,1,0,0,1,1,0,0,0,0],
        [0,0,0,1,0,0,0,0,0,0],
        [0,0,0,1,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,1,1,0],
        [0,0,1,0,0,0,1,0,0,1],
        [1,1,0,0,0,0,1,0,0,1],
        [1,0,0,0,0,0,0,1,1,0]
    ]
    )

    test_initnet1 = UndirectedNetwork(test_structure1)
    test_initnet2 = UndirectedNetwork(test_structure2)
    test_initnet3 = UndirectedNetwork(test_structure3)
    test_obs_seq = [test_initnet1, test_initnet2, test_initnet3]


    def model_netStat(network):
        model = []
        #define model
        model.append(network.statCal_edgeNum())
        model.append(network.statCal_geoWeightedESP(0.5))
        return np.array(model)

    def gof_netStat(network):
        gof_netstat = []
        node_degree_dist = network.statCal_nodeInDegreeDist()
        ESP_dist = network.statCal_EdgewiseSharedPartnerDist()

        for val in node_degree_dist:
            gof_netstat.append(val)
        for val in ESP_dist:
            gof_netstat.append(val)
        return np.array(gof_netstat)


    initial_formation_param = np.array([0.1, 0.1])
    initial_dissolution_param = np.array([0.1, 0.1])
    test_BSTERGM_sampler = BSTERGM(model_netStat, initial_formation_param, initial_dissolution_param, test_obs_seq, 2021)
    test_BSTERGM_sampler.run(3000, exchange_iter=50)


    gof_inst1 = BSTERGM_GOF(model_netStat, gof_netStat, test_initnet1, 
        test_BSTERGM_sampler.MC_formation_samples[-2000::10], test_BSTERGM_sampler.MC_dissolution_samples[-2000::10])
    gof_inst1.gof_run(num_sim=100, exchange_iter=200)
    gof_inst1.show_boxplot(next_net=test_initnet2)


    gof_inst2 = BSTERGM_GOF(model_netStat, gof_netStat, test_initnet2, 
        test_BSTERGM_sampler.MC_formation_samples[-2000::10], test_BSTERGM_sampler.MC_dissolution_samples[-2000::10])
    gof_inst2.gof_run(num_sim=100, exchange_iter=200)
    gof_inst2.show_boxplot(next_net=test_initnet3)