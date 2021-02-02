import numpy as np
import matplotlib.pyplot as plt

from network import UndirectedNetwork, DirectedNetwork
from network_sampler import NetworkSampler
from BSTERGM import BSTERGM

def dissociate_network(last_network, now_network):
    y_last_structure = last_network.structure
    y_now_structure = now_network.structure
    y_plus = y_last_structure.copy()
    y_minus = y_last_structure.copy()
    for row in range(last_network.node_num):
        for col in range(last_network.node_num):
            if y_now_structure[row,col]==1:
                y_plus[row,col]=1
            if y_now_structure[row,col]==0:
                y_minus[row,col]=0
    result = 0
    if isinstance(last_network,DirectedNetwork):
        result = (DirectedNetwork(y_plus), DirectedNetwork(y_minus))
    else:
        result = (UndirectedNetwork(y_plus), UndirectedNetwork(y_minus))
    return result

class BSTERGM_GOF:
    def __init__(self, model_function, posterior_parameter_samples, init_network, additional_netstat_function=None , rng_seed=2021):
        self.model = model_function
        self.additional_netstats = additional_netstat_function
        self.posterior_parameter_samples = posterior_parameter_samples
        self.init_network = init_network

        self.random_seed = rng_seed
        self.random_gen = np.random.default_rng(seed=rng_seed)
        self.additional_netstat_list = []
        self.isDirected = False
        if isinstance(init_network, DirectedNetwork):
            self.isDirected = True
            self.sample_nodeInDegreeDist = []
            self.sample_nodeOutDegreeDist = []
            self.sample_ESPDist = []
            self.sample_minGeodesicDist = []
            self.sample_modelStats = []

        else:
            self.isDirected = False
            self.sample_nodeDegreeDist = []
            self.sample_ESPDist = []
            self.sample_minGeodesicDist = []


    def choose_samples(self):
        num_posterior_sample = len(self.posterior_parameter_samples)
        sample_index = self.random_gen.integers(num_posterior_sample)
        return self.posterior_parameter_samples[sample_index]
    
    # def get_next_network(self, last_net, formation_net, dissolution_net):
    #     dissolution_net_structure = dissolution_net.structure
    #     y_last_minus_ydis_structure = last_net.structure.copy()
    #     y_now_structure = formation_net.structure.copy()
    #     node_num = last_net.node_num

    #     for row in range(node_num):
    #         for col in range(node_num):
    #             if dissolution_net_structure[row,col]==1:
    #                 y_last_minus_ydis_structure[row,col]=0
        
    #     for row in range(node_num):
    #         for col in range(node_num):
    #             if y_last_minus_ydis_structure[row,col]==1:
    #                 y_now_structure[row,col] = 0
        
    #     result_network = 0
    #     if self.isDirected:
    #         result_network = DirectedNetwork(y_now_structure)
    #     else:
    #         result_network = UndirectedNetwork(y_now_structure)
    #     return result_network


    def gof_sampler(self, parameter, exchange_iter, rng_seed):
        Net_sampler = NetworkSampler(self.model, parameter, self.init_network, rng_seed)
        Net_sampler.run(exchange_iter)

        return Net_sampler.network_samples[-1]


    def collect_netstat(self, network):
        self.additional_netstat_list.append(self.additional_netstats(network))
        if self.isDirected:
            self.sample_nodeInDegreeDist.append(network.statCal_nodeInDegreeDist())
            self.sample_nodeOutDegreeDist.append(network.statCal_nodeOutDegreeDist())
            self.sample_ESPDist.append(network.statCal_EdgewiseSharedPartnerDist())
            self.sample_minGeodesicDist.append(network.statCal_MinGeodesicDist())
            self.sample_modelStats.append(
                [network.statCal_edgeNum(), network.statCal_mutuality(),
                network.statCal_transitiveTies(), network.statCal_cyclicalTies()])
        else:
            self.sample_nodeDegreeDist.append(network.statCal_nodeDegreeDist())
            self.sample_ESPDist.append(network.statCal_EdgewiseSharedPartnerDist())
            self.sample_minGeodesicDist.append(network.statCal_MinGeodesicDist())



    def gof_run(self, num_sim, exchange_iter):
        for i in range(num_sim):
            if i%10==9:
                print("gof iter: ",i+1)
            parameter = self.choose_samples()
            gof_sample_network = self.gof_sampler(parameter, exchange_iter, self.random_seed + i)
            self.collect_netstat(gof_sample_network)
    
    def netstat_make_trace(self, sample_netstat_list):
        netstat_array = np.array(sample_netstat_list)
        return netstat_array.T.tolist()

    def each_netstat_trace_directed(self):
        return (
            self.netstat_make_trace(self.sample_nodeInDegreeDist),
            self.netstat_make_trace(self.sample_nodeOutDegreeDist),
            self.netstat_make_trace(self.sample_ESPDist),
            self.netstat_make_trace(self.sample_minGeodesicDist),
            self.netstat_make_trace(self.sample_modelStats),
            self.netstat_make_trace(self.additional_netstat_list)
        )

    def make_boxplot_directed(self, next_net=None):
        node_InDegree, node_OutDegree, ESP, minGeoDist, modelStats, additional = self.each_netstat_trace_directed()
        grid_column = 6
        grid_row = 1
        plt.figure(figsize=(3*grid_column, 5*grid_row))

        plt.subplot(grid_row, grid_column, 1)
        plt.boxplot(node_InDegree)
        if next_net is not None:
            line_val = next_net.statCal_nodeInDegreeDist()
            plt.plot([i+1 for i in range(len(line_val))], line_val, '-b')
            plt.xlabel("in degree")
            plt.xticks([i for i in range(1, len(line_val)+1)],[i for i in range(0, len(line_val))])


        plt.subplot(grid_row, grid_column, 2)
        plt.boxplot(node_OutDegree)
        if next_net is not None:
            line_val = next_net.statCal_nodeOutDegreeDist()
            plt.plot([i+1 for i in range(len(line_val))], line_val, '-b')
            plt.xlabel("out degree")
            plt.xticks([i for i in range(1, len(line_val)+1)],[i for i in range(0, len(line_val))])

        plt.subplot(grid_row, grid_column, 3)
        plt.boxplot(ESP)
        if next_net is not None:
            line_val = next_net.statCal_EdgewiseSharedPartnerDist()
            plt.plot([i+1 for i in range(len(line_val))], line_val, '-b')
            plt.xlabel("edge-wise shared partners")
            plt.xticks([i for i in range(1, len(line_val)+1)],[i for i in range(0, len(line_val))])
        
        plt.subplot(grid_row, grid_column, 4)
        plt.boxplot(minGeoDist)
        if next_net is not None:
            line_val = next_net.statCal_MinGeodesicDist()
            plt.plot([i+1 for i in range(len(line_val))], line_val, '-b')
            plt.xlabel("minimum geodesic distance")
            plt.xticks([i for i in range(1, len(line_val)+1)],[i for i in range(1, len(line_val))]+['NR'])

        plt.subplot(grid_row, grid_column, 5)
        plt.boxplot(modelStats)
        if next_net is not None:
            line_val = [next_net.statCal_edgeNum(), next_net.statCal_mutuality(),
                next_net.statCal_transitiveTies(), next_net.statCal_cyclicalTies()]
            plt.plot([i+1 for i in range(len(line_val))], line_val, '-b')
            plt.xlabel("model statistics")
            plt.xticks([i for i in range(1, len(line_val)+1)],['edge','mutual','transitiveties','cyclicalties'])

        plt.subplot(grid_row, grid_column, 6)
        plt.boxplot(additional)
        if next_net is not None:
            line_val = self.additional_netstats(next_net)
            plt.plot([i+1 for i in range(len(line_val))], line_val, '-b')
            plt.xlabel("additional statistics")


    
    def each_netstat_trace_undirected(self):
        return (
            self.netstat_make_trace(self.sample_nodeDegreeDist),
            self.netstat_make_trace(self.sample_ESPDist),
            self.netstat_make_trace(self.sample_minGeodesicDist),
            self.netstat_make_trace(self.additional_netstat_list)
        )


    def make_boxplot_undirected(self, next_net=None):
        node_Degree, ESP, minGeoDist, additional = self.each_netstat_trace_undirected()
        grid_column = 4
        grid_row = 1
        plt.figure(figsize=(3*grid_column, 5*grid_row))

        plt.subplot(grid_row, grid_column, 1)
        plt.boxplot(node_Degree)
        if next_net is not None:
            line_val = next_net.statCal_nodeDegreeDist()
            plt.plot([i+1 for i in range(len(line_val))], line_val, '-b')
            plt.xlabel("degree")
            plt.xticks([i for i in range(1, len(line_val)+1)],[i for i in range(0, len(line_val))])

        plt.subplot(grid_row, grid_column, 2)
        plt.boxplot(ESP)
        if next_net is not None:
            line_val = next_net.statCal_EdgewiseSharedPartnerDist()
            plt.plot([i+1 for i in range(len(line_val))], line_val, '-b')
            plt.xlabel("edge-wise shared partners")
            plt.xticks([i for i in range(1, len(line_val)+1)],[i for i in range(0, len(line_val))])
        
        plt.subplot(grid_row, grid_column, 3)
        plt.boxplot(minGeoDist)
        if next_net is not None:
            line_val = next_net.statCal_MinGeodesicDist()
            plt.plot([i+1 for i in range(len(line_val))], line_val, '-b')
            plt.xlabel("minimum geodesic distance")
            plt.xticks([i for i in range(1, len(line_val)+1)],[i for i in range(1, len(line_val))]+['NR'])

        plt.subplot(grid_row, grid_column, 4)
        plt.boxplot(additional)
        if next_net is not None:
            line_val = self.additional_netstats(next_net)
            plt.plot([i+1 for i in range(len(line_val))], line_val, '-b')
            plt.xlabel("additional statistics")


    def show_boxplot(self, next_net=None, show=True):
        if self.isDirected:
            self.make_boxplot_directed(next_net=next_net)        
        else:
            self.make_boxplot_undirected(next_net=next_net)

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

    test_initnet1 = DirectedNetwork(test_structure1)
    test_initnet2 = DirectedNetwork(test_structure2)
    test_initnet3 = DirectedNetwork(test_structure3)
    test_obs_seq = [test_initnet1, test_initnet2, test_initnet3]


    def model_netStat(network):
        model = []
        #define model
        model.append(network.statCal_edgeNum())
        model.append(network.statCal_geoWeightedESP(0.5))
        return np.array(model)

    def gof_additional_netStat(network):
        gof_netstat = []
        gof_netstat.append(network.statCal_edgeNum())
        return np.array(gof_netstat)


    initial_formation_param = np.array([0.1, 0.1])
    initial_dissolution_param = np.array([0.1, 0.1])
    test_BSTERGM_sampler = BSTERGM(model_netStat, initial_formation_param, initial_dissolution_param, test_obs_seq, 2021)
    test_BSTERGM_sampler.run(3000, exchange_iter=10)


    # def __init__(self, model_function, posterior_parameter_samples, init_network, additional_netstat_function=None , rng_seed=2021):
    #first lag
    yplus2_net, yminus2_net = dissociate_network(test_initnet1, test_initnet2)

    gof_inst11 = BSTERGM_GOF(model_netStat, test_BSTERGM_sampler.MC_formation_samples[-2000::10], 
        test_initnet1, additional_netstat_function=gof_additional_netStat)
    gof_inst11.gof_run(num_sim=100, exchange_iter=50)
    gof_inst11.show_boxplot(next_net=yplus2_net)

    gof_inst12 = BSTERGM_GOF(model_netStat, test_BSTERGM_sampler.MC_dissolution_samples[-2000::10], 
        test_initnet1, additional_netstat_function=gof_additional_netStat)
    gof_inst12.gof_run(num_sim=100, exchange_iter=50)
    gof_inst12.show_boxplot(next_net=yminus2_net)

    #second lag
    yplus3_net, yminus3_net = dissociate_network(test_initnet2, test_initnet3)
    
    gof_inst11 = BSTERGM_GOF(model_netStat, test_BSTERGM_sampler.MC_formation_samples[-2000::10], 
        test_initnet2, additional_netstat_function=gof_additional_netStat)
    gof_inst11.gof_run(num_sim=100, exchange_iter=50)
    gof_inst11.show_boxplot(next_net=yplus3_net)

    gof_inst12 = BSTERGM_GOF(model_netStat, test_BSTERGM_sampler.MC_dissolution_samples[-2000::10], 
        test_initnet2, additional_netstat_function=gof_additional_netStat)
    gof_inst12.gof_run(num_sim=100, exchange_iter=50)
    gof_inst12.show_boxplot(next_net=yminus3_net)


