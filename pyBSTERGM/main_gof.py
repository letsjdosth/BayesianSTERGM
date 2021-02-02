import numpy as np
import matplotlib.pyplot as plt

from network import UndirectedNetwork, DirectedNetwork
from network_sampler import NetworkSampler
from BSTERGM import BSTERGM
from BSTERGM_diagnosis import BSTERGM_posterior_work
from BSTERGM_GOF import BSTERGM_GOF

import data_samplk
import data_knecht_friendship

if __name__ == "__main__":
    samplk_sequence = [
        DirectedNetwork(np.array(data_samplk.samplk1)),
        DirectedNetwork(np.array(data_samplk.samplk2)),
        DirectedNetwork(np.array(data_samplk.samplk3))
    ]

    friendship_sequence = [
        DirectedNetwork(np.array(data_knecht_friendship.friendship_t1)), #node=26
        DirectedNetwork(np.array(data_knecht_friendship.friendship_t2)), #26
        DirectedNetwork(np.array(data_knecht_friendship.friendship_t3)), #node=25 (when R controls NAs...)
        DirectedNetwork(np.array(data_knecht_friendship.friendship_t4)) #25 #<-1/2에서 빠진줄을 빼면 될 듯
    ]

    reader_inst = BSTERGM_posterior_work()
    reader_inst.read_from_csv("samplk_results/samplk_1chain", 2, 2)
    # print(reader_inst.MC_formation_samples[0:10])
    reader_inst.MC_dissolution_samples = reader_inst.MC_dissolution_samples[10000::10]
    reader_inst.MC_formation_samples = reader_inst.MC_formation_samples[10000::10]
    # print(np.mean(reader_inst.MC_sample_trace()[0][0]), np.mean(reader_inst.MC_sample_trace()[0][1]),
    #     np.mean(reader_inst.MC_sample_trace()[1][0]), np.mean(reader_inst.MC_sample_trace()[1][1]))
    # reader_inst.show_traceplot()
    # reader_inst.show_histogram()
    # reader_inst.show_acfplot()


    
    def model_netStat(network):
        model = []
        #define model
        model.append(network.statCal_edgeNum())
        model.append(network.statCal_geoWeightedESP(0.5))
        return np.array(model)

    def gof_netStat(network):
        gof_netstat = []
        node_InDegree_dist = network.statCal_nodeInDegreeDist()
        node_OutDegree_dist = network.statCal_nodeOutDegreeDist()
        ESP_dist = network.statCal_EdgewiseSharedPartnerDist()

        for val in node_InDegree_dist:
            gof_netstat.append(val)
        for val in node_OutDegree_dist:
            gof_netstat.append(val)
        for val in ESP_dist:
            gof_netstat.append(val)
        return np.array(gof_netstat)


    gof_inst1 = BSTERGM_GOF(model_netStat, gof_netStat, samplk_sequence[0], 
        reader_inst.MC_formation_samples, reader_inst.MC_dissolution_samples)
    gof_inst1.gof_run(num_sim=500, exchange_iter=50)
    gof_inst1.show_boxplot(next_net=samplk_sequence[1])


    gof_inst1 = BSTERGM_GOF(model_netStat, gof_netStat, samplk_sequence[1], 
        reader_inst.MC_formation_samples, reader_inst.MC_dissolution_samples)
    gof_inst1.gof_run(num_sim=500, exchange_iter=50)
    gof_inst1.show_boxplot(next_net=samplk_sequence[2])


    # gof_inst1 = BSTERGM_GOF(model_netStat, gof_netStat, friendship_sequence[0], 
    #     reader_inst.MC_formation_samples, reader_inst.MC_dissolution_samples)
    # gof_inst1.gof_run(num_sim=500, exchange_iter=50)
    # gof_inst1.show_boxplot(next_net=friendship_sequence[1])

    
    # gof_inst2 = BSTERGM_GOF(model_netStat, gof_netStat, friendship_sequence[1], 
    #     reader_inst.MC_formation_samples, reader_inst.MC_dissolution_samples)
    # gof_inst2.gof_run(num_sim=500, exchange_iter=50)
    # gof_inst2.show_boxplot(next_net=friendship_sequence[2])

    # gof_inst3 = BSTERGM_GOF(model_netStat, gof_netStat, friendship_sequence[2], 
    #     reader_inst.MC_formation_samples, reader_inst.MC_dissolution_samples)
    # gof_inst3.gof_run(num_sim=500, exchange_iter=50)
    # gof_inst3.show_boxplot(next_net=friendship_sequence[3])