import numpy as np
import matplotlib.pyplot as plt

from network import UndirectedNetwork, DirectedNetwork
from network_sampler import NetworkSampler
from BSTERGM import BSTERGM
from BSTERGM_diagnosis import BSTERGM_posterior_work, BSTERGM_latest_exchangeSampler_work
from BSTERGM_GOF import BSTERGM_GOF, dissociate_network

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
    reader_inst.read_from_csv("samplk_results/samplk_6chain", 2, 2)

    # print(reader_inst.MC_formation_samples[0:10])
    reader_inst.MC_dissolution_samples = reader_inst.MC_dissolution_samples[10000::10]
    reader_inst.MC_formation_samples = reader_inst.MC_formation_samples[10000::10]
    # print(np.mean(reader_inst.MC_sample_trace()[0][0]), np.mean(reader_inst.MC_sample_trace()[0][1]),
    #     np.mean(reader_inst.MC_sample_trace()[1][0]), np.mean(reader_inst.MC_sample_trace()[1][1]))
    reader_inst.show_traceplot()
    reader_inst.show_histogram()
    reader_inst.show_acfplot()
    
    netstat_reader_inst = BSTERGM_latest_exchangeSampler_work()
    netstat_reader_inst.read_from_csv("samplk_results/samplk_6chain_NetworkStat")
    netstat_reader_inst.show_traceplot()
    
    def model_netStat(network):
        model = []
        #define model
        model.append(network.statCal_edgeNum())
        model.append(network.statCal_geoWeightedESP(0.5))
        return np.array(model)

    def gof_additional_netStat(network):
        gof_netstat = []
        gof_netstat.append(network.statCal_edgeNum())
        gof_netstat.append(network.statCal_geoWeightedESP(0.5))
        return np.array(gof_netstat)

    #BSTERGM_GOF
    # def __init__(self, model_function, posterior_parameter_samples, init_network, additional_netstat_function=None , rng_seed=2021):
    
    #first lag
    net_plus1, net_minus1 = dissociate_network(samplk_sequence[0], samplk_sequence[1])

    gof_inst1f = BSTERGM_GOF(model_netStat, reader_inst.MC_formation_samples, samplk_sequence[0], gof_additional_netStat)
    gof_inst1f.gof_run(num_sim=500, exchange_iter=60)
    gof_inst1f.show_boxplot(next_net=net_plus1)

    gof_inst1d = BSTERGM_GOF(model_netStat, reader_inst.MC_dissolution_samples, samplk_sequence[0], gof_additional_netStat)
    gof_inst1d.gof_run(num_sim=500, exchange_iter=60)
    gof_inst1d.show_boxplot(next_net=net_minus1)

    #second lag
    net_plus2, net_minus2 = dissociate_network(samplk_sequence[1], samplk_sequence[2])

    gof_inst2f = BSTERGM_GOF(model_netStat, reader_inst.MC_formation_samples, samplk_sequence[1], gof_additional_netStat)
    gof_inst2f.gof_run(num_sim=500, exchange_iter=60)
    gof_inst2f.show_boxplot(next_net=net_plus2)

    gof_inst2d = BSTERGM_GOF(model_netStat, reader_inst.MC_dissolution_samples, samplk_sequence[1], gof_additional_netStat)
    gof_inst2d.gof_run(num_sim=500, exchange_iter=60)
    gof_inst2d.show_boxplot(next_net=net_minus2)

