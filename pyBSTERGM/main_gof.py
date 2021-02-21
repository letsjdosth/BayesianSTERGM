import numpy as np
import matplotlib.pyplot as plt

from network import UndirectedNetwork, DirectedNetwork
from network_sampler import NetworkSampler
from BSTERGM import BSTERGM
from BSTERGM_diagnosis import BSTERGM_posterior_work, BSTERGM_latest_exchangeSampler_work
from BSTERGM_GOF import BSTERGM_GOF, dissociate_network

import data_samplk
import data_knecht_friendship
import data_tailor

if __name__ == "__main__":
    samplk_sequence = [
        DirectedNetwork(np.array(data_samplk.samplk1)),
        DirectedNetwork(np.array(data_samplk.samplk2)),
        DirectedNetwork(np.array(data_samplk.samplk3))
    ]

    friendship_sequence = [
        DirectedNetwork(np.array(data_knecht_friendship.friendship_t1)),
        DirectedNetwork(np.array(data_knecht_friendship.friendship_t2)),
        DirectedNetwork(np.array(data_knecht_friendship.friendship_t3)),
        DirectedNetwork(np.array(data_knecht_friendship.friendship_t4))
    ]

    sociational_interactions = [
        UndirectedNetwork(np.array(data_tailor.KAPFTS1)),
        UndirectedNetwork(np.array(data_tailor.KAPFTS2))
    ]


    reader_inst = BSTERGM_posterior_work()
    reader_inst.read_from_csv("tailorSoc_results/tailorSoc_edgeGWESPl2_model_0chain", 2, 2)

    # print(reader_inst.MC_formation_samples[0:10])
    reader_inst.MC_dissolution_samples = reader_inst.MC_dissolution_samples[5000::10]
    reader_inst.MC_formation_samples = reader_inst.MC_formation_samples[5000::10]
    # print(np.mean(reader_inst.MC_sample_trace()[0][0]), np.mean(reader_inst.MC_sample_trace()[0][1]),
    #     np.mean(reader_inst.MC_sample_trace()[1][0]), np.mean(reader_inst.MC_sample_trace()[1][1]))
    reader_inst.show_traceplot()
    reader_inst.show_histogram()
    reader_inst.show_acfplot()
    
    # netstat_reader_inst = BSTERGM_latest_exchangeSampler_work()
    # netstat_reader_inst.read_from_csv("friendship_KH_example_model/friendship_sequence_Exmodel_run_0chain_NetworkStat")
    # netstat_reader_inst.show_traceplot()
    
    # def model_netStat(network):
    #     model = []
    #     #define model
    #     model.append(network.statCal_edgeNum())
    #     model.append(network.statCal_homophily(data_knecht_friendship.friendship_sex_girl_index)) #girls
    #     model.append(network.statCal_homophily(data_knecht_friendship.friendship_sex_boy_index)) #boys
    #     model.append(network.statCal_heterophily(data_knecht_friendship.friendship_sex_girl_index, data_knecht_friendship.friendship_sex_boy_index))#girls->boys
    #     model.append(network.statCal_match_matrix(np.array(data_knecht_friendship.friendship_primary)))
    #     model.append(network.statCal_mutuality())
    #     model.append(network.statCal_transitiveTies())
    #     model.append(network.statCal_cyclicalTies())
        
    #     return np.array(model)

        
    def model_netStat_samplk_vignettesEx(network): #directed
        model = []
        #define model
        model.append(network.statCal_edgeNum())
        model.append(network.statCal_mutuality())
        model.append(network.statCal_cyclicTriples())
        model.append(network.statCal_transitiveTriples())
        return np.array(model)

    def model_netStat_friendship_KHEx(network): #directed
        model = []
        #define model
        model.append(network.statCal_edgeNum())
        model.append(network.statCal_homophily(data_knecht_friendship.friendship_sex_girl_index)) #girls
        model.append(network.statCal_homophily(data_knecht_friendship.friendship_sex_boy_index)) #boys
        model.append(network.statCal_heterophily(data_knecht_friendship.friendship_sex_girl_index, data_knecht_friendship.friendship_sex_boy_index))#girls->boys
        model.append(network.statCal_match_matrix(np.array(data_knecht_friendship.friendship_primary)))
        model.append(network.statCal_mutuality())
        model.append(network.statCal_transitiveTies())
        model.append(network.statCal_cyclicalTies())
        return np.array(model)

    def model_netStat_friendship_KH_simplified(network): #directed
        model = []
        model.append(network.statCal_edgeNum())
        model.append(network.statCal_heterophily(data_knecht_friendship.friendship_sex_girl_index, data_knecht_friendship.friendship_sex_boy_index))#girls->boys
        model.append(network.statCal_match_matrix(np.array(data_knecht_friendship.friendship_primary)))
        model.append(network.statCal_mutuality())
        model.append(network.statCal_transitiveTies())
        model.append(network.statCal_cyclicalTies())
        return np.array(model)

    def model_netStat_tailorshop_social(network): #undirected
        model = []
        model.append(network.statCal_edgeNum())
        model.append(network.statCal_geoWeightedESP(tau=np.log(2)))
        return np.array(model)

    def gof_additional_netStat(network): 
        model = []
        #define model
        model.append(network.statCal_edgeNum())
        model.append(network.statCal_geoWeightedESP(np.log(2)))
        
        return np.array(model)

    # def gof_additional_netStat(network):
    #     gof_netstat = []
    #     gof_netstat.append(network.statCal_homophily(data_knecht_friendship.friendship_sex_girl_index)) #girls
    #     gof_netstat.append(network.statCal_homophily(data_knecht_friendship.friendship_sex_boy_index)) #boys
    #     gof_netstat.append(network.statCal_heterophily(data_knecht_friendship.friendship_sex_girl_index, data_knecht_friendship.friendship_sex_boy_index))#girls->boys
    #     gof_netstat.append(network.statCal_match_matrix(np.array(data_knecht_friendship.friendship_primary)))
    #     return np.array(gof_netstat)

        

    # friendship============================================================================================
    # #first lag
    # net_plus1, net_minus1 = dissociate_network(friendship_sequence[0], friendship_sequence[1])

    # gof_inst1f = BSTERGM_GOF(model_netStat, reader_inst.MC_formation_samples, friendship_sequence[0], gof_additional_netStat)
    # gof_inst1f.gof_run(num_sim=200, exchange_iter=100)
    # gof_inst1f.show_boxplot(next_net=net_plus1)

    # gof_inst1d = BSTERGM_GOF(model_netStat, reader_inst.MC_dissolution_samples, friendship_sequence[0], gof_additional_netStat)
    # gof_inst1d.gof_run(num_sim=200, exchange_iter=100)
    # gof_inst1d.show_boxplot(next_net=net_minus1)

    # #second lag
    # net_plus2, net_minus2 = dissociate_network(friendship_sequence[1], friendship_sequence[2])

    # gof_inst2f = BSTERGM_GOF(model_netStat, reader_inst.MC_formation_samples, friendship_sequence[1], gof_additional_netStat)
    # gof_inst2f.gof_run(num_sim=200, exchange_iter=30)
    # gof_inst2f.show_boxplot(next_net=net_plus2)

    # gof_inst2d = BSTERGM_GOF(model_netStat, reader_inst.MC_dissolution_samples, friendship_sequence[1], gof_additional_netStat)
    # gof_inst2d.gof_run(num_sim=200, exchange_iter=30)
    # gof_inst2d.show_boxplot(next_net=net_minus2)


    # #third lag
    # net_plus3, net_minus3 = dissociate_network(friendship_sequence[2], friendship_sequence[3])

    # gof_inst3f = BSTERGM_GOF(model_netStat, reader_inst.MC_formation_samples, friendship_sequence[2], gof_additional_netStat)
    # gof_inst3f.gof_run(num_sim=200, exchange_iter=30)
    # gof_inst3f.show_boxplot(next_net=net_plus2)

    # gof_inst3d = BSTERGM_GOF(model_netStat, reader_inst.MC_dissolution_samples, friendship_sequence[2], gof_additional_netStat)
    # gof_inst3d.gof_run(num_sim=200, exchange_iter=30)
    # gof_inst3d.show_boxplot(next_net=net_minus2)


    #tailorshop =========================================================================
    #BSTERGM_GOF
    # def __init__(self, model_function, posterior_parameter_samples, init_network, additional_netstat_function=None , rng_seed=2021):
    net_plus, net_minus = dissociate_network(sociational_interactions[0], sociational_interactions[1])
    gof_inst2f = BSTERGM_GOF(model_netStat_tailorshop_social, reader_inst.MC_formation_samples, sociational_interactions[0], gof_additional_netStat)
    gof_inst2f.gof_run(num_sim=200, exchange_iter=10)
    gof_inst2f.show_boxplot(next_net=net_plus)

    gof_inst2d = BSTERGM_GOF(model_netStat_tailorshop_social, reader_inst.MC_dissolution_samples, sociational_interactions[0], gof_additional_netStat)
    gof_inst2d.gof_run(num_sim=200, exchange_iter=10)
    gof_inst2d.show_boxplot(next_net=net_minus)

