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

    #read / quick diagnosis

    # reader_inst = BSTERGM_posterior_work()
    # reader_inst.read_from_csv("tailorSoc_results/tailorSoc_edgeGWESPl2_model_0chain", 2, 2)

    # print(reader_inst.MC_formation_samples[0:10])
    # reader_inst.MC_dissolution_samples = reader_inst.MC_dissolution_samples[5000::10]
    # reader_inst.MC_formation_samples = reader_inst.MC_formation_samples[5000::10]
    # print(np.mean(reader_inst.MC_sample_trace()[0][0]), np.mean(reader_inst.MC_sample_trace()[0][1]),
    #     np.mean(reader_inst.MC_sample_trace()[1][0]), np.mean(reader_inst.MC_sample_trace()[1][1]))
    # reader_inst.show_traceplot()
    # reader_inst.show_histogram()
    # reader_inst.show_acfplot()
    
    # netstat_reader_inst = BSTERGM_latest_exchangeSampler_work()
    # netstat_reader_inst.read_from_csv("friendship_KH_example_model/friendship_sequence_Exmodel_run_0chain_NetworkStat")
    # netstat_reader_inst.show_traceplot()


    # gof
    from model_settings import model_netStat_edgeonly, edgeonly_initial_formation_vec, edgeonly_initial_dissolution_vec
    from model_settings import model_netStat_edgeGWdgre, edgeGWdgre_initial_formation_vec, edgeGWdgre_initial_dissolution_vec
    from model_settings import model_netStat_edgeGWESP, edgeGWESP_initial_formation_vec, edgeGWESP_initial_dissolution_vec
    from model_settings import model_netStat_edgeGWDSP, edgeGWDSP_initial_formation_vec, edgeGWDSP_initial_dissolution_vec
    
    # def gof_additional_netStat(network): 
    #     model = []
    #     #define model
    #     model.append(network.statCal_edgeNum())
    #     model.append(network.statCal_geoWeightedESP(np.log(2)))
        
    #     return np.array(model)

    # friendship============================================================================================
    #edge only/edge+gwesp switch
    # reader_inst = BSTERGM_posterior_work()
    # reader_inst.read_from_csv("example_results_bygroupProposal/friendship_bygroupsample_normPrior_edgeonly_1chain", 1, 1)
    # reader_inst.read_from_csv("example_results_pairProposal/friendship_pairsample_normPrior_edgeGWESP_0chain", 2, 2)
    # print(len(reader_inst.MC_dissolution_samples))
    # reader_inst.MC_dissolution_samples = reader_inst.MC_dissolution_samples[5000::10]
    # reader_inst.MC_formation_samples = reader_inst.MC_formation_samples[5000::10]


    #first lag
    # net_plus1, net_minus1 = dissociate_network(friendship_sequence[0], friendship_sequence[1])

    # gof_inst1f = BSTERGM_GOF(model_netStat_edgeGWESP, reader_inst.MC_formation_samples, friendship_sequence[0],
    #                         is_formation=True, additional_netstat_function=model_netStat_edgeGWESP)
    # gof_inst1f.gof_run(num_sim=200, exchange_iter=1500)
    # gof_inst1f.show_boxplot(next_net=net_plus1)

    # gof_inst1d = BSTERGM_GOF(model_netStat_edgeGWESP, reader_inst.MC_dissolution_samples, friendship_sequence[0], 
    #                         is_formation=False, additional_netstat_function=model_netStat_edgeGWESP)
    # gof_inst1d.gof_run(num_sim=200, exchange_iter=1500)
    # gof_inst1d.show_boxplot(next_net=net_minus1)

    # #second lag
    # net_plus2, net_minus2 = dissociate_network(friendship_sequence[1], friendship_sequence[2])

    # gof_inst2f = BSTERGM_GOF(model_netStat_edgeGWESP, reader_inst.MC_formation_samples, friendship_sequence[1], 
    #                         is_formation=True, additional_netstat_function=model_netStat_edgeGWESP)
    # gof_inst2f.gof_run(num_sim=200, exchange_iter=3000)
    # gof_inst2f.show_boxplot(next_net=net_plus2)

    # gof_inst2d = BSTERGM_GOF(model_netStat_edgeGWESP, reader_inst.MC_dissolution_samples, friendship_sequence[1],
    #                         is_formation=False, additional_netstat_function=model_netStat_edgeGWESP)
    # gof_inst2d.gof_run(num_sim=200, exchange_iter=3000)
    # gof_inst2d.show_boxplot(next_net=net_minus2)

    # #third lag
    # net_plus3, net_minus3 = dissociate_network(friendship_sequence[2], friendship_sequence[3])

    # gof_inst3f = BSTERGM_GOF(model_netStat_edgeGWESP, reader_inst.MC_formation_samples, friendship_sequence[2],
    #                         is_formation=True, additional_netstat_function=model_netStat_edgeGWESP)
    # gof_inst3f.gof_run(num_sim=200, exchange_iter=3000)
    # gof_inst3f.show_boxplot(next_net=net_plus3)

    # gof_inst3d = BSTERGM_GOF(model_netStat_edgeGWESP, reader_inst.MC_dissolution_samples, friendship_sequence[2],
    #                         is_formation=False, additional_netstat_function=model_netStat_edgeGWESP)
    # gof_inst3d.gof_run(num_sim=200, exchange_iter=3000)
    # gof_inst3d.show_boxplot(next_net=net_minus3)


    # #tailorshop =========================================================================
    
    #edge only/edge+gwesp switch
    reader_inst = BSTERGM_posterior_work()
    # reader_inst.read_from_csv("example_results_pairProposal/tailorshop_pairsample_normPrior_edgeonly_2chain", 1, 1)
    reader_inst.read_from_csv("example_results_pairProposal/tailorshop_pairsample_normPrior_edgeGWESP_1chain", 2, 2)
    # print(len(reader_inst.MC_dissolution_samples))
    reader_inst.MC_dissolution_samples = reader_inst.MC_dissolution_samples[5000::10]
    reader_inst.MC_formation_samples = reader_inst.MC_formation_samples[5000::10]
    print(np.mean(reader_inst.MC_sample_trace()[0][0]), np.mean(reader_inst.MC_sample_trace()[0][1]),
        np.mean(reader_inst.MC_sample_trace()[1][0]), np.mean(reader_inst.MC_sample_trace()[1][1]))    

    #bstergm: formation [-2.7086, 0.9838], dissolution [-2.7814, 1.0281]
    # formation_mark = [np.array([-2.5998, 0.9106])]
    # dissolution_mark=[np.array([-0.1921, 0.5155])]
    


    # net_plus, net_minus = dissociate_network(sociational_interactions[0], sociational_interactions[1])
    
    # gof_inst_f = BSTERGM_GOF(model_netStat_edgeGWESP, reader_inst.MC_formation_samples, sociational_interactions[0],
    #                         is_formation=True, additional_netstat_function=model_netStat_edgeGWESP)
    # gof_inst_f.gof_run(num_sim=200, exchange_iter=500)
    # gof_inst_f.show_boxplot(next_net=net_plus)

    # gof_inst_d = BSTERGM_GOF(model_netStat_edgeGWESP, dissolution_mark, sociational_interactions[0],
    #                         is_formation=False, additional_netstat_function=model_netStat_edgeGWESP)
    # gof_inst_d.gof_run(num_sim=200, exchange_iter=30)
    # gof_inst_d.show_boxplot(next_net=net_minus)
