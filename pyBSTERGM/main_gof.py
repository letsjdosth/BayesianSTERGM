import numpy as np
import matplotlib.pyplot as plt

from network import UndirectedNetwork, DirectedNetwork
from network_sampler import NetworkSampler
from BSTERGM import BSTERGM
from BSTERGM_diagnosis import BSTERGM_posterior_work, BSTERGM_latest_exchangeSampler_work
from BSTERGM_GOF import BSTERGM_GOF

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

    #samplk =======================================================================================================================
    from model_settings import model_netStat_samplk_vignettesEx, samplk_vignettesEx_initial_formation_vec, samplk_vignettesEx_initial_dissolution_vec
    
    #samplk: read / quick diagnosis
    reader_inst_samplk_vig = BSTERGM_posterior_work()
    reader_inst_samplk_vig.read_from_BSTERGM_csv("example_results_samplk/samplk_jointtimelag_normPrior_vignettesEx_4chain", 4, 4)
    reader_inst_samplk_vig.MC_formation_samples = reader_inst_samplk_vig.MC_formation_samples[10000::20]
    reader_inst_samplk_vig.MC_dissolution_samples = reader_inst_samplk_vig.MC_dissolution_samples[10000::20]
    # reader_inst_samplk_vig.show_traceplot()
    # reader_inst_samplk_vig.show_histogram(formation_mark=[-3.5586, 2.2624, -0.4994, 0.2945],
        # dissolution_mark=[-0.1164, 1.5791, -1.6957, 0.6847])
    # reader_inst_samplk_vig.show_acfplot()
    
    netstat_reader_inst_samplk_vig = BSTERGM_latest_exchangeSampler_work()
    netstat_reader_inst_samplk_vig.read_from_csv("example_results_samplk/samplk_jointtimelag_normPrior_vignettesEx_4chain_NetworkStat")
    # netstat_reader_inst_samplk_vig.show_traceplot()

    # #samplk: gof
    # #time 0,1 
    gof_inst_samplk_vig_f = BSTERGM_GOF(model_netStat_samplk_vignettesEx, reader_inst_samplk_vig.MC_formation_samples, 
        is_formation=True, obs_network_seq=samplk_sequence, time_lag=1, additional_netstat_function=model_netStat_samplk_vignettesEx)
    gof_inst_samplk_vig_f.gof_run(num_sim=300, exchange_iter=200)
    gof_inst_samplk_vig_f.show_boxplot(compare=True)

    gof_inst_samplk_vig_d = BSTERGM_GOF(model_netStat_samplk_vignettesEx, reader_inst_samplk_vig.MC_dissolution_samples, 
        is_formation=False, obs_network_seq=samplk_sequence, time_lag=1, additional_netstat_function=model_netStat_samplk_vignettesEx)
    gof_inst_samplk_vig_d.gof_run(num_sim=300, exchange_iter=200)
    gof_inst_samplk_vig_d.show_boxplot(compare=True)


    # #friendship =======================================================================================================================
    from model_settings import model_netStat_friendship_KHEx, friendship_KHEx_initial_formation_vec, friendship_KHEx_initial_dissolution_vec
    
    #friendship: read / quick diagnosis
    reader_inst_friendship_KHEx = BSTERGM_posterior_work()
    reader_inst_friendship_KHEx.read_from_BERGM_csv("example_results_friendship/friendship_jointtimelag_normPrior_KHEx_4chain_formation",
                                                     "example_results_friendship/friendship_jointtimelag_normPrior_KHEx_1chain_dissolution")
    reader_inst_friendship_KHEx.MC_formation_samples = reader_inst_friendship_KHEx.MC_formation_samples[10000::30]
    reader_inst_friendship_KHEx.MC_dissolution_samples = reader_inst_friendship_KHEx.MC_dissolution_samples[10000::30]
    # reader_inst_friendship_KHEx.show_traceplot()
    # reader_inst_friendship_KHEx.show_histogram(formation_mark=[-3.336, 0.480, 0.973, -0.358, 0.650, 1.384, 0.886, -0.389],
    #     dissolution_mark=[-1.132, 0.122, 1.168, -0.577, 0.451, 2.682, 1.121, -1.016])
    # reader_inst_friendship_KHEx.show_acfplot()
        
    netstat_reader_inst_friendship_KHEx_f = BSTERGM_latest_exchangeSampler_work()
    netstat_reader_inst_friendship_KHEx_f.read_from_csv("example_results_friendship/friendship_jointtimelag_normPrior_KHEx_4chain_formation_NetworkStat")
    # netstat_reader_inst_friendship_KHEx_f.show_traceplot()
    netstat_reader_inst_friendship_KHEx_d = BSTERGM_latest_exchangeSampler_work()
    netstat_reader_inst_friendship_KHEx_d.read_from_csv("example_results_friendship/friendship_jointtimelag_normPrior_KHEx_1chain_dissolution_NetworkStat")
    # netstat_reader_inst_friendship_KHEx_d.show_traceplot()

    #friendship: gof
    #time 0,1 
    # # def __init__(self, model_fn, posterior_parameter_samples, is_formation, obs_network_seq, time_lag, additional_netstat_function=None , rng_seed=2021):
    gof_inst_friendship_KHEx_f = BSTERGM_GOF(model_netStat_friendship_KHEx, reader_inst_friendship_KHEx.MC_formation_samples, 
        is_formation=True, obs_network_seq=friendship_sequence, time_lag=0, additional_netstat_function=model_netStat_friendship_KHEx)
    gof_inst_friendship_KHEx_f.gof_run(num_sim=300, exchange_iter=200)
    gof_inst_friendship_KHEx_f.show_boxplot(compare=True)

    gof_inst_friendship_KHEx_d = BSTERGM_GOF(model_netStat_friendship_KHEx, reader_inst_friendship_KHEx.MC_dissolution_samples, 
        is_formation=False, obs_network_seq=friendship_sequence, time_lag=0, additional_netstat_function=model_netStat_friendship_KHEx)
    gof_inst_friendship_KHEx_d.gof_run(num_sim=300, exchange_iter=200)
    gof_inst_friendship_KHEx_d.show_boxplot(compare=True)


    # # tailorshop =======================================================================================================================
    from model_settings import model_netStat_edgeGWESP, edgeGWESP_initial_formation_vec, edgeGWESP_initial_dissolution_vec
    
    #tailorshop: read / quick diagnosis
    reader_inst_tailorshop_edgeGWESP = BSTERGM_posterior_work()
    reader_inst_tailorshop_edgeGWESP.read_from_BERGM_csv("example_results_tailorshop/tailorshop_t01_normPrior_edgeGWESP_0chain_formation",
                                                     "example_results_tailorshop/tailorshop_t01_normPrior_edgeGWESP_2chain_dissolution")
    reader_inst_tailorshop_edgeGWESP.MC_formation_samples = reader_inst_tailorshop_edgeGWESP.MC_formation_samples[8000::20]
    reader_inst_tailorshop_edgeGWESP.MC_dissolution_samples = reader_inst_tailorshop_edgeGWESP.MC_dissolution_samples[8000::20]
    # reader_inst_tailorshop_edgeGWESP.show_traceplot()
    # reader_inst_tailorshop_edgeGWESP.show_histogram(formation_mark=[-2.5621, 0.8827], dissolution_mark=[-0.1878, 0.5118])
    # reader_inst_tailorshop_edgeGWESP.show_acfplot()
        
    netstat_reader_inst_tailorshop_edgeGWESP_f = BSTERGM_latest_exchangeSampler_work()
    netstat_reader_inst_tailorshop_edgeGWESP_f.read_from_csv("example_results_tailorshop/tailorshop_t01_normPrior_edgeGWESP_0chain_formation_NetworkStat")
    # netstat_reader_inst_tailorshop_edgeGWESP_f.show_traceplot()
    netstat_reader_inst_tailorshop_edgeGWESP_d = BSTERGM_latest_exchangeSampler_work()
    netstat_reader_inst_tailorshop_edgeGWESP_d.read_from_csv("example_results_tailorshop/tailorshop_t01_normPrior_edgeGWESP_2chain_dissolution_NetworkStat")
    # netstat_reader_inst_tailorshop_edgeGWESP_d.show_traceplot()

    #tailorshop: gof
    #time 0,1 
    # # def __init__(self, model_fn, posterior_parameter_samples, is_formation, obs_network_seq, time_lag, additional_netstat_function=None , rng_seed=2021):
    gof_inst_tailorshop_KHEx_f = BSTERGM_GOF(model_netStat_edgeGWESP, reader_inst_tailorshop_edgeGWESP.MC_formation_samples, 
        is_formation=True, obs_network_seq=sociational_interactions, time_lag=0, additional_netstat_function=model_netStat_edgeGWESP)
    gof_inst_tailorshop_KHEx_f.gof_run(num_sim=300, exchange_iter=200)
    gof_inst_tailorshop_KHEx_f.show_boxplot(compare=True)

    gof_inst_tailorshop_KHEx_d = BSTERGM_GOF(model_netStat_edgeGWESP, reader_inst_tailorshop_edgeGWESP.MC_dissolution_samples, 
        is_formation=False, obs_network_seq=sociational_interactions, time_lag=0, additional_netstat_function=model_netStat_edgeGWESP)
    gof_inst_tailorshop_KHEx_d.gof_run(num_sim=300, exchange_iter=200)
    gof_inst_tailorshop_KHEx_d.show_boxplot(compare=True)

