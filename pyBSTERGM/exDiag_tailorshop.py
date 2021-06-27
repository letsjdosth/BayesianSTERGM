from BSTERGM_diagnosis import BSTERGM_posterior_work, BSTERGM_latest_exchangeSampler_work
basic_plots = False
netStat_plots = False
gof = False

#============================================================================================
# #tailorshop joint(=t01)
# #formation_good: 0
# #formation_bad: 1 2 3 4
# #dissolution_good: 2 0 3
# #dissolution_bad: 4 1
reader_inst_tailorshop_edgeGWESP = BSTERGM_posterior_work()
reader_inst_tailorshop_edgeGWESP.read_from_BERGM_csv("example_results_tailorshop/tailorshop_t01_normPrior_edgeGWESP_0chain_formation",
                                                    "example_results_tailorshop/tailorshop_t01_normPrior_edgeGWESP_2chain_dissolution")

#tailorshop joint conti
#formation_good: 0>4
#formation_bad: 1,2,3
#dissolution_good: 3>0>1>2>4
#dissolution_bad: 
reader_inst_tailorshop_edgeGWESP_conti = BSTERGM_posterior_work()
reader_inst_tailorshop_edgeGWESP_conti.read_from_BERGM_csv("example_results_tailorshop/tailorshop_jointly_normPrior_edgeGWESP_conti_0chain_formation",
                                                    "example_results_tailorshop/tailorshop_jointly_normPrior_edgeGWESP_conti_3chain_dissolution")

reader_inst_tailorshop_edgeGWESP.MC_formation_samples = reader_inst_tailorshop_edgeGWESP.MC_formation_samples + reader_inst_tailorshop_edgeGWESP_conti.MC_formation_samples
reader_inst_tailorshop_edgeGWESP.MC_dissolution_samples = reader_inst_tailorshop_edgeGWESP.MC_dissolution_samples + reader_inst_tailorshop_edgeGWESP_conti.MC_dissolution_samples
reader_inst_tailorshop_edgeGWESP.MC_formation_samples = reader_inst_tailorshop_edgeGWESP.MC_formation_samples[10000::40]
reader_inst_tailorshop_edgeGWESP.MC_dissolution_samples = reader_inst_tailorshop_edgeGWESP.MC_dissolution_samples[10000::40]

reader_inst_tailorshop_edgeGWESP.print_summary()


if basic_plots:
    reader_inst_tailorshop_edgeGWESP.show_traceplot(layout=(4,1))
    reader_inst_tailorshop_edgeGWESP.show_histogram(formation_mark=[-2.5621, 0.8827],
        dissolution_mark=[-0.1878, 0.5118], layout=(4,1), mean_vline=True)
    reader_inst_tailorshop_edgeGWESP.show_acfplot(layout=(4,1))

if netStat_plots:
    netstat_reader_inst_tailorshop_edgeGWESP_f = BSTERGM_latest_exchangeSampler_work()
    netstat_reader_inst_tailorshop_edgeGWESP_f.read_from_csv("example_results_tailorshop/tailorshop_jointly_normPrior_edgeGWESP_conti_0chain_formation_NetworkStat")
    netstat_reader_inst_tailorshop_edgeGWESP_f.show_traceplot()
    netstat_reader_inst_tailorshop_edgeGWESP_d = BSTERGM_latest_exchangeSampler_work()
    netstat_reader_inst_tailorshop_edgeGWESP_d.read_from_csv("example_results_tailorshop/tailorshop_jointly_normPrior_edgeGWESP_conti_3chain_dissolution_NetworkStat")
    netstat_reader_inst_tailorshop_edgeGWESP_d.show_traceplot()

#============================================================================================
if gof:
    import numpy as np
    from network import UndirectedNetwork, DirectedNetwork
    from BSTERGM_GOF import BSTERGM_GOF
    
    import data_tailor
    from model_settings import model_netStat_edgeGWESP
    
    sociational_interactions = [
        UndirectedNetwork(np.array(data_tailor.KAPFTS1)),
        UndirectedNetwork(np.array(data_tailor.KAPFTS2))
    ]

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

