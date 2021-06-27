from BSTERGM_diagnosis import BSTERGM_posterior_work, BSTERGM_latest_exchangeSampler_work
basic_plots = False
netStat_plots = False
gof = True


# #samplk joint
# #good: 2/3/4 chain / others: bad
reader_inst_samplk_vig = BSTERGM_posterior_work()
reader_inst_samplk_vig.read_from_BSTERGM_csv("example_results_samplk/samplk_jointtimelag_normPrior_vignettesEx_4chain", 4, 4)
#samplk joint conti
#formation_good: 2>4>0
#formation_bad: 1,3
#dissolution_good: 2>1,3 >4
#dissolution_bad: 0
reader_inst_samplk_vig_conti = BSTERGM_posterior_work()
reader_inst_samplk_vig_conti.read_from_BERGM_csv("example_results_samplk/samplk_jointly_normPrior_vignettesEx_conti_2chain_formation",
                                                    "example_results_samplk/samplk_jointly_normPrior_vignettesEx_conti_2chain_dissolution")
                                                    
reader_inst_samplk_vig.MC_formation_samples = reader_inst_samplk_vig.MC_formation_samples + reader_inst_samplk_vig_conti.MC_formation_samples
reader_inst_samplk_vig.MC_dissolution_samples = reader_inst_samplk_vig.MC_dissolution_samples + reader_inst_samplk_vig_conti.MC_dissolution_samples
reader_inst_samplk_vig.MC_formation_samples = reader_inst_samplk_vig.MC_formation_samples[10000::40]
reader_inst_samplk_vig.MC_dissolution_samples = reader_inst_samplk_vig.MC_dissolution_samples[10000::40]


if basic_plots:
    reader_inst_samplk_vig.show_traceplot()
    reader_inst_samplk_vig.show_histogram(formation_mark=[-3.5586, 2.2624, -0.4994, 0.2945],
        dissolution_mark=[-0.1164, 1.5791, -1.6957, 0.6847])
    reader_inst_samplk_vig.show_acfplot()


    netstat_reader_inst_samplk_vig = BSTERGM_latest_exchangeSampler_work()
    netstat_reader_inst_samplk_vig.read_from_csv("example_results_samplk/samplk_jointtimelag_normPrior_vignettesEx_4chain_NetworkStat")
    netstat_reader_inst_samplk_vig.show_traceplot()

if netStat_plots:
    netstat_reader_inst_samplk_vig_conti_f = BSTERGM_latest_exchangeSampler_work()
    netstat_reader_inst_samplk_vig_conti_f.read_from_csv("example_results_samplk/samplk_jointly_normPrior_vignettesEx_conti_2chain_formation_NetworkStat")
    netstat_reader_inst_samplk_vig_conti_f.show_traceplot()
    netstat_reader_inst_samplk_vig_conti_d = BSTERGM_latest_exchangeSampler_work()
    netstat_reader_inst_samplk_vig_conti_d.read_from_csv("example_results_samplk/samplk_jointly_normPrior_vignettesEx_conti_2chain_dissolution_NetworkStat")
    netstat_reader_inst_samplk_vig_conti_d.show_traceplot()


#============================================================================================
if gof:
    import numpy as np
    from network import UndirectedNetwork, DirectedNetwork
    from BSTERGM_GOF import BSTERGM_GOF
    
    import data_samplk
    from model_settings import model_netStat_samplk_vignettesEx
    
    samplk_sequence = [
        DirectedNetwork(np.array(data_samplk.samplk1)),
        DirectedNetwork(np.array(data_samplk.samplk2)),
        DirectedNetwork(np.array(data_samplk.samplk3))
    ]

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

