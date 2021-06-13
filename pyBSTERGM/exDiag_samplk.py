from BSTERGM_diagnosis import BSTERGM_posterior_work, BSTERGM_latest_exchangeSampler_work

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
reader_inst_samplk_vig.read_from_BERGM_csv("example_results_samplk/samplk_jointly_normPrior_vignettesEx_conti_2chain_formation",
                                                    "example_results_samplk/samplk_jointly_normPrior_vignettesEx_conti_2chain_dissolution")
reader_inst_samplk_vig.MC_formation_samples = reader_inst_samplk_vig.MC_formation_samples + reader_inst_samplk_vig_conti.MC_formation_samples
reader_inst_samplk_vig.MC_dissolution_samples = reader_inst_samplk_vig.MC_dissolution_samples + reader_inst_samplk_vig_conti.MC_dissolution_samples


reader_inst_samplk_vig.MC_formation_samples = reader_inst_samplk_vig.MC_formation_samples[10000::40]
reader_inst_samplk_vig.MC_dissolution_samples = reader_inst_samplk_vig.MC_dissolution_samples[10000::40]
reader_inst_samplk_vig.show_traceplot()
reader_inst_samplk_vig.show_histogram(formation_mark=[-3.5586, 2.2624, -0.4994, 0.2945],
    dissolution_mark=[-0.1164, 1.5791, -1.6957, 0.6847])
reader_inst_samplk_vig.show_acfplot()


netstat_reader_inst_samplk_vig = BSTERGM_latest_exchangeSampler_work()
netstat_reader_inst_samplk_vig.read_from_csv("example_results_samplk/samplk_jointtimelag_normPrior_vignettesEx_4chain_NetworkStat")
netstat_reader_inst_samplk_vig.show_traceplot()

netstat_reader_inst_samplk_vig_conti_f = BSTERGM_latest_exchangeSampler_work()
netstat_reader_inst_samplk_vig_conti_f.read_from_csv("example_results_samplk/samplk_jointly_normPrior_vignettesEx_conti_2chain_formation_NetworkStat")
netstat_reader_inst_samplk_vig_conti_f.show_traceplot()
netstat_reader_inst_samplk_vig_conti_d = BSTERGM_latest_exchangeSampler_work()
netstat_reader_inst_samplk_vig_conti_d.read_from_csv("example_results_samplk/samplk_jointly_normPrior_vignettesEx_conti_2chain_dissolution_NetworkStat")
netstat_reader_inst_samplk_vig_conti_d.show_traceplot()



# print(len(reader_inst_samplk_vig.MC_formation_samples))
# print(reader_inst_samplk_vig.MC_formation_samples[-1].round(3).tolist(),"\n",
#     reader_inst_samplk_vig.MC_formation_samples[-205].round(3).tolist(),"\n",
#     reader_inst_samplk_vig.MC_formation_samples[-655].round(3).tolist(),"\n",
#     reader_inst_samplk_vig.MC_formation_samples[-802].round(3).tolist(),"\n",
#     reader_inst_samplk_vig.MC_formation_samples[-1000].round(3).tolist(), "~",
# )
# print(reader_inst_samplk_vig.MC_dissolution_samples[-1].round(3).tolist(),"\n",
#     reader_inst_samplk_vig.MC_dissolution_samples[-205].round(3).tolist(),"\n",
#     reader_inst_samplk_vig.MC_dissolution_samples[-655].round(3).tolist(),"\n",
#     reader_inst_samplk_vig.MC_dissolution_samples[-802].round(3).tolist(),"\n",
#     reader_inst_samplk_vig.MC_dissolution_samples[-1000].round(3).tolist()
# )   
