from BSTERGM_diagnosis import BSTERGM_posterior_work, BSTERGM_latest_exchangeSampler_work

#friendship joint
#formation_good: 4 > 2 > 1 
#formation_bad: 0, 3
#dissolution_good: 1 > 2 > 0 > 4 
#dissolution_bad: 3
reader_inst_friendship_KHEx = BSTERGM_posterior_work()
reader_inst_friendship_KHEx.read_from_BERGM_csv("example_results_friendship/friendship_jointtimelag_normPrior_KHEx_4chain_formation",
                                                    "example_results_friendship/friendship_jointtimelag_normPrior_KHEx_1chain_dissolution")
#friendship joint conti
#formation_good: 4>1>2
#formation_bad: 3>0
#dissolution_good: 0>4>1,
#dissolution_bad: 2,3
reader_inst_friendship_KHEx_conti = BSTERGM_posterior_work()
reader_inst_friendship_KHEx.read_from_BERGM_csv("example_results_friendship/friendship_jointly_normPrior_KHEx_conti_4chain_formation",
                                                    "example_results_friendship/friendship_jointly_normPrior_KHEx_conti_0chain_dissolution")
reader_inst_friendship_KHEx.MC_formation_samples = reader_inst_friendship_KHEx.MC_formation_samples + reader_inst_friendship_KHEx_conti.MC_formation_samples
reader_inst_friendship_KHEx.MC_dissolution_samples = reader_inst_friendship_KHEx.MC_dissolution_samples + reader_inst_friendship_KHEx_conti.MC_dissolution_samples

reader_inst_friendship_KHEx.MC_formation_samples = reader_inst_friendship_KHEx.MC_formation_samples[10000::40]
reader_inst_friendship_KHEx.MC_dissolution_samples = reader_inst_friendship_KHEx.MC_dissolution_samples[10000::40] #90000~110000/10000/40 -> 2000~2500
reader_inst_friendship_KHEx.show_traceplot()
reader_inst_friendship_KHEx.show_histogram(formation_mark=[-3.336, 0.480, 0.973, -0.358, 0.650, 1.384, 0.886, -0.389],
    dissolution_mark=[-1.132, 0.122, 1.168, -0.577, 0.451, 2.682, 1.121, -1.016])
reader_inst_friendship_KHEx.show_acfplot()


netstat_reader_inst_friendship_KHEx_f = BSTERGM_latest_exchangeSampler_work()
netstat_reader_inst_friendship_KHEx_f.read_from_csv("example_results_friendship/friendship_jointtimelag_normPrior_KHEx_4chain_formation_NetworkStat")
netstat_reader_inst_friendship_KHEx_f.show_traceplot()
netstat_reader_inst_friendship_KHEx_d = BSTERGM_latest_exchangeSampler_work()
netstat_reader_inst_friendship_KHEx_d.read_from_csv("example_results_friendship/friendship_jointtimelag_normPrior_KHEx_1chain_dissolution_NetworkStat")
netstat_reader_inst_friendship_KHEx_d.show_traceplot()


netstat_reader_inst_friendship_KHEx_conti_f = BSTERGM_latest_exchangeSampler_work()
netstat_reader_inst_friendship_KHEx_conti_f.read_from_csv("example_results_friendship/friendship_jointly_normPrior_KHEx_conti_0chain_dissolution_NetworkStat")
netstat_reader_inst_friendship_KHEx_conti_f.show_traceplot()
netstat_reader_inst_friendship_KHEx_conti_d = BSTERGM_latest_exchangeSampler_work()
netstat_reader_inst_friendship_KHEx_conti_d.read_from_csv("example_results_friendship/friendship_jointly_normPrior_KHEx_conti_4chain_formation_NetworkStat")
netstat_reader_inst_friendship_KHEx_conti_d.show_traceplot()



# print(len(reader_inst_friendship_KHEx.MC_formation_samples))
# print(reader_inst_friendship_KHEx.MC_formation_samples[-1].round(3).tolist(),"\n",
#     reader_inst_friendship_KHEx.MC_formation_samples[-205].round(3).tolist(),"\n",
#     reader_inst_friendship_KHEx.MC_formation_samples[-655].round(3).tolist(),"\n",
#     reader_inst_friendship_KHEx.MC_formation_samples[-802].round(3).tolist(),"\n",
#     reader_inst_friendship_KHEx.MC_formation_samples[-1000].round(3).tolist(), "~",
# )
# print(reader_inst_friendship_KHEx.MC_dissolution_samples[-1].round(3).tolist(),"\n",
#     reader_inst_friendship_KHEx.MC_dissolution_samples[-205].round(3).tolist(),"\n",
#     reader_inst_friendship_KHEx.MC_dissolution_samples[-655].round(3).tolist(),"\n",
#     reader_inst_friendship_KHEx.MC_dissolution_samples[-802].round(3).tolist(),"\n",
#     reader_inst_friendship_KHEx.MC_dissolution_samples[-1000].round(3).tolist()
# )   
