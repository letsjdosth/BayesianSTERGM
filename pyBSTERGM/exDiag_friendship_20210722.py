from BSTERGM_diagnosis import BSTERGM_posterior_work, BSTERGM_latest_exchangeSampler_work
basic_plots = False
basic_plots_save = False
netStat_plots = False
gof = True
table = False

#friendship joint
#formation_good: 
#formation_bad: 
#dissolution_good: 
#dissolution_bad: 
reader_inst_friendship_KHEx = BSTERGM_posterior_work()
reader_inst_friendship_KHEx.read_from_BERGM_csv("example_results_friendship_20210722/friendship_jointly_normPrior_KHEx_20210722_0chain_formation",
                                                    "example_results_friendship_20210722/friendship_jointly_normPrior_KHEx_20210722_0chain_dissolution")
reader_inst_friendship_KHEx.MC_formation_samples = reader_inst_friendship_KHEx.MC_formation_samples[10000::20]
reader_inst_friendship_KHEx.MC_dissolution_samples = reader_inst_friendship_KHEx.MC_dissolution_samples[10000::20] #90000~110000/10000/40 -> 2000~2500

# reader_inst_friendship_KHEx.print_summary()


if basic_plots:
    reader_inst_friendship_KHEx.show_traceplot(layout=(16,1))
    reader_inst_friendship_KHEx.show_histogram(formation_mark=[-3.336, 0.480, 0.973, -0.358, 0.650, 1.384, 0.886, -0.389],
        dissolution_mark=[-1.132, 0.122, 1.168, -0.577, 0.451, 2.682, 1.121, -1.016], layout=(16,1), mean_vline=True)
    reader_inst_friendship_KHEx.show_acfplot(layout=(16,1))

if basic_plots_save:
    reader_inst_friendship_KHEx.save_traceplot("/example_results_friendship_20210722/friendship")
    reader_inst_friendship_KHEx.save_histogram("/example_results_friendship_20210722/friendship", 
        formation_mark=[-3.336, 0.480, 0.973, -0.358, 0.650, 1.384, 0.886, -0.389],
        dissolution_mark=[-1.132, 0.122, 1.168, -0.577, 0.451, 2.682, 1.121, -1.016], mean_vline=True)
    reader_inst_friendship_KHEx.save_acfplot("/example_results_friendship_20210722/friendship")


if netStat_plots:
    netstat_reader_inst_friendship_KHEx_conti_f = BSTERGM_latest_exchangeSampler_work()
    netstat_reader_inst_friendship_KHEx_conti_f.read_from_csv("example_results_friendship_20210722/friendship_jointly_normPrior_KHEx_20210722_0chain_formation_NetworkStat")
    netstat_reader_inst_friendship_KHEx_conti_f.show_traceplot()
    netstat_reader_inst_friendship_KHEx_conti_d = BSTERGM_latest_exchangeSampler_work()
    netstat_reader_inst_friendship_KHEx_conti_d.read_from_csv("example_results_friendship_20210722/friendship_jointly_normPrior_KHEx_20210722_0chain_dissolution_NetworkStat")
    netstat_reader_inst_friendship_KHEx_conti_d.show_traceplot()



#============================================================================================
if gof:
    import numpy as np
    from network import UndirectedNetwork, DirectedNetwork
    from BSTERGM_GOF import BSTERGM_GOF
    
    import data_knecht_friendship
    from model_settings import model_netStat_friendship_KHEx
    
    friendship_sequence = [
        DirectedNetwork(np.array(data_knecht_friendship.friendship_t1)),
        DirectedNetwork(np.array(data_knecht_friendship.friendship_t2)),
        DirectedNetwork(np.array(data_knecht_friendship.friendship_t3)),
        DirectedNetwork(np.array(data_knecht_friendship.friendship_t4))
    ]

    #friendship: gof
    #time 0,1 
    # # def __init__(self, model_fn, posterior_parameter_samples, is_formation, obs_network_seq, time_lag, additional_netstat_function=None , rng_seed=2021):
    gof_inst_friendship_KHEx_f = BSTERGM_GOF(model_netStat_friendship_KHEx, reader_inst_friendship_KHEx.MC_formation_samples, 
        is_formation=True, obs_network_seq=friendship_sequence, time_lag=0, additional_netstat_function=model_netStat_friendship_KHEx)
    gof_inst_friendship_KHEx_f.gof_run(num_sim=300, exchange_iter=200)
    gof_inst_friendship_KHEx_f.show_boxplot(compare=True)
    gof_inst_friendship_KHEx_f.save_boxplot("/example_results_friendship_20210722/friendship", compare=True)

    gof_inst_friendship_KHEx_d = BSTERGM_GOF(model_netStat_friendship_KHEx, reader_inst_friendship_KHEx.MC_dissolution_samples, 
        is_formation=False, obs_network_seq=friendship_sequence, time_lag=0, additional_netstat_function=model_netStat_friendship_KHEx)
    gof_inst_friendship_KHEx_d.gof_run(num_sim=300, exchange_iter=200)
    gof_inst_friendship_KHEx_d.show_boxplot(compare=True)
    gof_inst_friendship_KHEx_d.save_boxplot("/example_results_friendship_20210722/friendship", compare=True)




#table info
if table:
    import numpy as np
    pair = [(0,0),(1,1),(2,2),(3,3),(4,4)]

    f_mean_vec = []
    f_sd_vec = []
    d_mean_vec = []
    d_sd_vec = []


    for idx_pair in pair:
        fhead, dhead = idx_pair
        print("chain combination: formation-", fhead, " dissolution-", dhead)

        reader_inst_friendship_KHEx = BSTERGM_posterior_work()
        reader_inst_friendship_KHEx.read_from_BERGM_csv("example_results_friendship_20210722/friendship_jointly_normPrior_KHEx_20210722_"+str(fhead)+"chain_formation",
                                                            "example_results_friendship_20210722/friendship_jointly_normPrior_KHEx_20210722_"+str(dhead)+"chain_dissolution")
        reader_inst_friendship_KHEx.MC_formation_samples = reader_inst_friendship_KHEx.MC_formation_samples[10000::20]
        reader_inst_friendship_KHEx.MC_dissolution_samples = reader_inst_friendship_KHEx.MC_dissolution_samples[10000::20]


        # reader_inst_tailorshop_edgeGWESP.print_summary()
        formation_means, formation_sds, dissolution_means, dissolution_sds = reader_inst_friendship_KHEx.get_summary()

        f_mean_vec.append(formation_means)
        f_sd_vec.append(formation_sds)
        d_mean_vec.append(dissolution_means)
        d_sd_vec.append(dissolution_sds)

        param_string=["edge","homophily-girls","homophily-boys","heterophily-girls to boys","primary school match","mutual","tties","cties"]
        print(reader_inst_friendship_KHEx.get_summary_LATEXver(param_string))

    # reader_inst_friendship_KHEx.show_histogram(formation_mark=[-3.336, 0.480, 0.973, -0.358, 0.650, 1.384, 0.886, -0.389],
    #     dissolution_mark=[-1.132, 0.122, 1.168, -0.577, 0.451, 2.682, 1.121, -1.016], layout=(16,1), mean_vline=True)

   
    print("\n")
    print("f_mean")
    print(np.array(f_mean_vec).T.round(3))
    print("avg", np.array(f_mean_vec).T.mean(1).round(3))
    print("std", (np.array(f_mean_vec).T.std(1)/np.sqrt(5)).round(3))
    print("\n")
    print("f_sd")
    print(np.array(f_sd_vec).T.round(3))
    print("avg", np.array(f_sd_vec).T.mean(1).round(3))
    print("std", (np.array(f_sd_vec).T.std(1)/np.sqrt(5)).round(3))
    print("\n")
    print("d_mean")
    print(np.array(d_mean_vec).T.round(3))
    print("avg", np.array(d_mean_vec).T.mean(1).round(3))
    print("std", (np.array(d_mean_vec).T.std(1)/np.sqrt(5)).round(3))
    print("\n")
    print("d_sd")
    print(np.array(d_sd_vec).T.round(3))
    print("avg", np.array(d_sd_vec).T.mean(1).round(3))
    print("std", (np.array(d_sd_vec).T.std(1)/np.sqrt(5)).round(3))