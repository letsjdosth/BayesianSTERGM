from BSTERGM_diagnosis import BSTERGM_posterior_work, BSTERGM_latest_exchangeSampler_work
basic_plots = False
basic_plots_save = False
netStat_plots = False
gof = True
table = False

# #samplk joint
reader_inst_samplk_vig = BSTERGM_posterior_work()
reader_inst_samplk_vig.read_from_BERGM_csv("example_results_samplk_20210722/samplk_jointly_normPrior_vigEx_20210722_0chain_formation",
                                                    "example_results_samplk_20210722/samplk_jointly_normPrior_vigEx_20210722_0chain_dissolution")
reader_inst_samplk_vig.MC_formation_samples = reader_inst_samplk_vig.MC_formation_samples[10000::20]
reader_inst_samplk_vig.MC_dissolution_samples = reader_inst_samplk_vig.MC_dissolution_samples[10000::20]

# reader_inst_samplk_vig.print_summary()

if basic_plots:
    reader_inst_samplk_vig.show_traceplot(layout=(8,1))
    reader_inst_samplk_vig.show_histogram(formation_mark=[-3.5586, 2.2624, -0.4994, 0.2945],
        dissolution_mark=[-0.1164, 1.5791, -1.6957, 0.6847], layout=(8,1), mean_vline=True)
    reader_inst_samplk_vig.show_acfplot(layout=(8,1))


if basic_plots_save:
    reader_inst_samplk_vig.save_traceplot("/example_results_samplk_20210722/samplk")
    reader_inst_samplk_vig.save_histogram("/example_results_samplk_20210722/samplk", 
        formation_mark=[-3.5586, 2.2624, -0.4994, 0.2945],
        dissolution_mark=[-0.1164, 1.5791, -1.6957, 0.6847], mean_vline=True)
    reader_inst_samplk_vig.save_acfplot("/example_results_samplk_20210722/samplk")


if netStat_plots:
    netstat_reader_inst_samplk_vig_conti_f = BSTERGM_latest_exchangeSampler_work()
    netstat_reader_inst_samplk_vig_conti_f.read_from_csv("example_results_samplk_20210722/samplk_jointly_normPrior_vigEx_20210722_0chain_formation_NetworkStat")
    netstat_reader_inst_samplk_vig_conti_f.show_traceplot()
    netstat_reader_inst_samplk_vig_conti_d = BSTERGM_latest_exchangeSampler_work()
    netstat_reader_inst_samplk_vig_conti_d.read_from_csv("example_results_samplk_20210722/samplk_jointly_normPrior_vigEx_20210722_0chain_dissolution_NetworkStat")
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
    gof_inst_samplk_vig_f.save_boxplot("/example_results_samplk_20210722/samplk", compare=True)

    gof_inst_samplk_vig_d = BSTERGM_GOF(model_netStat_samplk_vignettesEx, reader_inst_samplk_vig.MC_dissolution_samples, 
        is_formation=False, obs_network_seq=samplk_sequence, time_lag=1, additional_netstat_function=model_netStat_samplk_vignettesEx)
    gof_inst_samplk_vig_d.gof_run(num_sim=300, exchange_iter=200)
    gof_inst_samplk_vig_d.show_boxplot(compare=True)
    gof_inst_samplk_vig_d.save_boxplot("/example_results_samplk_20210722/samplk", compare=True)


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

        reader_inst_samplk_vig = BSTERGM_posterior_work()
        reader_inst_samplk_vig.read_from_BERGM_csv("example_results_samplk_20210722/samplk_jointly_normPrior_vigEx_20210722_"+str(fhead)+"chain_formation",
                                                            "example_results_samplk_20210722/samplk_jointly_normPrior_vigEx_20210722_"+str(dhead)+"chain_dissolution")
        reader_inst_samplk_vig.MC_formation_samples = reader_inst_samplk_vig.MC_formation_samples[10000::20]
        reader_inst_samplk_vig.MC_dissolution_samples = reader_inst_samplk_vig.MC_dissolution_samples[10000::20]


        # reader_inst_tailorshop_edgeGWESP.print_summary()
        formation_means, formation_sds, dissolution_means, dissolution_sds = reader_inst_samplk_vig.get_summary()

        f_mean_vec.append(formation_means)
        f_sd_vec.append(formation_sds)
        d_mean_vec.append(dissolution_means)
        d_sd_vec.append(dissolution_sds)

        param_string=["edge","mutual","ctriad","ttriad"]
        print(reader_inst_samplk_vig.get_summary_LATEXver(param_string))

    # formation_mark=[-3.5586, 2.2624, -0.4994, 0.2945],
    # dissolution_mark=[-0.1164, 1.5791, -1.6957, 0.6847]
   
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