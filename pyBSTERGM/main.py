import multiprocessing as mp
from os import getpid

import numpy as np

import data_samplk, data_knecht_friendship, data_tailor
from network import UndirectedNetwork, DirectedNetwork
from network_sampler import NetworkSampler
from BSTERGM import BSTERGM

#data

#Jdata
# sequence1 = [UndirectedNetwork(np.array(data_Jdata.m1_19_structure)),
#     UndirectedNetwork(np.array(data_Jdata.w1_19_structure)),
#     UndirectedNetwork(np.array(data_Jdata.f1_19_structure))
# ]
# sequence2 = [UndirectedNetwork(np.array(data_Jdata.m2_19_structure)),
#     UndirectedNetwork(np.array(data_Jdata.w2_19_structure)),
#     UndirectedNetwork(np.array(data_Jdata.f2_19_structure))
# ]
# sequence3 = [UndirectedNetwork(np.array(data_Jdata.m3_19_structure)),
#     UndirectedNetwork(np.array(data_Jdata.w3_19_structure)),
#     UndirectedNetwork(np.array(data_Jdata.f3_19_structure))
# ]

#samplk
# samplk_sequence = [
#     DirectedNetwork(np.array(data_samplk.samplk1)),
#     DirectedNetwork(np.array(data_samplk.samplk2)),
#     DirectedNetwork(np.array(data_samplk.samplk3))
# ]

# knecht_friendship
friendship_sequence = [
    DirectedNetwork(np.array(data_knecht_friendship.friendship_t1)),
    DirectedNetwork(np.array(data_knecht_friendship.friendship_t2)),
    DirectedNetwork(np.array(data_knecht_friendship.friendship_t3)),
    DirectedNetwork(np.array(data_knecht_friendship.friendship_t4))
]

# # tailor shop
# instrumental_interactions = [
#     DirectedNetwork(np.array(data_tailor.KAPFTI1)),
#     DirectedNetwork(np.array(data_tailor.KAPFTI2)),
# ]

sociational_interactions = [
    UndirectedNetwork(np.array(data_tailor.KAPFTS1)),
    UndirectedNetwork(np.array(data_tailor.KAPFTS2))
]


#for multiprocessing
def procedure(result_queue, network_sequence, model_netStat_func, 
        initial_formation_param, initial_dissolution_param, result_string, rng_seed=2021, main_iter=30000, ex_iter=50):
    
    proc_pid = getpid()
    print("pid: ", proc_pid, "start!")

    BSTERGM_sampler = BSTERGM(model_netStat_func, initial_formation_param, initial_dissolution_param, network_sequence, rng_seed, pid=proc_pid)
    BSTERGM_sampler.run(main_iter, exchange_iter=ex_iter)
    # print(BSTERGM_sampler.MC_formation_samples)
    # print(BSTERGM_sampler.MC_dissolution_samples)
    BSTERGM_sampler.write_posterior_samples(result_string)
    BSTERGM_sampler.write_latest_exchangeSampler_netStat(result_string + "_NetworkStat")

    result_queue.put(BSTERGM_sampler)

    BSTERGM_sampler.show_traceplot()
    # BSTERGM_sampler.show_latest_exchangeSampler_netStat_traceplot()

def procedure_1dim_sampler(result_queue, network_sequence, model_netStat_func,
        initial_formation_param, initial_dissolution_param, result_string, rng_seed=2021, main_iter=30000, ex_iter=50):
    
    proc_pid = getpid()
    print("pid: ", proc_pid, "start!")

    BSTERGM_sampler = BSTERGM(model_netStat_func, initial_formation_param, initial_dissolution_param, network_sequence, rng_seed, pid=proc_pid)
    BSTERGM_sampler.run_1dim(main_iter, exchange_iter=ex_iter)
    # print(BSTERGM_sampler.MC_formation_samples)
    # print(BSTERGM_sampler.MC_dissolution_samples)
    BSTERGM_sampler.write_posterior_samples(result_string)
    BSTERGM_sampler.write_latest_exchangeSampler_netStat(result_string + "_NetworkStat")

    result_queue.put(BSTERGM_sampler)

    BSTERGM_sampler.show_traceplot()
    # BSTERGM_sampler.show_latest_exchangeSampler_netStat_traceplot()



if __name__=="__main__":
    #core
    core_num = 6
    process_vec = []
    proc_queue = mp.Queue()

    # from model_settings import model_netStat_samplk_vignettesEx, samplk_vignettesEx_initial_formation_vec, samplk_vignettesEx_initial_dissolution_vec

    # from model_settings import model_netStat_friendship_KHEx, friendship_KHEx_initial_formation_vec, friendship_KHEx_initial_dissolution_vec
    # from model_settings import model_netStat_friendship_2hom, friendship_2hom_initial_formation_vec, friendship_2hom_initial_dissolution_vec
    # from model_settings import model_netStat_friendship_2hom_noprisch, friendship_2hom_noprisch_initial_formation_vec, friendship_2hom_noprisch_initial_dissolution_vec
    # from model_settings import model_netstat_friendship_nondds, friendship_nondds_initial_formation_vec, friendship_nondds_initial_dissolution_vec

    from model_settings import model_netStat_tailor_social_edgeDegrESP, tailor_social_edgeDegrESP_initial_formation_vec, tailor_social_edgeDegrESP_initial_dissolution_vec
    from model_settings import model_netStat_tailor_social_edgeDegrESPDSP, tailor_social_edgeDegrESPDSP_initial_formation_vec, tailor_social_edgeDegrESPDSP_initial_dissolution_vec
    


    for i in range(core_num):
        # format
        # def procedure(result_queue, network_sequence, model_netStat_func, 
        #       initial_formation_param, initial_dissolution_param, 
        #       result_string, rng_seed=2021, main_iter=30000, ex_iter=50):

        # samplk
        # process_unit = mp.Process(target=procedure, 
        # args=(proc_queue, samplk_sequence, model_netStat_samplk_vignettesEx, 
        #     samplk_vignettesEx_initial_formation_vec[i], samplk_vignettesEx_initial_dissolution_vec[i], 
        #     "samplk_normPrior_vignetModel_"+str(i)+"chain", 2021+i*10, 80000, 30))


        # friendship
        # process_unit = mp.Process(target=procedure, 
        # args=(proc_queue, friendship_sequence, model_netstat_friendship_nondds, 
        #     friendship_nondds_initial_formation_vec[i], friendship_nondds_initial_dissolution_vec[i], 
        #     "friendship_normPrior_noddsModel_"+str(i)+"chain", 2021+i*10, 80000, 30))

        # process_unit = mp.Process(target=procedure_1dim_sampler, 
        # args=(proc_queue, friendship_sequence, model_netStat_friendship_simplified, 
        #     friendship_simplified_initial_formation_vec[i], friendship_simplified_initial_dissolution_vec[i], 
        #     "friendship_sequence_simplified_run1dim_"+str(i)+"chain", 2021+i*10, 20000, 20))

        # process_unit = mp.Process(target=procedure, 
        # args=(proc_queue, friendship_sequence, model_netStat_friendship_2hom_noprisch, 
        #     friendship_2hom_noprisch_initial_formation_vec[i], friendship_2hom_noprisch_initial_dissolution_vec[i], 
        #     "friendship_sequence_2homNoprisch_"+str(i)+"chain", 2021+i*10, 80000, 30))

        # tailorshop-social
        process_unit = mp.Process(target=procedure, 
        args=(proc_queue, sociational_interactions, model_netStat_tailor_social_edgeDegrESP, 
            tailor_social_edgeDegrESP_initial_formation_vec[i], tailor_social_edgeDegrESP_initial_dissolution_vec[i], 
            "tailorshop_social_normPrior_edgeGWDgr025GWESP025"+str(i)+"chain", 2021+i*10, 80000, 30))

        # process_unit = mp.Process(target=procedure, 
        # args=(proc_queue, sociational_interactions, model_netStat_tailor_social_edgeDegrESPDSP, 
        #     tailor_social_edgeDegrESPDSP_initial_formation_vec[i], tailor_social_edgeDegrESPDSP_initial_dissolution_vec[i], 
        #     "tailorshop_social_normPrior_edgeGWDgr025GWESP025GWDSP025_"+str(i)+"chain", 2021+i*10, 80000, 30))



        process_vec.append(process_unit)
    
    for unit_proc in process_vec:
        unit_proc.start()
    
    mp_result_vec = []
    for _ in range(core_num):
        each_result = proc_queue.get()
        # print("mp_result_vec_object:", each_result)
        mp_result_vec.append(each_result)

    for unit_proc in process_vec:
        unit_proc.join()
    print("exit multiprocessing")
