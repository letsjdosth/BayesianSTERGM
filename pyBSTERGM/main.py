import multiprocessing as mp
from os import getpid

import numpy as np

from network import UndirectedNetwork, DirectedNetwork
from network_sampler import NetworkSampler
from BSTERGM import BSTERGM


#import data
import data_samplk, data_knecht_friendship, data_tailor
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

# samplk
samplk_sequence = [
    DirectedNetwork(np.array(data_samplk.samplk1)),
    DirectedNetwork(np.array(data_samplk.samplk2)),
    DirectedNetwork(np.array(data_samplk.samplk3))
]

# knecht_friendship
friendship_sequence = [
    DirectedNetwork(np.array(data_knecht_friendship.friendship_t1)),
    DirectedNetwork(np.array(data_knecht_friendship.friendship_t2)),
    DirectedNetwork(np.array(data_knecht_friendship.friendship_t3)),
    DirectedNetwork(np.array(data_knecht_friendship.friendship_t4))
]

# # tailorshop
# instrumental_interactions = [
#     DirectedNetwork(np.array(data_tailor.KAPFTI1)),
#     DirectedNetwork(np.array(data_tailor.KAPFTI2)),
# ]
sociational_interactions = [
    UndirectedNetwork(np.array(data_tailor.KAPFTS1)),
    UndirectedNetwork(np.array(data_tailor.KAPFTS2))
]


#import model
# from model_settings import model_netStat_edgeonly, edgeonly_initial_formation_vec, edgeonly_initial_dissolution_vec
# from model_settings import model_netStat_edgeGWdgre, edgeGWdgre_initial_formation_vec, edgeGWdgre_initial_dissolution_vec
from model_settings import model_netStat_edgeGWESP, edgeGWESP_initial_formation_vec, edgeGWESP_initial_dissolution_vec
from model_settings import tailorshop_edgeGWESP_initial_formation_vec_conti, tailorshop_edgeGWESP_initial_dissolution_vec_conti
# from model_settings import model_netStat_edgeGWDSP, edgeGWDSP_initial_formation_vec, edgeGWDSP_initial_dissolution_vec

from model_settings import model_netStat_samplk_vignettesEx, samplk_vignettesEx_initial_formation_vec, samplk_vignettesEx_initial_dissolution_vec
from model_settings import samplk_vignettesEx_initial_formation_vec_conti, samplk_vignettesEx_initial_dissolution_vec_conti

from model_settings import model_netStat_friendship_KHEx, model_netStat_friendship_KHEx_jointly, friendship_KHEx_initial_formation_vec, friendship_KHEx_initial_dissolution_vec
from model_settings import friendship_KHEx_initial_formation_vec_conti, friendship_KHEx_initial_dissolution_vec_conti
# from model_settings import model_netStat_friendship_2hom, friendship_2hom_initial_formation_vec, friendship_2hom_initial_dissolution_vec
# from model_settings import model_netStat_friendship_2hom_noprisch, friendship_2hom_noprisch_initial_formation_vec, friendship_2hom_noprisch_initial_dissolution_vec
# from model_settings import model_netstat_friendship_nondds, friendship_nondds_initial_formation_vec, friendship_nondds_initial_dissolution_vec
# from model_settings import model_netstat_friendship_edge1hom, friendship_edge1hom_initial_formation_vec, friendship_edge1hom_initial_dissolution_vec
# from model_settings import model_netstat_friendship_edge1homMs, friendship_edge1homMs_initial_formation_vec, friendship_edge1homMs_initial_dissolution_vec

# from model_settings import model_netStat_tailor_social_edgeDegrESP, tailor_social_edgeDegrESP_initial_formation_vec, tailor_social_edgeDegrESP_initial_dissolution_vec
# from model_settings import model_netStat_tailor_social_edgeDegrESPDSP, tailor_social_edgeDegrESPDSP_initial_formation_vec, tailor_social_edgeDegrESPDSP_initial_dissolution_vec



#for multiprocessing
def procedure_run_each_bergm(result_queue, bergm_object, main_iter, ex_iter, proposal_cov_rate, result_string):
    proc_pid = getpid()
    print("pid: ", proc_pid, "start!")
    bergm_object.pid = proc_pid
    bergm_object.run(main_iter, ex_iter, proposal_cov_rate, console_output_str=result_string)
    bergm_object.write_posterior_samples(result_string)
    bergm_object.write_latest_exchangeSampler_netStat(result_string + "_NetworkStat")
    result_queue.put(bergm_object)

    bergm_object.show_traceplot()


if __name__=="__main__":
    parallel_BSTERGM_num = 5
    process_vec = []
    proc_queue = mp.Queue()
    for i in range(parallel_BSTERGM_num):
        # BSTERGM:: def __init__(self, model_fn, initial_formation_param, initial_dissolution_param, obs_network_seq, rng_seed=2021, pid=None):
        bstergm_object = BSTERGM(model_netStat_edgeGWESP, 
                            tailorshop_edgeGWESP_initial_formation_vec_conti[i], tailorshop_edgeGWESP_initial_dissolution_vec_conti[i],
                            sociational_interactions, rng_seed=i*10+1)

        bergm_object_formation, bergm_object_disolution = bstergm_object.get_bergm_objects_with_setting(time_lag='joint')
        # def procedure_run_each_bergm(result_queue, bergm_object, main_iter, ex_iter, proposal_cov_rate, result_string):
        process_unit_f = mp.Process(target=procedure_run_each_bergm, 
                                args=(proc_queue, bergm_object_formation, 30000, 50, 0.01,
                                    "tailorshop_jointly_normPrior_edgeGWESP_conti_"+str(i)+"chain_formation"))
        process_vec.append(process_unit_f)
        process_unit_d = mp.Process(target=procedure_run_each_bergm, 
                                args=(proc_queue, bergm_object_disolution, 30000, 50, 0.01,
                                    "tailorshop_jointly_normPrior_edgeGWESP_conti_"+str(i)+"chain_dissolution"))
        process_vec.append(process_unit_d)



    for unit_proc in process_vec:
        unit_proc.start()
    
    mp_result_vec = []
    for i, _ in enumerate(process_vec):
        each_result = proc_queue.get()
        print("mp_result_vec_object:", i)
        mp_result_vec.append(each_result)

    for unit_proc in process_vec:
        unit_proc.join()
    print("exit multiprocessing")
