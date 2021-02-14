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

#tailor shop
# instrumental_interactions = [
#     DirectedNetwork(np.array(data_tailor.KAPFTI1)),
#     DirectedNetwork(np.array(data_tailor.KAPFTI2)),
# ]

# sociational_interactions = [
#     UndirectedNetwork(np.array(data_tailor.KAPFTS1)),
#     UndirectedNetwork(np.array(data_tailor.KAPFTS2))
# ]


def model_netStat_samplk_vignettesEx(network):
    model = []
    #define model
    model.append(network.statCal_edgeNum())
    model.append(network.statCal_mutuality())
    model.append(network.statCal_cyclicTriples())
    model.append(network.statCal_transitiveTriples())
    return np.array(model)

def model_netStat_friendship_KHEx(network):
    model = []
    #define model
    model.append(network.statCal_edgeNum())
    model.append(network.statCal_homophily(data_knecht_friendship.friendship_sex_girl_index)) #girls
    model.append(network.statCal_homophily(data_knecht_friendship.friendship_sex_boy_index)) #boys
    model.append(network.statCal_heterophily(data_knecht_friendship.friendship_sex_girl_index, data_knecht_friendship.friendship_sex_boy_index))#girls->boys
    model.append(network.statCal_match_matrix(np.array(data_knecht_friendship.friendship_primary)))
    model.append(network.statCal_mutuality())
    model.append(network.statCal_transitiveTies())
    model.append(network.statCal_cyclicalTies())
    return np.array(model)

#for multiprocessing
def procedure(result_queue, network_sequence, model_netStat_func, initial_formation_param, initial_dissolution_param, result_string, rng_seed=2021, main_iter=30000, ex_iter=50):
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


if __name__=="__main__":
    #core
    core_num = 6
    process_vec = []
    proc_queue = mp.Queue()

    friendship_KHEx_initial_formation_vec = [
        np.array([0, 0, 0, 0, 0, 0, 0, 0]), 
        np.array([-1, 0, 0, 0, 0, 0, 0, 0]), 
        np.array([1, 0, 0, 0, 0, 0, 0, 0]), 
        np.array([0, 0, 1, 0, 1, 1, 0, 0]),
        np.array([1, 0, 1, 0, 1, 1, 0, 0]),
        np.array([-1, 0, 1, 0, 1, 1, 0, 0]), 
    ]
    friendship_KHEx_initial_dissolution_vec = [
        np.array([0, 0, 0, 0, 0, 0, 0, 0]), 
        np.array([-1, 0, 0, 0, 0, 0, 0, 0]), 
        np.array([1, 0, 0, 0, 0, 0, 0, 0]), 
        np.array([0, 0, 1, 0, 1, 1, 0, 0]),
        np.array([1, 0, 1, 0, 1, 1, 0, 0]),
        np.array([-1, 0, 1, 0, 1, 1, 0, 0]), 
    ]

    samplk_vignettesEx_initial_formation_vec = [
        np.array([0,0,0,0]),
        np.array([1,0,0,0]),
        np.array([-1,0,0,0]),
        np.array([0,0.5,-0.5,0]),
        np.array([1,0.5,-0.5,0]),
        np.array([-1,0.5,-0.5,0]),
    ]
    samplk_vignettesEx_initial_dissolution_vec = [
        np.array([0,0,0,0]),
        np.array([1,0,0,0]),
        np.array([-1,0,0,0]),
        np.array([0,0.5,-0.5,0]),
        np.array([1,0.5,-0.5,0]),
        np.array([-1,0.5,-0.5,0]),
    ]

    for i in range(core_num):
        process_unit = mp.Process(target=procedure, 
        args=(proc_queue, samplk_sequence, model_netStat_samplk_vignettesEx, 
            samplk_vignettesEx_initial_formation_vec[i], samplk_vignettesEx_initial_dissolution_vec[i], 
            "samplk_sequence_Exmodel_run_"+str(i)+"chain", 2021+i*10, 80000, 30))
        # def procedure(result_queue, network_sequence, model_netStat_func, 
        #       initial_formation_param, initial_dissolution_param, 
        #       result_string, rng_seed=2021, main_iter=30000, ex_iter=50):
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
