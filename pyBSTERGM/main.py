import multiprocessing as mp
from os import getpid

import numpy as np

import Jdata
from network import UndirectedNetwork
from network_sampler import NetworkSampler
from BSTERGM import BSTERGM


def model_netStat(network : UndirectedNetwork):
    model = []
    #define model
    model.append(network.statCal_edgeNum())
    model.append(network.statCal_geoWeightedESP(0.5))
    return np.array(model)

def procedure(result_queue, network_sequence, initial_formation_param, initial_dissolution_param, result_string, rng_seed=2021, main_iter=30000, ex_iter=50):
    proc_pid = getpid()
    print("pid: ", proc_pid, "start!")

    BSTERGM_sampler = BSTERGM(model_netStat, initial_formation_param, initial_dissolution_param, network_sequence, rng_seed, pid=proc_pid)
    BSTERGM_sampler.run(main_iter, exchange_iter=ex_iter)
    # print(BSTERGM_sampler.MC_formation_samples)
    # print(BSTERGM_sampler.MC_dissolution_samples)
    BSTERGM_sampler.write_posterior_samples(result_string)
    BSTERGM_sampler.write_latest_exchangeSampler_netStat(result_string + "_statNet")

    result_queue.put(BSTERGM_sampler)


    BSTERGM_sampler.show_traceplot()
    # BSTERGM_sampler.show_latest_exchangeSampler_netStat_traceplot()



if __name__=="__main__":
    # sequence1 = [UndirectedNetwork(np.array(Jdata.m1_19_structure)),
    #     UndirectedNetwork(np.array(Jdata.w1_19_structure)),
    #     UndirectedNetwork(np.array(Jdata.f1_19_structure))
    # ]
    # sequence2 = [UndirectedNetwork(np.array(Jdata.m2_19_structure)),
    #     UndirectedNetwork(np.array(Jdata.w2_19_structure)),
    #     UndirectedNetwork(np.array(Jdata.f2_19_structure))
    # ]
    sequence3 = [UndirectedNetwork(np.array(Jdata.m3_19_structure)),
        UndirectedNetwork(np.array(Jdata.w3_19_structure)),
        UndirectedNetwork(np.array(Jdata.f3_19_structure))
    ]


    #core
    core_num = 8
    process_vec = []
    proc_queue = mp.Queue()

    initial_formation_vec = [
        np.array([0,0]), 
        np.array([-0.05, -0.05]), 
        np.array([0.05, 0.05]), 
        np.array([-0.1, -0.1]), 
        np.array([0.1, 0.1]), 
        np.array([-0.2, -0.2]), 
        np.array([0.2, 0.2]), 
        np.array([-0.1, 0.1])
    ]
    initial_dissolution_vec = [
        np.array([0,0]), 
        np.array([-0.05, -0.05]), 
        np.array([0.05, 0.05]), 
        np.array([-0.1, -0.1]), 
        np.array([0.1, 0.1]), 
        np.array([-0.2, -0.2]), 
        np.array([0.2, 0.2]), 
        np.array([-0.1, 0.1])
    ]

    for i in range(core_num):
        process_unit = mp.Process(target=procedure, 
        args=(proc_queue, sequence3, initial_formation_vec[i], initial_dissolution_vec[i], "seq3_"+str(i)+"chain", 2021+i*10, 30000, 50))
        # def procedure(result_queue, network_sequence, initial_formation_param, initial_dissolution_param, result_string, rng_seed=2021, main_iter=30000, ex_iter=50)
        
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

