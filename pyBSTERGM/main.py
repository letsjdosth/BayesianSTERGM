import numpy as np

import Jdata
from network import UndirectedNetwork
from network_sampler import NetworkSampler
from BSTERGM import BSTERGM

sequence1 = [UndirectedNetwork(np.array(Jdata.m1_19_structure)),
    UndirectedNetwork(np.array(Jdata.w1_19_structure)),
    UndirectedNetwork(np.array(Jdata.f1_19_structure))
]

def model_netStat(network : UndirectedNetwork):
    model = []
    #define model
    model.append(network.statCal_edgeNum())
    model.append(network.statCal_geoWeightedESP(0.5))
    return np.array(model)

initial_formation_param = np.array([0.1, 0.1])
initial_dissolution_param = np.array([0.1, 0.1])

test_BSTERGM_sampler = BSTERGM(model_netStat, initial_formation_param, initial_dissolution_param, sequence1, 2021)
test_BSTERGM_sampler.run(10000, exchange_iter=200)
# print(test_BSTERGM_sampler.MC_formation_samples)
# print(test_BSTERGM_sampler.MC_dissolution_samples)
test_BSTERGM_sampler.show_traceplot()
test_BSTERGM_sampler.show_latest_exchangeSampler_netStat_traceplot()