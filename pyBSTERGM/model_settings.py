import numpy as np

from network import UndirectedNetwork, DirectedNetwork
import data_samplk, data_knecht_friendship, data_tailor


# samplk model
def model_netStat_samplk_vignettesEx(network): #directed
    model = []
    #define model
    model.append(network.statCal_edgeNum())
    model.append(network.statCal_mutuality())
    model.append(network.statCal_cyclicTriples())
    model.append(network.statCal_transitiveTriples())
    return np.array(model)

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



#friendship model
def model_netStat_friendship_KHEx(network):
    #directed network
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



def model_netStat_friendship_simplified(network):
    #directed network
    model = []
    #define model
    model.append(network.statCal_edgeNum())
    model.append(network.statCal_heterophily(data_knecht_friendship.friendship_sex_girl_index, data_knecht_friendship.friendship_sex_boy_index))#girls->boys
    model.append(network.statCal_match_matrix(np.array(data_knecht_friendship.friendship_primary)))
    model.append(network.statCal_mutuality())
    model.append(network.statCal_transitiveTies())
    model.append(network.statCal_cyclicalTies())
    return np.array(model)

friendship_simplified_initial_formation_vec = [
    np.array([0, 0, 0, 0, 0, 0]), 
    np.array([-1, 0, 0, 0, 0, 0]), 
    np.array([1, 0, 0, 0, 0, 0]), 
    np.array([0, -1, 1, 0, 0, 0]),
    np.array([1, -1, 1, 0, 0, 0]),
    np.array([-1, -1, 1, 0, 0, 0])
]
friendship_simplified_initial_dissolution_vec = [
    np.array([0, 0, 0, 0, 0, 0]), 
    np.array([-1, 0, 0, 0, 0, 0]), 
    np.array([1, 0, 0, 0, 0, 0]), 
    np.array([0, -1, 1, 0, 0, 0]),
    np.array([1, -1, 1, 0, 0, 0]),
    np.array([-1, -1, 1, 0, 0, 0])
]



#tailorshop model
def model_netStat_tailor_social(network):
    #undirected network
    model = []
    model.append(network.statCal_edgeNum)
    model.append(network.statCal_geoWeightedESP(0.3))
    return np.array(model)
