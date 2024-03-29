import numpy as np

from network import UndirectedNetwork, DirectedNetwork
import data_samplk, data_knecht_friendship, data_tailor

# default models
def model_netStat_edgeonly(network):
    model = []
    model.append(network.statCal_edgeNum())
    return np.array(model)

edgeonly_initial_formation_vec = [
    np.array([0]),
    np.array([1]),
    np.array([-1]),
    np.array([2]),
    np.array([-2]),
    np.array([3])
]

edgeonly_initial_dissolution_vec = [
    np.array([0]),
    np.array([1]),
    np.array([-1]),
    np.array([2]),
    np.array([-2]),
    np.array([3])
]

def model_netStat_edgeGWdgre(network):
    model = []
    model.append(network.statCal_edgeNum())
    model.append(network.statCal_geoWeightedDegree(0.25))
    return np.array(model)


edgeGWdgre_initial_formation_vec = [
    np.array([0,0]),
    np.array([1,0]),
    np.array([-1,0]),
    np.array([0,-1]),
    np.array([0,1]),
    np.array([0.5,0.5])
]

edgeGWdgre_initial_dissolution_vec = [
    np.array([0,0]),
    np.array([1,0]),
    np.array([-1,0]),
    np.array([0,-1]),
    np.array([0,1]),
    np.array([0.5,0.5])
]


def model_netStat_edgeGWESP(network):
    model = []
    model.append(network.statCal_edgeNum())
    model.append(network.statCal_geoWeightedESP(0.25))
    return np.array(model)


edgeGWESP_initial_formation_vec = [
    np.array([0,0]),
    np.array([1,0]),
    np.array([-1,0]),
    np.array([0,-1]),
    np.array([0,1]),
    np.array([0.5,0.5])
]

edgeGWESP_initial_dissolution_vec = [
    np.array([0,0]),
    np.array([1,0]),
    np.array([-1,0]),
    np.array([0,-1]),
    np.array([0,1]),
    np.array([0.5,0.5])
]



def model_netStat_edgeGWDSP(network):
    model = []
    model.append(network.statCal_edgeNum())
    model.append(network.statCal_geoWeightedDSP(0.25))
    return np.array(model)


edgeGWDSP_initial_formation_vec = [
    np.array([0,0]),
    np.array([1,0]),
    np.array([-1,0]),
    np.array([0,-1]),
    np.array([0,1]),
    np.array([0.5,0.5])
]

edgeGWDSP_initial_dissolution_vec = [
    np.array([0,0]),
    np.array([1,0]),
    np.array([-1,0]),
    np.array([0,-1]),
    np.array([0,1]),
    np.array([0.5,0.5])
]

tailorshop_edgeGWESP_initial_formation_vec_conti = [
    np.array([-1.331, 0.098]),
    np.array([-0.662, -0.141]),
    np.array([0.708, -1.956]),
    np.array([1.675, -2.248]),
    np.array([2.652, -3.028]),
]
tailorshop_edgeGWESP_initial_dissolution_vec_conti = [
    np.array([0.453, 0.123]),
    np.array([1.587, -0.149]),
    np.array([-0.592, 0.885]),
    np.array([1.118, -0.302]),
    np.array([-0.076, 0.605]),
]


# samplk model
def model_netStat_samplk_vignettesEx(network): #directed
    model = []
    #define model
    model.append(network.statCal_edgeNum())
    model.append(network.statCal_mutuality())
    model.append(network.statCal_cyclicTriples())
    model.append(network.statCal_transitiveTriples())
    return np.array(model)

# samplk_vignettesEx_initial_formation_vec = [
#     np.array([0,0,0,0]),
#     np.array([1,0,0,0]),
#     np.array([-1,0,0,0]),
#     np.array([0,0.5,-0.5,0]),
#     np.array([1,0.5,-0.5,0]),
#     np.array([-1,0.5,-0.5,0]),
# ]
# samplk_vignettesEx_initial_dissolution_vec = [
#     np.array([0,0,0,0]),
#     np.array([1,0,0,0]),
#     np.array([-1,0,0,0]),
#     np.array([0,0.5,-0.5,0]),
#     np.array([1,0.5,-0.5,0]),
#     np.array([-1,0.5,-0.5,0]),
# ]


# samplk_vignettesEx_initial_formation_vec_conti = [
#     np.array([-4.027, 1.844, 0.287, 0.134]),
#     np.array([-7.958, 5.184, -2.728, 1.573]),
#     np.array([-2.89, 1.708, -1.339, 0.374]),
#     np.array([-6.325, 2.925, -0.778, 0.837]),
#     np.array([-5.574, -0.55, 0.6, -0.085]),
# ]
# samplk_vignettesEx_initial_dissolution_vec_conti = [
#     np.array([-0.291, -0.05, -1.743, 1.586]),
#     np.array([-0.629, 1.956, 0.629, 1.051]),
#     np.array([0.307, -1.005, 0.711, 0.643]),
#     np.array([0.897, 0.507, -1.429, 0.01]),
#     np.array([0.49, 0.641, -1.589, -0.155]),
# ]


samplk_vignettesEx_initial_formation_vec = [
    np.array([-5.14, 2.50, -0.70, 0.42]),
    np.array([-4.12, 1.68, -0.60, 0.28]),
    np.array([-3.55, 2.24, -0.49, 0.29]),
    np.array([-4.84, 2.04, -0.56, 0.37]),
    np.array([-3.48, 1.84, -1.02, 0.54]),
    np.array([-3.94, 2.54, -0.67, 0.37]),
]
samplk_vignettesEx_initial_dissolution_vec = [
    np.array([0.22, 0.81, -0.16, 0.49]),
    np.array([0.25, 0.75, -0.42, 0.54]),
    np.array([-0.12, 1.59, -1.69, 0.70]),
    np.array([0.21, 0.93, -0.33, 0.50]),
    np.array([0.27, 0.74, 0.02, 0.44]),
    np.array([0.20, 0.78, -0.08, 0.49]),
]


# extended things
from scipy.linalg import block_diag
friendship_joint_times = 3 #length 4
friendship_node_num = 25
extended_friendship_sex_girl_index = []
extended_friendship_sex_boy_index = []
extended_primary_matrix = np.array(data_knecht_friendship.friendship_primary)
for i in range(friendship_joint_times):
    extended_friendship_sex_girl_index = extended_friendship_sex_girl_index + [val+i*friendship_node_num for val in data_knecht_friendship.friendship_sex_girl_index]
    extended_friendship_sex_boy_index = extended_friendship_sex_boy_index + [val+i*friendship_node_num for val in data_knecht_friendship.friendship_sex_boy_index]
    extended_primary_matrix = block_diag(extended_primary_matrix, np.array(data_knecht_friendship.friendship_primary))


#friendship model
def model_netStat_friendship_KHEx(network):
    #directed network
    model = []
    #define model
    model.append(network.statCal_edgeNum())
    model.append(network.statCal_homophily(data_knecht_friendship.friendship_sex_girl_index, joint_model=False, network_seq_length=4)) #girls
    model.append(network.statCal_homophily(data_knecht_friendship.friendship_sex_boy_index, joint_model=False, network_seq_length=4)) #boys
    model.append(network.statCal_heterophily(data_knecht_friendship.friendship_sex_girl_index, 
                                                data_knecht_friendship.friendship_sex_boy_index, joint_model=False, network_seq_length=4))#girls->boys
    model.append(network.statCal_match_matrix(np.array(data_knecht_friendship.friendship_primary), joint_model=False, network_seq_length=4))
    model.append(network.statCal_mutuality())
    model.append(network.statCal_transitiveTies())
    model.append(network.statCal_cyclicalTies())
    return np.array(model)

def model_netStat_friendship_KHEx_jointly(network):
    #directed network
    model = []
    #define model
    model.append(network.statCal_edgeNum())
    
    #extended args
    model.append(network.statCal_homophily(extended_friendship_sex_girl_index)) #girls
    model.append(network.statCal_homophily(extended_friendship_sex_boy_index)) #boys
    model.append(network.statCal_heterophily(extended_friendship_sex_girl_index, 
                                                extended_friendship_sex_boy_index))#girls->boys
    model.append(network.statCal_match_matrix(np.array(extended_primary_matrix)))
    #others
    model.append(network.statCal_mutuality())
    model.append(network.statCal_transitiveTies())
    model.append(network.statCal_cyclicalTies())
    return np.array(model)

# friendship_KHEx_initial_formation_vec = [
#     np.array([0, 0, 0, 0, 0, 0, 0, 0]), 
#     np.array([1, 0, 0, 0, 0, 0, 0, 0]), 
#     np.array([-1, 0, 1, 0, 1, 1, 0, 0]), 
#     np.array([0, -1, -1, 1, 0, 0, 0, 0]),
#     np.array([0, 1, 1, -1, 0, 0, 0, 0]),
#     np.array([0, 1, -1, 0, 1, 0, 0, 0]),
# ]
# friendship_KHEx_initial_dissolution_vec = [
#     np.array([0, 0, 0, 0, 0, 0, 0, 0]), 
#     np.array([1, 0, 0, 0, 0, 0, 0, 0]), 
#     np.array([-1, 0, 1, 0, 1, 1, 0, 0]), 
#     np.array([0, -1, -1, 1, 0, 0, 0, 0]),
#     np.array([0, 1, 1, -1, 0, 0, 0, 0]),
#     np.array([0, 1, -1, 0, 1, 0, 0, 0]),
# ]

# friendship_KHEx_initial_formation_vec_conti = [
#     np.array([-6.045, 1.645, 0.872, -2.631, 2.379, -0.304, 1.348, 0.07]),
#     np.array([-3.774, -1.134, 0.064, -1.832, 1.915, 2.899, 2.516, -1.721]),
#     np.array([-4.052, -0.664, 1.656, -1.113, 0.041, 4.773, 1.805, -1.466]),
#     np.array([-3.251, -1.332, 0.487, -5.079, 0.472, 2.08, 2.127, -0.914]),
#     np.array([-4.001, 0.738, 1.298, -1.56, 2.805, 1.971, 0.971, -0.224]),

# ]
# friendship_KHEx_initial_dissolution_vec_conti = [
#     np.array([-0.735, -0.944, -1.248, -0.457, -0.612, 4.36, 2.146, -1.176]),
#     np.array([-2.335, -1.63, 1.849, -4.225, 1.893, 2.222, 3.258, -1.578]),
#     np.array([-2.121, -1.056, 0.668, -5.03, 4.435, 2.462, 3.142, -1.989]),
#     np.array([1.05, -0.882, -1.203, 3.247, 0.206, 4.707, 2.116, -3.619]),
#     np.array([-2.448, 1.504, 2.493, 0.479, 0.875, 3.914, 0.824, -1.572]),
# ]
friendship_KHEx_initial_formation_vec = [
    np.array([-4.80, 0.51, -0.26, -2.00, 0.39, 2.97, 1.28, -1.03]),
    np.array([-3.33, 0.48, 0.97, -0.35, 0.65, 1.38, 0.88, -0.38]),
    np.array([-4.20, 0.59, -0.00, -0.68, -0.17, 2.10, 1.05, -0.86]),
    np.array([-4.78, 0.32, -1.25, -2.09, 1.23, 2.99, 1.29, -0.94]),
    np.array([-3.77, -1.13, 0.06, -0.83, 0.91, 0.89, 0.51, -0.72]),
]
friendship_KHEx_initial_dissolution_vec = [
    np.array([-1.51, -0.13, 1.79, -0.54, 0.70, 3.34, 1.90, -1.59]),
    np.array([-1.13, 0.12, 1.16, -0.57, 0.45, 2.68, 1.12, -1.01]),
    np.array([-1.05, -0.63, 1.11, -1.09, 0.73, 3.36, 1.86, -1.56]),
    np.array([-1.58, -0.07, 1.93, -0.85, 0.98, 3.53, 1.82, -1.60]),
    np.array([-0.73, -0.94, -1.24, -0.45, 0.61, 2.3, 1.14, -1.17]),
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



def model_netStat_friendship_2hom(network):
    #directed network
    model = []
    #define model
    model.append(network.statCal_edgeNum())
    model.append(network.statCal_homophily(data_knecht_friendship.friendship_sex_girl_index)) #girls
    model.append(network.statCal_homophily(data_knecht_friendship.friendship_sex_boy_index)) #boys
    model.append(network.statCal_match_matrix(np.array(data_knecht_friendship.friendship_primary)))
    model.append(network.statCal_mutuality())
    model.append(network.statCal_transitiveTies())
    model.append(network.statCal_cyclicalTies())
    return np.array(model)


friendship_2hom_initial_formation_vec = [
    np.array([0, 0, 0, 0, 0, 0, 0]), 
    np.array([-1, 0, 0, 0, 0, 0, 0]), 
    np.array([1, 0, 0, 0, 0, 0, 0]), 
    np.array([0, -1, 1, 0, 0, 0, 0]),
    np.array([1, -1, 1, 0, 0, 0, 0]),
    np.array([-1, -1, 1, 0, 0, 0, 0])
]
friendship_2hom_initial_dissolution_vec = [
    np.array([0, 0, 0, 0, 0, 0, 0]), 
    np.array([-1, 0, 0, 0, 0, 0, 0]),
    np.array([1, 0, 0, 0, 0, 0, 0]), 
    np.array([0, -1, 1, 0, 0, 0, 0]),
    np.array([1, -1, 1, 0, 0, 0, 0]),
    np.array([-1, -1, 1, 0, 0, 0, 0])
]


def model_netStat_friendship_2hom_noprisch(network):
    #directed network
    model = []
    #define model
    model.append(network.statCal_edgeNum())
    model.append(network.statCal_homophily(data_knecht_friendship.friendship_sex_girl_index)) #girls
    model.append(network.statCal_homophily(data_knecht_friendship.friendship_sex_boy_index)) #boys
    model.append(network.statCal_mutuality()) #밑 3개 빼자
    model.append(network.statCal_transitiveTies())
    model.append(network.statCal_cyclicalTies())
    return np.array(model)


friendship_2hom_noprisch_initial_formation_vec = [
    np.array([0, 0, 0, 0, 0, 0]),
    np.array([-1, 0, 0, 0, 0, 0]),
    np.array([1, 0, 0, 0, 0, 0]),
    np.array([0, -1, 1, 0, 0, 0]),
    np.array([1, -1, 1, 0, 0, 0]),
    np.array([-1, -1, 1, 0, 0, 0])
]
friendship_2hom_noprisch_initial_dissolution_vec = [
    np.array([0, 0, 0, 0, 0, 0]),
    np.array([-1, 0, 0, 0, 0, 0]),
    np.array([1, 0, 0, 0, 0, 0]),
    np.array([0, -1, 1, 0, 0, 0]),
    np.array([1, -1, 1, 0, 0, 0]),
    np.array([-1, -1, 1, 0, 0, 0])
]

def model_netstat_friendship_nondds(network):
    #directed network
    model = []
    #define model
    model.append(network.statCal_edgeNum())
    model.append(network.statCal_homophily(data_knecht_friendship.friendship_sex_girl_index)) #girls
    model.append(network.statCal_homophily(data_knecht_friendship.friendship_sex_boy_index)) #boys
    return np.array(model)

friendship_nondds_initial_formation_vec = [
    np.array([0,0,0]),
    np.array([1,0,0]),
    np.array([-1,0,0]),
    np.array([0,1,-1]),
    np.array([0,-1,1]),
    np.array([1,1,0])
]

friendship_nondds_initial_dissolution_vec = [
    np.array([0,0,0]),
    np.array([1,0,0]),
    np.array([-1,0,0]),
    np.array([0,1,-1]),
    np.array([0,-1,1]),
    np.array([1,1,0])
]


def model_netstat_friendship_edge1hom(network):
    #directed network
    model = []
    #define model
    model.append(network.statCal_edgeNum())
    model.append(network.statCal_homophily(data_knecht_friendship.friendship_sex_girl_index)) #girls
    return np.array(model)

friendship_edge1hom_initial_formation_vec = [
    np.array([0,0]),
    np.array([1,0]),
    np.array([-1,0]),
    np.array([0,-1]),
    np.array([0,1]),
    np.array([0.5,0.5])
]

friendship_edge1hom_initial_dissolution_vec = [
    np.array([0,0]),
    np.array([1,0]),
    np.array([-1,0]),
    np.array([0,-1]),
    np.array([0,1]),
    np.array([0.5,0.5])
]


def model_netstat_friendship_edge1homMs(network):
    #directed network
    model = []
    #define model
    model.append(network.statCal_edgeNum())
    model.append(network.statCal_homophily(data_knecht_friendship.friendship_sex_girl_index)) #girls
    model.append(network.statCal_mutuality())
    model.append(network.statCal_transitiveTies())
    model.append(network.statCal_cyclicalTies())
    return np.array(model)


friendship_edge1homMs_initial_formation_vec = [
    np.array([0,0,0,0,0]),
    np.array([1,0,0,0,0]),
    np.array([-1,0,0,0,0]),
    np.array([0,1,-1,0,0]),
    np.array([0,-1,1,0,0]),
    np.array([1,1,0,0,0])
]

friendship_edge1homMs_initial_dissolution_vec = [
    np.array([0,0,0,0,0]),
    np.array([1,0,0,0,0]),
    np.array([-1,0,0,0,0]),
    np.array([0,1,-1,0,0]),
    np.array([0,-1,1,0,0]),
    np.array([1,1,0,0,0])
]

#tailorshop model
def model_netStat_tailor_social_edgeGWESP(network): #model_netStat_tailor_social
    #undirected network
    model = []
    model.append(network.statCal_edgeNum())
    model.append(network.statCal_geoWeightedESP(0.25))
    return np.array(model)



tailor_social_edgeGWESP_initial_formation_vec = [
    np.array([-3.03, 1.19]),
    np.array([-2.73, 0.98]),
    np.array([-2.60, 0.90]),
    np.array([-2.69, 0.95]),
    np.array([-3.28, 1.39]),
    np.array([-3.23, 1.35])
]

tailor_social_edgeGWESP_initial_dissolution_vec = [
    np.array([0.05, 0.42]),
    np.array([0.04, 0.44]),
    np.array([-0.19, 0.51]),
    np.array([-0.05, 0.41]),
    np.array([-0.09, 0.40]),
    np.array([0.03, 0.43])
]


def model_netStat_tailor_social_edgeESPDSP(network):
    #undirected network
    model = []
    model.append(network.statCal_edgeNum())
    model.append(network.statCal_geoWeightedESP(0.25))
    model.append(network.statCal_geoWeightedDSP(0.25))
    return np.array(model)

tailor_social_edgeESPDSP_initial_formation_vec = [
    np.array([0,0,0]),
    np.array([0.5,-0.5,-0.5]),
    np.array([-0.5,0.5,-0.5]),
    np.array([-0.5,-0.5,-0.5]),
    np.array([0,-0.5,0.5]),
    np.array([0.5,0,0.5])
]

tailor_social_edgeESPDSP_initial_dissolution_vec = [
    np.array([0,0,0]),
    np.array([0.5,-0.5,-0.5]),
    np.array([-0.5,0.5,-0.5]),
    np.array([-0.5,-0.5,-0.5]),
    np.array([0,-0.5,0.5]),
    np.array([0.5,0,0.5])
]


def model_netStat_tailor_social_edgeDegrESP(network):
    #undirected network
    model = []
    model.append(network.statCal_edgeNum())
    model.append(network.statCal_geoWeightedDegree(0.25))
    model.append(network.statCal_geoWeightedESP(0.25))
    return np.array(model)

tailor_social_edgeDegrESP_initial_formation_vec = [
    np.array([0,0,0]),
    np.array([0.5,-0.5,-0.5]),
    np.array([-0.5,0.5,-0.5]),
    np.array([-0.5,-0.5,-0.5]),
    np.array([0,-0.5,0.5]),
    np.array([0.5,0,0.5])
]

tailor_social_edgeDegrESP_initial_dissolution_vec = [
    np.array([0,0,0]),
    np.array([0.5,-0.5,-0.5]),
    np.array([-0.5,0.5,-0.5]),
    np.array([-0.5,-0.5,-0.5]),
    np.array([0,-0.5,0.5]),
    np.array([0.5,0,0.5])
]


def model_netStat_tailor_social_edgeDegrESPDSP(network):
    #undirected network
    model = []
    model.append(network.statCal_edgeNum())
    model.append(network.statCal_geoWeightedDegree(0.25))
    model.append(network.statCal_geoWeightedESP(0.25))
    model.append(network.statCal_geoWeightedDSP(0.25))
    return np.array(model)


tailor_social_edgeDegrESPDSP_initial_formation_vec = [
    np.array([0,0,0,0]),
    np.array([0.5,-0.5,-0.5,0.5]),
    np.array([-0.5,0.5,-0.5,0.5]),
    np.array([-0.5,-0.5,-0.5,-0.5]),
    np.array([0,-0.5,0.5,-0.5]),
    np.array([0.5,0,0.5,0])
]

tailor_social_edgeDegrESPDSP_initial_dissolution_vec = [
    np.array([0,0,0,0]),
    np.array([0.5,-0.5,-0.5,0.5]),
    np.array([-0.5,0.5,-0.5,0.5]),
    np.array([-0.5,-0.5,-0.5,-0.5]),
    np.array([0,-0.5,0.5,-0.5]),
    np.array([0.5,0,0.5,0])
]

