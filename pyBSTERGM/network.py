import numpy as np


class UndirectedNetwork:
    def __init__(self, structure):
        #variables
        self.node_num = 0
        self.structure = np.array(0)
        self.stat_nodeDegree = np.array(0)

        #initialize
        if not np.allclose(structure, structure.T, rtol=1e-05, atol=1e-08):
            raise ValueError("The structure is not symmetric matrix.")
        
        self.structure = structure
        self.node_num = structure.shape[0]    

    def __str__(self):
        string_val = "<network.UndirectedNetwork object>\n" + self.structure.__str__()
        return string_val

    def statCal_edgeNum(self):
        return np.sum(self.structure)/2    
    def statCal_nodeDegree(self):
        return np.sum(self.structure, axis=1) #rowsum
    def statCal_nodeDegreeDist(self):
        node_degree_dist = np.zeros(self.node_num)
        node_degree = self.statCal_nodeDegree()
        for i in node_degree:
            node_degree_dist[i] += 1
        return node_degree_dist
    def statCal_EdgewiseSharedPartner(self):
        ESP = np.zeros((self.node_num, self.node_num))
        for row_ind in range(1, self.node_num):
            for col_ind in range(row_ind):
                if self.structure[row_ind, col_ind] == 1:
                    ESP[row_ind, col_ind] = np.dot(self.structure[row_ind,:].T, self.structure[col_ind,:])
                    ESP[col_ind, row_ind] = ESP[row_ind, col_ind]
        return ESP
    def statCal_EdgewiseSharedPartnerDist(self):
        fullESP= self.statCal_EdgewiseSharedPartner()
        ESP_dist = np.zeros(self.node_num-1)
        for row_ind in range(1, self.node_num):
            for col_ind in range(row_ind):
                ESP_dist[int(fullESP[row_ind,col_ind])] += 1
        ESP_dist[0] = self.statCal_edgeNum() - np.sum(ESP_dist[1:])
        return ESP_dist
    def statCal_geoWeightedESP(self, tau=0.5):
        val = 0
        ESPdist = self.statCal_EdgewiseSharedPartnerDist()
        for k in range(1, self.node_num - 1):
            nested = (-np.expm1(-tau))**k
            val += (1 - nested) * ESPdist[k] * np.exp(tau)
        return val

    def statCal_MinGeodesic(self):
        # Dijkstra algorithm
        def dijkstra(self,source):
            from math import inf
            visited_node = [False for _ in range(self.node_num)]
            dist_vec = [inf for i in range(self.node_num)]
            prev = [None for i in range(self.node_num)]
            
            dist_vec[source]=0

            while not all(visited_node):
                unvisited = [(i, dist) for i, dist in enumerate(dist_vec) if visited_node[i]==False]
                chosen_node = min(unvisited, key=lambda tup:tup[1])[0]

                visited_node[chosen_node] = True
                neighbors = [i for i, val in enumerate(self.structure[chosen_node,:]) if val==1]
                for nb_node in neighbors:
                    alt = dist_vec[chosen_node] + 1 #length = 1
                    if alt < dist_vec[nb_node]:
                        dist_vec[nb_node] = alt
                        prev[nb_node] = chosen_node

            return (dist_vec, prev)
        min_geodesic_vec = []
        for i in range(self.node_num):
            dist, _ = dijkstra(self, i)
            min_geodesic_vec.append(dist)
        return min_geodesic_vec
    
    def statCal_MinGeodesicDist(self):
        from math import inf
        min_geodesic = self.statCal_MinGeodesic()
        distrib_vec = [0 for _ in range(self.node_num+1)]
        for from_source_node in min_geodesic:
            for val in from_source_node:
                if val is inf:
                    distrib_vec[-1] += 1
                else:
                    distrib_vec[val] += 1
        distrib_vec[0] = 0
        undirected_distrib_vec = [val/2 for val in distrib_vec]
        return undirected_distrib_vec[1:] #dist0 = node itself




class DirectedNetwork:
    def __init__(self, structure):
        #variables
        self.node_num = 0
        self.structure = np.array(0)
        self.stat_nodeDegree = np.array(0)

        #initialize
        self.structure = structure
        self.node_num = structure.shape[0]    

    def __str__(self):
        string_val = "<network.DirectedNetwork object>\n" + self.structure.__str__()
        return string_val

    def statCal_edgeNum(self):
        return np.sum(self.structure)
    def statCal_nodeOutDegree(self):
        return np.sum(self.structure, axis=1) #rowsum
    def statCal_nodeInDegree(self):
        return np.sum(self.structure, axis=0) #rowsum
    
    def statCal_nodeOutDegreeDist(self):
        node_degree_dist = np.zeros(self.node_num)
        node_degree = self.statCal_nodeOutDegree()
        for i in node_degree:
            node_degree_dist[i] += 1
        return node_degree_dist
    def statCal_nodeInDegreeDist(self):
        node_degree_dist = np.zeros(self.node_num)
        node_degree = self.statCal_nodeInDegree()
        for i in node_degree:
            node_degree_dist[i] += 1
        return node_degree_dist
    
    def statCal_EdgewiseSharedPartner(self):
        ESP = np.zeros((self.node_num, self.node_num))
        for row_ind in range(self.node_num):
            for col_ind in range(self.node_num):
                if self.structure[row_ind, col_ind] == 1:
                    ESP[row_ind, col_ind] = np.dot(self.structure[row_ind,:].T, self.structure[col_ind,:])
        return ESP
    def statCal_EdgewiseSharedPartnerDist(self):
        fullESP= self.statCal_EdgewiseSharedPartner()
        ESP_dist = np.zeros(self.node_num-1)
        for row_ind in range(self.node_num):
            for col_ind in range(self.node_num):
                ESP_dist[int(fullESP[row_ind,col_ind])] += 1
        ESP_dist[0] = self.statCal_edgeNum() - np.sum(ESP_dist[1:])
        return ESP_dist
    def statCal_geoWeightedESP(self, tau=0.5):
        val = 0
        ESPdist = self.statCal_EdgewiseSharedPartnerDist()
        for k in range(1, self.node_num - 1):
            nested = (-np.expm1(-tau))**k
            val += (1 - nested) * ESPdist[k] * np.exp(tau)
        return val
    
    
    def statCal_MinGeodesic(self):
        # Dijkstra algorithm
        def dijkstra(self,source):
            from math import inf
            visited_node = [False for _ in range(self.node_num)]
            dist_vec = [inf for i in range(self.node_num)]
            prev = [None for i in range(self.node_num)]
            
            dist_vec[source]=0

            while not all(visited_node):
                unvisited = [(i, dist) for i, dist in enumerate(dist_vec) if visited_node[i]==False]
                chosen_node = min(unvisited, key=lambda tup:tup[1])[0]

                visited_node[chosen_node] = True
                neighbors = [i for i, val in enumerate(self.structure[chosen_node,:]) if val==1]
                for nb_node in neighbors:
                    alt = dist_vec[chosen_node] + 1 #length = 1
                    if alt < dist_vec[nb_node]:
                        dist_vec[nb_node] = alt
                        prev[nb_node] = chosen_node

            return (dist_vec, prev)
        min_geodesic_vec = []
        for i in range(self.node_num):
            dist, _ = dijkstra(self, i)
            min_geodesic_vec.append(dist)
        return min_geodesic_vec
    
    def statCal_MinGeodesicDist(self):
        from math import inf
        min_geodesic = self.statCal_MinGeodesic()
        distrib_vec = [0 for _ in range(self.node_num + 1)]
        # print(min_geodesic)
        for from_source_node in min_geodesic:
            for val in from_source_node:
                if val is inf:
                    distrib_vec[-1] += 1
                else:
                    distrib_vec[val] += 1
        distrib_vec[0] = 0
        return distrib_vec[1:] #dist0 = node itself
    
    def statCal_mutuality(self):
        mutual = 0
        for i in range(self.node_num):
            for j in range(i):
                if self.structure[i,j]==1 and self.structure[j,i]==1:
                    mutual +=1
        return mutual

    def statCal_existTwoPath(self):
        existance = np.zeros((self.node_num, self.node_num))
        for start_node in range(self.node_num):
            for end_node in range(self.node_num):
                if start_node == end_node:
                    pass
                else:
                    idx_set = [i for i in range(self.node_num) if i != start_node and i != end_node]
                    # print(start_node, end_node, idx_set)
                    for inter_node in idx_set:
                        if self.structure[start_node, inter_node]==1 and self.structure[inter_node,end_node]==1:
                            existance[start_node, end_node] += 1
        return existance

    def statCal_transitiveTriples(self):
        result = 0
        twoPath = self.statCal_existTwoPath()
        for start_node in range(self.node_num):
            for end_node in range(self.node_num):
                if self.structure[start_node, end_node]==1 and twoPath[start_node, end_node]>0:
                    result += twoPath[start_node, end_node]
        return result/3
    
    def statCal_transitiveTies(self):
        result = 0
        twoPath = self.statCal_existTwoPath()
        for start_node in range(self.node_num):
            for end_node in range(self.node_num):
                if self.structure[start_node, end_node]==1 and twoPath[start_node, end_node]>0:
                    result += 1
        return result

    def statCal_cyclicTriples(self):
        result = 0
        twoPath = self.statCal_existTwoPath()
        for start_node in range(self.node_num):
            for end_node in range(self.node_num):
                if twoPath[start_node, end_node]>1 and self.structure[end_node, start_node]==1 :
                    result += twoPath[start_node, end_node]
        return result/3
    
    def statCal_cyclicalTies(self):
        result = 0
        twoPath = self.statCal_existTwoPath()
        for start_node in range(self.node_num):
            for end_node in range(self.node_num):
                if twoPath[start_node, end_node]>1 and self.structure[end_node, start_node]==1 :
                    result += 1
        return result


# class InterNetwork_Statistics:
#     @classmethod
#     def stability(cls, network_t1, network_t2):
#         #number of edges not changed from t1 to t2
#         network_

#     @classmethod
#     def reciprocity(cls, network_t1, network_t2):
#         #number of edges that become reversed from t1 to t2

#     @classmethod
#     def transitivity(cls, network_t1, network_t2):
#         #number of pair of nodes that have dist 2 at t1 but dist 1 at t2




if __name__ == "__main__":
    test_structure = np.array(
        [[0,1,1,0,0],
        [1,0,1,1,0],
        [1,1,0,1,0],
        [0,1,1,0,1],
        [0,0,0,1,0]]
        )
    test_net = UndirectedNetwork(test_structure)
    # print(test_net.node_num, test_net.structure)
    # print(test_net.statCal_nodeDegree())
    print(test_net.statCal_nodeDegreeDist()) #true: 0,1,1,3,0
    print(test_net.statCal_edgeNum()) #true: 6
    # print(test_net.statCal_EdgewiseSharedPartner())
    print(test_net.statCal_EdgewiseSharedPartnerDist()) #true: 1,4,1,0
    print(test_net.statCal_geoWeightedESP()) #true: 5.393469 (R과 cross check 완료)
    print(test_net.statCal_MinGeodesic())
    print(test_net.statCal_MinGeodesicDist())

    test_structure_2 = np.array(
        [
        [0,1,1,0,0],
        [1,0,0,0,1],
        [1,1,0,1,0],
        [0,0,0,0,1],
        [1,0,0,1,0]]
    )
    test_net_2 = DirectedNetwork(test_structure_2)
    print(test_net_2.statCal_edgeNum())
    print(test_net_2.statCal_nodeOutDegreeDist()) #true [0,1,3,1,0]
    print(test_net_2.statCal_nodeInDegreeDist()) #true [0,1,3,1,0]
    print(test_net_2.statCal_EdgewiseSharedPartnerDist()) #true [6,4,0,0]
    print(test_net_2.statCal_geoWeightedESP(0.5)) #true 4.0
    print(test_net_2.statCal_MinGeodesic())
    print(test_net_2.statCal_MinGeodesicDist())
    print(test_net_2.statCal_mutuality()) #true 3

    print(test_net_2.statCal_existTwoPath())
    print(test_net_2.statCal_transitiveTies())#true 4
    print(test_net_2.statCal_cyclicalTies()) #true 6

    print(test_net_2.statCal_transitiveTriples())#true 4
    print(test_net_2.statCal_cyclicTriples())#true 2