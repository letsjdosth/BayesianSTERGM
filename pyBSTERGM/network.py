import numpy as np


class UndirectedNetwork:
    node_num = 0
    structure = np.array(0)
    stat_nodeDegree = np.array(0)

    def __init__(self, structure):
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
    print(test_net.statCal_nodeDegreeDist())
    print(test_net.statCal_edgeNum())
    # print(test_net.statCal_EdgewiseSharedPartner())
    print(test_net.statCal_EdgewiseSharedPartnerDist())
    print(test_net.statCal_geoWeightedESP()) #5.393469 (R과 cross check 완료)