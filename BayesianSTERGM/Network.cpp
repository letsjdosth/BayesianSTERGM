#pragma once
#include <vector>
#include <iostream>
#include <armadillo>

#include "Network.h"


void Network::updateNetworkInfo() {
    n_Node = netStructure.n_rows;
    n_Edge = edgeNum();
    dist_nodeDegree = nodeDegreeDist();
}

int Network::Network::edgeNum() {
    if (isDirected) {
        return accu(netStructure);
    }
    else {
        return accu(netStructure) / 2;
    }
}

// network statistics
//1. 모델에서/진단에서 호출할 stat이 정해진다면, 
//호출시마다 계산하지 말고 인스턴스 생성시 미리 계산해둔 후 미리 계산해둔 값을 리턴할 것
//(public/private 수정: private는, 리턴대신 class 멤버변수로 값을 저장하도록/ public은, 멤버변수에서 꺼내도록)

Col<int> Network::nodeDegree() {
    Col<int> nodeDegree = sum(netStructure, 1);
    return nodeDegree;
}

Col<int> Network::nodeDegreeDist() {
    Col<int> degreeDistRes;
    degreeDistRes.zeros(n_Node + 1);

    Col<int> netNodeDegree = nodeDegree();
    Col<int>::iterator it = netNodeDegree.begin();
    Col<int>::iterator it_end = netNodeDegree.end();
    for (; it != it_end; ++it) {
        degreeDistRes(*it)++;
    }
    return degreeDistRes;
}

double Network::geoWeightedNodeDegree(double tau) {
    double gwd=0;
    double d = 1 - exp(-tau);
    double s = 1;
    Col<int> degreeDist = nodeDegreeDist();

    for (int i = 1; i < n_Node; i++) {
        s *= d;
        gwd += ((1 - s) * degreeDist(i));
    }
    gwd *= exp(tau);
    return gwd;
}

Mat<int> Network::edgewiseSharedPartner() {
    //undirected
    Mat<int> ESP;
    ESP.zeros(n_Node, n_Node);

    for (int c = 1; c < n_Node; c++) {
        for (int r = 0; r < c; r++) {
            if (netStructure(r, c) == 1) {
                for (int k = 0; k < n_Node; k++) {
                    if (netStructure(r, k) == 1 && netStructure(k, c) == 1) {
                        ESP(c, r) += 1;
                        ESP(r, c) += 1;
                    }
                }
            }
        }
    }
    return ESP;
}

Col<int> Network::edgewiseSharedPartnerDist() {
    //undirected
    Col<int> ESPDistRes;
    ESPDistRes.zeros(n_Node - 1); // 0 ~ (n-2);
    Mat<int> ESP = edgewiseSharedPartner();
    for (int c = 1;c < n_Node; c++) {
        for (int r = 0; r < c; r++) {
            int i = ESP(r, c);
            ESPDistRes(i) += 1;
        }
    }
    ESPDistRes(0) = 0;
    ESPDistRes(0) = n_Node - sum(ESPDistRes);

    return ESPDistRes;
}

double Network::geoWeightedESP(double tau) {
    double GWESP = 0;
    double d = 1 - exp(-tau);
    double s = 1;
    Col<int> ESPdist = edgewiseSharedPartnerDist();
    for (int i = 1; i < (n_Node - 1); i++) {
        s *= d;
        GWESP += (1 - s) * ESPdist(i);
    }
    GWESP *= exp(tau);
    return GWESP;
}

Mat<int> Network::dyadwiseSharedPartner() {
    //undirected
    Mat<int> DSP;
    DSP.zeros(n_Node, n_Node);

    for (int c = 1; c < n_Node; c++) {
        for (int r = 0; r < c; r++) {
            for (int k = 0; k < n_Node; k++) {
                if (netStructure(r, k) == 1 && netStructure(k, c) == 1) {
                    DSP(c, r) += 1;
                    DSP(r, c) += 1;
                }
            }
        }
    }
    return DSP;
}
Col<int> Network::dyadwiseSharedPartnerDist() {
    //undirected
    Col<int> DSPDistRes;
    DSPDistRes.zeros(n_Node - 1); // 0 ~ (n-2);
    Mat<int> ESP = dyadwiseSharedPartner();
    for (int c = 1;c < n_Node; c++) {
        for (int r = 0; r < c; r++) {
            int i = ESP(r, c);
            DSPDistRes(i) += 1;
        }
    }
    return DSPDistRes;

}
double Network::geoWeightedDSP(double tau) {
    double GWDSP = 0;
    double d = 1 - exp(-tau);
    double s = 1;
    Col<int> ESPdist = dyadwiseSharedPartnerDist();
    for (int i = 1; i < (n_Node - 1); i++) {
        s *= d;
        GWDSP += (1 - s) * ESPdist(i);
    }
    GWDSP *= exp(tau);
    return GWDSP;
}



int Network::fact(int n) {
    int res = 1;
    for (int i = 2; i <= n; i++) {
        res = res * i;
    }
    return res;
}

int Network::nCr(int n, int r) {
    return fact(n) / (fact(r) * fact(n - r));
}

Col<int> Network::k_starDist() {
    Col<int> k_starDistRes;
    k_starDistRes.zeros(n_Node);
    // k_starDistRes(0) = 0; //undefined
    k_starDistRes(1) = n_Edge;
    for (int k = 2; k < n_Node; k++) {
        for (int i = 1; i < n_Node; i++) {
            k_starDistRes(k) += (nCr(i, k) * dist_nodeDegree(i));
        }
    }
    return k_starDistRes;
}

//int Network::n_triangle() {
//    if (isDirected) {
//        int count_Triangle = 0;
//
//        for (int i = 0; i < n_Node; i++)
//        {
//            for (int j = 0; j < n_Node; j++)
//            {
//                for (int k = 0; k < n_Node; k++)
//                {
//                    // check the triplet if 
//                    // it satisfies the condition 
//                    if (netStructure(i, j) && netStructure(j, k) && netStructure(k, i))
//                        count_Triangle++;
//                }
//            }
//        }
//    }
//    else {
//        return trace((netStructure * netStructure) * netStructure) / 6;
//    }
//}

Col<int> Network::n_triangleDist() {
    Col<int> triangleDistRes;
    triangleDistRes.zeros(n_Node-1);
    
    Col<int> ESPdist = edgewiseSharedPartnerDist();
    // triangleDistRes(0) = 0; //undefined
    // 1-order
    for (int i = 0; i < ESPdist.size(); i++) {
        triangleDistRes(1) += (i * ESPdist(i));
    }
    triangleDistRes(1) /= 3;

    // 2-order ~
    for (int k = 2; k < (n_Node - 1); k++) {
        for (int i = 1; i < (n_Node - 1); i++) {
            triangleDistRes(k) += nCr(i, k) * ESPdist(i);
        }
    }

    return triangleDistRes;
}


Network::Network(Mat<int> inputNet, bool isDirectedInput) {
    netStructure = inputNet;
    isDirected = isDirectedInput;
    if (!(isDirectedInput) && !(inputNet.is_symmetric())) {
        cout << "The input structure is not symmetric, but the network is set as undirected graph." << endl;
    }
    updateNetworkInfo();
}

Network::Network() {
    //빈 생성자
}

Mat<int> Network::get_netStructure() {
    return netStructure;
}

// get_networkStatistics
//1. 모델에서/진단에서 호출할 stat이 정해진다면, 
//호출시마다 계산하지 말고 인스턴스 생성시 미리 계산해둔 후 미리 계산해둔 값을 리턴할 것
//(public/private 수정: private는, 리턴대신 class 멤버변수로 값을 저장하도록/ public은, 멤버변수에서 꺼내도록)
//2. 나중에 degree 범위 조건 추가하고 안맞으면 예외 던질 것

int Network::get_n_Node() {
    return n_Node;
}
int Network::get_n_Edge() {
    return n_Edge;
}
int Network::get_nodeDegreeDist(int degree) {
    return nodeDegreeDist()[degree];
}
Col<int> Network::get_nodeDegreeDist() {
    return nodeDegreeDist();
}
double Network::get_geoWeightedNodeDegree(double tau) {
    return geoWeightedNodeDegree(tau);
}
int Network::get_edgewiseSharedPartnerDist(int degree) {
    return edgewiseSharedPartnerDist()[degree];
}
Col<int> Network::get_edgewiseSharedPartnerDist() {
    return edgewiseSharedPartnerDist();
}
double Network::get_geoWeightedESP(double tau) {
    return geoWeightedESP(tau);
}
int Network::get_dyadwiseSharedPartnerDist(int degree) {
    return dyadwiseSharedPartnerDist()[degree];
}
Col<int> Network::get_dyadwiseSharedPartnerDist() {
    return dyadwiseSharedPartnerDist();
}
double Network::get_geoWeightedDSP(double tau) {
    return geoWeightedDSP(tau);
}
int Network::get_k_starDist(int degree_k) {
    return k_starDist()[degree_k];
}
Col<int> Network::get_k_starDist() {
    return k_starDist();
}
int Network::get_triangleDist(int degree) {
    return n_triangleDist()[degree];
}
Col<int> Network::get_triangleDist() {
    return n_triangleDist();
}




void Network::printSummary() {
    cout << "===========================" << endl;
    cout << netStructure << endl;
    cout << "isDirected : " << isDirected << endl;
    cout << "n_node : " << n_Node << endl;
    cout << "n_edge : " << n_Edge << endl;
    cout << "node degree : " << dist_nodeDegree.t() << endl;
    cout << "degree_dist : " << nodeDegreeDist().t() << endl;
    cout << "geometrically weighted node degree : " << geoWeightedNodeDegree(0.3) << endl;
    cout << "triangle_dist : " << n_triangleDist().t() << endl;
    cout << "kstar_dist : " << k_starDist().t() << endl;
    cout << "Edgewise Shared Partnership : \n" << edgewiseSharedPartner() << endl;
    cout << "Edgewise Shared Partnership Dist : " << edgewiseSharedPartnerDist().t() << endl;
    cout << "geometrically weighted Edgewise Shared Partnership : " << geoWeightedESP(0.3) << endl;
    cout << "Dyadwise Shared Partnership : \n" << dyadwiseSharedPartner() << endl;
    cout << "Dyadwise Shared Partnership Dist : " << dyadwiseSharedPartnerDist().t() << endl;
    cout << "geometrically weighted Dyadwise Shared Partnership : " << geoWeightedDSP(0.3) << endl;

    cout << "===========================" << endl;
}

