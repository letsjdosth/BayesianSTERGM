#pragma once
#include <vector>
#include <iostream>
#include <armadillo>

#include "Network.h"


// network statistics
//1. 모델에서/진단에서 호출할 stat이 정해진다면, 
//호출시마다 계산하지 말고 인스턴스 생성시 미리 계산해둔 후 미리 계산해둔 값을 리턴할 것
//(public/private 수정: private는, 리턴대신 class 멤버변수로 값을 저장하도록/ public은, 멤버변수에서 꺼내도록)


int Network::undirected_edgeNum() {
    return accu(netStructure) / 2;
}

int Network::directed_edgeNum() {
    return accu(netStructure);
}


Col<int> Network::undirected_nodeDegree() {
    Col<int> nodeDegree = sum(netStructure, 1);
    return nodeDegree;
}
Col<int> Network::directed_nodeInDegree() {
    Col<int> nodeDegree = sum(netStructure, 0).t();
    return nodeDegree;
}
Col<int> Network::directed_nodeOutDegree() {
    Col<int> nodeDegree = sum(netStructure, 1);
    return nodeDegree;
}

Col<int> Network::undirected_nodeDegreeDist() {
    Col<int> degreeDistRes;
    degreeDistRes.zeros(n_Node + 1);

    Col<int> netNodeDegree = undirected_nodeDegree();
    Col<int>::iterator it = netNodeDegree.begin();
    Col<int>::iterator it_end = netNodeDegree.end();
    for (; it != it_end; ++it) {
        degreeDistRes(*it)++;
    }
    return degreeDistRes;
}

Col<int> Network::directed_nodeInDegreeDist() {
    Col<int> degreeDistRes;
    degreeDistRes.zeros(n_Node + 1);

    Col<int> netNodeDegree = directed_nodeInDegree();
    Col<int>::iterator it = netNodeDegree.begin();
    Col<int>::iterator it_end = netNodeDegree.end();
    for (; it != it_end; ++it) {
        degreeDistRes(*it)++;
    }
    return degreeDistRes;
}

Col<int> Network::directed_nodeOutDegreeDist() {
    Col<int> degreeDistRes;
    degreeDistRes.zeros(n_Node + 1);

    Col<int> netNodeDegree = directed_nodeOutDegree();
    Col<int>::iterator it = netNodeDegree.begin();
    Col<int>::iterator it_end = netNodeDegree.end();
    for (; it != it_end; ++it) {
        degreeDistRes(*it)++;
    }
    return degreeDistRes;
}


double Network::undirected_geoWeightedNodeDegree(double tau) {
    double gwd=0;
    double d = 1 - exp(-tau);
    double s = 1;
    Col<int> degreeDist = undirected_nodeDegreeDist();

    for (int i = 1; i < n_Node; i++) {
        s *= d;
        gwd += ((1 - s) * degreeDist(i));
    }
    gwd *= exp(tau);
    return gwd;
}

Mat<int> Network::undirected_edgewiseSharedPartner() {
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
Mat<int> Network::directed_edgewiseSharedPartner() {
    //directed
    Mat<int> ESP;
    ESP.zeros(n_Node, n_Node);

    for (int c = 0; c < n_Node; c++) { //c->r
        for (int r = 0; r < n_Node; r++) {
            if (netStructure(c, r) == 1) {
                for (int k = 0; k < n_Node; k++) {
                    if (netStructure(c, k) == 1 && netStructure(r, k) == 1) {
                        ESP(c, r) += 1;
                    }
                }
            }
        }
    }
    return ESP;
}

Col<int> Network::undirected_edgewiseSharedPartnerDist() {
    //undirected
    Col<int> ESPDistRes;
    ESPDistRes.zeros(n_Node - 1); // 0 ~ (n-2);
    Mat<int> ESP = undirected_edgewiseSharedPartner();
    for (int c = 1;c < n_Node; c++) {
        for (int r = 0; r < c; r++) {
            int i = ESP(r, c);
            ESPDistRes(i) += 1;
        }
    }
    ESPDistRes(0) = 0;
    ESPDistRes(0) = n_Edge - sum(ESPDistRes);

    return ESPDistRes;
}


Col<int> Network::directed_edgewiseSharedPartnerDist() {
    //undirected
    Col<int> ESPDistRes;
    ESPDistRes.zeros(n_Node - 1); // 0 ~ (n-2);
    Mat<int> ESP = directed_edgewiseSharedPartner();
    for (int c = 0;c < n_Node; c++) {
        for (int r = 0; r < n_Node; r++) {
            if (c != r) {
                int i = ESP(c, r);
                ESPDistRes(i) += 1;
            }
        }
    }
    ESPDistRes(0) = 0;
    ESPDistRes(0) = n_Edge - sum(ESPDistRes);
    return ESPDistRes;
}

double Network::undirected_geoWeightedESP(double tau) {
    double GWESP = 0;
    double d = 1 - exp(-tau);
    double s = 1;
    Col<int> ESPdist = undirected_edgewiseSharedPartnerDist();
    for (int i = 1; i < (n_Node - 1); i++) {
        s *= d;
        GWESP += (1 - s) * ESPdist(i);
    }
    GWESP *= exp(tau);
    return GWESP;
}


double Network::directed_geoWeightedESP(double tau) {
    double GWESP = 0;
    double d = 1 - exp(-tau);
    double s = 1;
    Col<int> ESPdist = directed_edgewiseSharedPartnerDist();
    for (int i = 1; i < (n_Node - 1); i++) {
        s *= d;
        GWESP += (1 - s) * ESPdist(i);
    }
    GWESP *= exp(tau);
    return GWESP;
}


Mat<int> Network::undirected_dyadwiseSharedPartner() {
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
Col<int> Network::undirected_dyadwiseSharedPartnerDist() {
    //undirected
    Col<int> DSPDistRes;
    DSPDistRes.zeros(n_Node - 1); // 0 ~ (n-2);
    Mat<int> ESP = undirected_dyadwiseSharedPartner();
    for (int c = 1;c < n_Node; c++) {
        for (int r = 0; r < c; r++) {
            int i = ESP(r, c);
            DSPDistRes(i) += 1;
        }
    }
    return DSPDistRes;

}
double Network::undirected_geoWeightedDSP(double tau) {
    double GWDSP = 0;
    double d = 1 - exp(-tau);
    double s = 1;
    Col<int> ESPdist = undirected_dyadwiseSharedPartnerDist();
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

Col<int> Network::undirected_k_starDist() {
    Col<int> k_starDistRes;
    k_starDistRes.zeros(n_Node);
    // k_starDistRes(0) = 0; //undefined
    k_starDistRes(1) = n_Edge;
    for (int k = 2; k < n_Node; k++) {
        for (int i = 1; i < n_Node; i++) {
            k_starDistRes(k) += (nCr(i, k) * undirected_nodeDegreeDist()(i));
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

Col<int> Network::undirected_n_triangleDist() {
    Col<int> triangleDistRes;
    triangleDistRes.zeros(n_Node-1);
    
    Col<int> ESPdist = undirected_edgewiseSharedPartnerDist();
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
    n_Node = netStructure.n_rows;
    if (isDirectedInput) {
        n_Edge = directed_edgeNum();
    }
    else {
        n_Edge = undirected_edgeNum();
    }
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

bool Network::is_directed_graph() {
    return isDirected;
}

int Network::get_n_Node() {
    return n_Node;
}
int Network::get_n_Edge() {
    return n_Edge;
}
int Network::get_undirected_nodeDegreeDist(int degree) {
    return undirected_nodeDegreeDist()(degree);
}
Col<int> Network::get_undirected_nodeDegreeDist() {
    return undirected_nodeDegreeDist();
}
double Network::get_undirected_geoWeightedNodeDegree(double tau) {
    return undirected_geoWeightedNodeDegree(tau);
}
int Network::get_directed_nodeInDegreeDist(int degree) {
    return directed_nodeInDegreeDist()(degree);
}
Col<int> Network::get_directed_nodeInDegreeDist() {
    return directed_nodeInDegreeDist();
}
//double Network::get_directed_geoWeightedNodeInDegree(double tau) {
//    return directed_geoWeightedNodeInDegree(tau);
//}
int Network::get_directed_nodeOutDegreeDist(int degree) {
    return directed_nodeOutDegreeDist()(degree);
}
Col<int> Network::get_directed_nodeOutDegreeDist() {
    return directed_nodeOutDegreeDist();
}
//double Network::get_directed_geoWeightedNodeOutDegree(double tau) {
//    return directed_geoWeightedNodeOutDegree(tau);
//}



int Network::get_undirected_edgewiseSharedPartnerDist(int degree) {
    return undirected_edgewiseSharedPartnerDist()(degree);
}
Col<int> Network::get_undirected_edgewiseSharedPartnerDist() {
    return undirected_edgewiseSharedPartnerDist();
}
double Network::get_undirected_geoWeightedESP(double tau) {
    return undirected_geoWeightedESP(tau);
}
int Network::get_directed_edgewiseSharedPartnerDist(int degree) {
    return directed_edgewiseSharedPartnerDist()(degree);
}
Col<int> Network::get_directed_edgewiseSharedPartnerDist() {
    return directed_edgewiseSharedPartnerDist();
}
double Network::get_directed_geoWeightedESP(double tau) {
    return directed_geoWeightedESP(tau);
}



int Network::get_undirected_dyadwiseSharedPartnerDist(int degree) {
    return undirected_dyadwiseSharedPartnerDist()(degree);
}
Col<int> Network::get_undirected_dyadwiseSharedPartnerDist() {
    return undirected_dyadwiseSharedPartnerDist();
}
double Network::get_undirected_geoWeightedDSP(double tau) {
    return undirected_geoWeightedDSP(tau);
}
int Network::get_undirected_k_starDist(int degree_k) {
    return undirected_k_starDist()(degree_k);
}
Col<int> Network::get_undirected_k_starDist() {
    return undirected_k_starDist();
}
int Network::get_undirected_triangleDist(int degree) {
    return undirected_n_triangleDist()(degree);
}
Col<int> Network::get_undirected_triangleDist() {
    return undirected_n_triangleDist();
}

void Network::undirected_printSummary() {
    cout << "===========================" << endl;
    cout << "~undirected graph summary~" << endl;
    cout << netStructure << endl;
    cout << "isDirected : " << isDirected << endl;
    cout << "n_node : " << n_Node << endl;
    cout << "n_edge : " << n_Edge << endl;
    cout << "node degree : " << undirected_nodeDegree().t() << endl;
    cout << "degree_dist : " << undirected_nodeDegreeDist().t() << endl;
    cout << "geometrically weighted node degree : " << undirected_geoWeightedNodeDegree(0.5) << endl;
    // cout << "triangle_dist : " << undirected_n_triangleDist().t() << endl;
    // cout << "kstar_dist : " << undirected_k_starDist().t() << endl;
    cout << "Edgewise Shared Partnership : \n" << undirected_edgewiseSharedPartner() << endl;
    cout << "Edgewise Shared Partnership Dist : " << undirected_edgewiseSharedPartnerDist().t() << endl;
    cout << "geometrically weighted Edgewise Shared Partnership : " << undirected_geoWeightedESP(0.3) << endl;
    cout << "Dyadwise Shared Partnership : \n" << undirected_dyadwiseSharedPartner() << endl;
    cout << "Dyadwise Shared Partnership Dist : " << undirected_dyadwiseSharedPartnerDist().t() << endl;
    cout << "geometrically weighted Dyadwise Shared Partnership : " << undirected_geoWeightedDSP(0.5) << endl;

    cout << "===========================" << endl;
}


void Network::directed_printSummary() {
    cout << "===========================" << endl;
    cout << "~directed graph summary~" << endl;
    cout << netStructure << endl;
    cout << "isDirected : " << isDirected << endl;
    cout << "n_node : " << n_Node << endl;
    cout << "n_edge : " << n_Edge << endl;
    
    cout << "node_InDegree : " << directed_nodeInDegree().t() << endl;
    cout << "InDegree_dist : " << directed_nodeInDegreeDist().t() << endl;
    // cout << "geometrically weighted node InDegree : " << undirected_geoWeightedNodeDegree(0.3) << endl;
    cout << "node_OutDegree : " << directed_nodeOutDegree().t() << endl;
    cout << "OutDegree_dist : " << directed_nodeOutDegreeDist().t() << endl;
    // cout << "geometrically weighted node OutDegree : " << undirected_geoWeightedNodeDegree(0.3) << endl;


    // cout << "triangle_dist : " << undirected_n_triangleDist().t() << endl;
    // cout << "kstar_dist : " << undirected_k_starDist().t() << endl;
    cout << "Edgewise Shared Partnership : \n" << directed_edgewiseSharedPartner() << endl;
    cout << "Edgewise Shared Partnership Dist : " << directed_edgewiseSharedPartnerDist().t() << endl;
    cout << "geometrically weighted Edgewise Shared Partnership : " << directed_geoWeightedESP(0.3) << endl;
    // cout << "Dyadwise Shared Partnership : \n" << undirected_dyadwiseSharedPartner() << endl;
    // cout << "Dyadwise Shared Partnership Dist : " << undirected_dyadwiseSharedPartnerDist().t() << endl;
    // cout << "geometrically weighted Dyadwise Shared Partnership : " << undirected_geoWeightedDSP(0.3) << endl;

    cout << "===========================" << endl;
}

