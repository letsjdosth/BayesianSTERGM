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

//Col<int> Network::sharedPartnerDist() {
//
//}

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
    k_starDistRes(0) = 0; //undefined
    k_starDistRes(1) = n_Edge;
    for (int k = 2; k < n_Node; k++) {
        for (int i = 1; i < n_Node; i++) {
            k_starDistRes(k) += (nCr(i, k) * dist_nodeDegree(i));
        }
    }
    return k_starDistRes;
}

int Network::n_triangle() {
    if (isDirected) {
        int count_Triangle = 0;

        for (int i = 0; i < n_Node; i++)
        {
            for (int j = 0; j < n_Node; j++)
            {
                for (int k = 0; k < n_Node; k++)
                {
                    // check the triplet if 
                    // it satisfies the condition 
                    if (netStructure(i, j) && netStructure(j, k) && netStructure(k, i))
                        count_Triangle++;
                }
            }
        }
    }
    else {
        return trace((netStructure * netStructure) * netStructure) / 6;
    }
}


Network::Network(Mat<int> inputNet, bool isDirectedInput) {
    netStructure = inputNet;
    isDirected = isDirectedInput;
    updateNetworkInfo();
}

Network::Network() {
    //ºó »ý¼ºÀÚ
}

int Network::get_n_Node() {
    return n_Node;
}

int Network::get_n_Edge() {
    return n_Edge;
}

Mat<int> Network::get_netStructure() {
    return netStructure;
}

void Network::printSummary() {
    cout << "===========================" << endl;
    cout << netStructure << endl;
    cout << "isDirected :" << isDirected << endl;
    cout << "n_node :" << n_Node << endl;
    cout << "n_edge :" << n_Edge << endl;
    cout << "node degree : " << dist_nodeDegree.t() << endl;
    cout << "degree_dist : " << nodeDegreeDist().t() << endl;
    cout << "triangle : " << n_triangle() << endl;
    cout << "kstar_dist :" << k_starDist().t() << endl;
    cout << "===========================" << endl;
}

