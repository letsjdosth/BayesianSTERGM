#pragma once
#include <armadillo>

using namespace std;
using namespace arma;

class Network {
private:
    Mat<int> netStructure;
    bool isDirected;
    int n_Node;
    int n_Edge;
    Col<int> dist_nodeDegree;

    void updateNetworkInfo();
    int edgeNum();
    Col<int> nodeDegree();
    Col<int> nodeDegreeDist();
    //Col<int> sharedPartnerDist();
    int fact(int n);
    int nCr(int n, int r);
    Col<int> k_starDist();
    int n_triangle();
public:
    Network(Mat<int> inputNet, bool isDirectedInput);
    Network();
    int get_n_Node();
    int get_n_Edge();
    Mat<int> get_netStructure();
    void printSummary();
};