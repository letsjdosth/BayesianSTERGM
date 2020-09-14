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
    double geoWeightedNodeDegree(double tau);

    Mat<int> edgewiseSharedPartner();
    Col<int> edgewiseSharedPartnerDist();
    double geoWeightedESP(double tau);
    
    Mat<int> dyadwiseSharedPartner();
    Col<int> dyadwiseSharedPartnerDist();
    double geoWeightedDSP(double tau);

    int fact(int n);
    int nCr(int n, int r);
    Col<int> k_starDist();
    Col<int> n_triangleDist();


public:
    Network(Mat<int> inputNet, bool isDirectedInput);
    Network();
    Mat<int> get_netStructure();
    
    int get_n_Node();
    int get_n_Edge();
    
    int get_nodeDegreeDist(int degree);
    Col<int> get_nodeDegreeDist();
    double get_geoWeightedNodeDegree(double tau);
    
    int get_edgewiseSharedPartnerDist(int degree);
    Col<int> get_edgewiseSharedPartnerDist();
    double get_geoWeightedESP(double tau);

    int get_dyadwiseSharedPartnerDist(int degree);
    Col<int> get_dyadwiseSharedPartnerDist();
    double get_geoWeightedDSP(double tau);

    int get_k_starDist(int degree_k);
    Col<int> get_k_starDist();
    int get_triangleDist(int degree);
    Col<int> get_triangleDist();

    void printSummary();
};