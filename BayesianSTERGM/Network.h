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

    int undirected_edgeNum();
    int directed_edgeNum();
    
    Col<int> undirected_nodeDegree();
    Col<int> directed_nodeInDegree();
    Col<int> directed_nodeOutDegree();

    Col<int> undirected_nodeDegreeDist();
    Col<int> directed_nodeInDegreeDist();
    Col<int> directed_nodeOutDegreeDist();

    double undirected_geoWeightedNodeDegree(double tau);

    Mat<int> undirected_edgewiseSharedPartner();
    Mat<int> directed_edgewiseSharedPartner();
    Col<int> undirected_edgewiseSharedPartnerDist();
    Col<int> directed_edgewiseSharedPartnerDist();
    double undirected_geoWeightedESP(double tau);
    double directed_geoWeightedESP(double tau);
    
    Mat<int> undirected_dyadwiseSharedPartner();
    Col<int> undirected_dyadwiseSharedPartnerDist();
    double undirected_geoWeightedDSP(double tau);

    int fact(int n);
    int nCr(int n, int r);
    Col<int> undirected_k_starDist();
    Col<int> undirected_n_triangleDist();


public:
    Network(Mat<int> inputNet, bool isDirectedInput);
    Network();
    Mat<int> get_netStructure();
    
    bool is_directed_graph();
    int get_n_Node();
    int get_n_Edge();
    
    int get_undirected_nodeDegreeDist(int degree);
    Col<int> get_undirected_nodeDegreeDist();
    double get_undirected_geoWeightedNodeDegree(double tau);
    
    int get_directed_nodeInDegreeDist(int degree);
    Col<int> get_directed_nodeInDegreeDist();
    // double get_directed_geoWeightedNodeInDegree(double tau);

    int get_directed_nodeOutDegreeDist(int degree);
    Col<int> get_directed_nodeOutDegreeDist();
    // double get_directed_geoWeightedNodeOutDegree(double tau);

    int get_undirected_edgewiseSharedPartnerDist(int degree);
    Col<int> get_undirected_edgewiseSharedPartnerDist();
    double get_undirected_geoWeightedESP(double tau);

    int get_directed_edgewiseSharedPartnerDist(int degree);
    Col<int> get_directed_edgewiseSharedPartnerDist();
    double get_directed_geoWeightedESP(double tau);

    int get_undirected_dyadwiseSharedPartnerDist(int degree);
    Col<int> get_undirected_dyadwiseSharedPartnerDist();
    double get_undirected_geoWeightedDSP(double tau);

    int get_undirected_k_starDist(int degree_k);
    Col<int> get_undirected_k_starDist();
    int get_undirected_triangleDist(int degree);
    Col<int> get_undirected_triangleDist();

    void undirected_printSummary();
    void directed_printSummary();
};