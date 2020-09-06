#pragma once
#include <vector>
#include <armadillo>
#include "Network.h"
#include "netMCMCSampler.h"

using namespace std;
using namespace arma;



class ERGM_MCML {
private:
    vector<Col<double>> ParamSequence;
    Network observedNet;
    int n_Node;

    void updateNetworkInfo();
    vector<Network> genSampleByMCMC(int m_Smpl, int m_burnIn);
    Col<double> genWeight(Col<double> lastParam, vector<Col<double>> MCMCSample_ModelVal);
    Mat<double> invInfoCal(Col<double> lastParam, vector<Col<double>> MCMCSample_ModelVal, Col<double> weight, Col<double> wZsum);
    Col<double> NRupdate1Step();


public:
    ERGM_MCML(Col<double> initialParam, Network observed);
    void RunOptimize();
    Col<double> getMCMLE();
    void testOut();
};
