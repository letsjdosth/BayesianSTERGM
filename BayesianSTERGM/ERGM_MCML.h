#pragma once
#include <vector>
#include <armadillo>
#include "Network.h"
#include "NetMCMCSampler_ERGM.h"

using namespace std;
using namespace arma;



class ERGM_MCML {
private:
    vector<Col<double>> ParamSequence;
    Network observedNet;
    int n_Node;

    //for diagnostic of MCMC
    NetMCMCSampler_ERGM latestStep_netMCSampler;


    Col<double> netOne_modelVal(Network net);
    vector<Col<double>> netVec_modelVal(vector<Network> netVec);

    void updateNetworkInfo();
    vector<Network> genSampleByMCMC(int m_Smpl, int m_burnIn);
    Col<double> genWeight(Col<double> lastParam, vector<Col<double>> MCMCSample_ModelVal);
    Mat<double> invInfoCal(Col<double> lastParam, vector<Col<double>> MCMCSample_ModelVal, Col<double> weight, Col<double> wZsum);
    Col<double> NRupdate1Step();


    //for diagnostic of MCMC
    double diag_logLiklihoodRatio(Col<double> upperParam, Col<double> lowerParam);
    double diag_estimatedLikelihood();
    vector<double> autoCorr(Col<double> sampleSequence, int maxLag);
    double diag_varianceLastMC(int estLag_K);


public:
    ERGM_MCML(Col<double> initialParam, Network observed);
    void RunOptimize();
    Col<double> getMCMLE();
    NetMCMCSampler_ERGM getLatestStep_netMCSampler();
    void print_checkConvergence();
    void testOut();
};
