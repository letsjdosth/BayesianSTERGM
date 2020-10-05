#pragma once
#include <vector>
#include <armadillo>
#include "Network.h"
#include "NetMCMCSampler_ERGM.h"

using namespace std;
using namespace arma;



class BERGM_MCMC {
private:
    vector<Col<double>> ParamPosteriorSmpl;
    Network observedNet;
    int n_Node;
    int n_paramDim;
    int n_accepted;
    int n_iterated;
    NetMCMCSampler_ERGM lastExchangeNetworkSampler;

    void updateNetworkInfo();
    Col<double> proposeParam(Col<double> mean, double covRate);
    Network genNetworkSampleByMCMC(Col<double> parameter, Network initialNet, int m_MCMCiter);
    double log_paramPriorPDF(Col<double> param);
    double log_r(Col<double> lastParam, Col<double> proposedParam, Network exchangeNet);
    void sampler(int num_exchangeMCiter);
public:
    BERGM_MCMC(Col<double> initialParam, Network observed);
    void generateSample(int num_mainMCiter, int num_exchangeMCiter);
    void cutBurnIn(int n_burn_in);
    void thinning(int n_lag);
    vector<Col<double>> getPosteriorSample();
    NetMCMCSampler_ERGM get_lastExchangeNetworkSampler();
    void testOut();
};