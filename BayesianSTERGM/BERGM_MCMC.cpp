#pragma once
#include <vector>
#include <iostream>
#include <armadillo>

#include "Network.h"
#include "NetMCMCSampler_ERGM.h"
#include "BERGM_MCMC.h"

using namespace std;
using namespace arma;


//private 
void BERGM_MCMC::updateNetworkInfo() {
    n_Node = observedNet.get_n_Node();
    n_paramDim = ParamPosteriorSmpl[0].size();
}

Col<double> BERGM_MCMC::proposeParam(Col<double> mean, double varRate) {
    //from multivariate normal
    Mat<double> covariance(n_paramDim, n_paramDim, fill::eye);
    Col<double> proposedParam = mvnrnd(mean, covariance * varRate);
    return proposedParam;
}

Network BERGM_MCMC::genNetworkSampleByMCMC(Col<double> parameter, Network initialNet, int m_MCMCiter) {
    //undirected graph
    NetMCMCSampler_ERGM MCMCsampler(parameter, initialNet);
    MCMCsampler.generateSample(m_MCMCiter);
    lastExchangeNetworkSampler = MCMCsampler;
    Network sampleNet = MCMCsampler.getMCMCSampleVec().back(); //last one
    return sampleNet;
}

double BERGM_MCMC::log_paramPriorPDF(Col<double> param) {
    //NOW: model parameter prior : 1

    return 0;

}

double BERGM_MCMC::log_r(Col<double> lastParam, Col<double> proposedParam, Network exchangeNet) {
    //model specify 분리할 수 있으면 좋긴할듯 (어떻게?)
    //NOW: model : n_Edge
    Col<double> model_delta = { (double)exchangeNet.get_n_Edge() - observedNet.get_n_Edge() ,
                                (double)exchangeNet.get_k_starDist(2) - observedNet.get_k_starDist(2) }; // <-model specify. s(y')-s(y)
    Col<double> log_r_col = (lastParam - proposedParam).t() * model_delta;
    double res = log_r_col(0) + log_paramPriorPDF(proposedParam) - log_paramPriorPDF(lastParam);
    return res;
}

void BERGM_MCMC::sampler(int num_exchangeMCiter) {
    Col<double> lastParam = ParamPosteriorSmpl.back();
    Col<double> proposedParam = proposeParam(lastParam, 0.05);
    Network exchangeInitNet = observedNet;
    Network exchangeNet = genNetworkSampleByMCMC(proposedParam, exchangeInitNet, num_exchangeMCiter);

    double log_unif_sample = log(randu());
    double log_r_val = log_r(lastParam, proposedParam, exchangeNet);
    if (log_unif_sample < log_r_val) {
        //accept
        ParamPosteriorSmpl.push_back(proposedParam);
        n_accepted++;
        n_iterated++;
    }
    else {
        //reject
        ParamPosteriorSmpl.push_back(lastParam);
        n_iterated++;
    }
}

// public

BERGM_MCMC::BERGM_MCMC(Col<double> initialParam, Network observed) {
    ParamPosteriorSmpl.push_back(initialParam);
    observedNet = observed;
    updateNetworkInfo();
}


void BERGM_MCMC::generateSample(int num_mainMCiter, int num_exchangeMCiter) {
    for (int i = 0; i < num_mainMCiter; i++) {
        if (i % 100 == 0) {
            cout << "MCMC : " << n_iterated << "/" << num_mainMCiter << endl;
        }
        sampler(num_exchangeMCiter);
    }
    cout << "MCMC done: " << n_iterated << " posterior samples are generated." << endl;
}

void BERGM_MCMC::cutBurnIn(int n_burn_in) {
    ParamPosteriorSmpl.erase(ParamPosteriorSmpl.begin(), ParamPosteriorSmpl.begin() + n_burn_in + 1);
}
void BERGM_MCMC::thinning(int n_lag) {
    vector<Col<double>> ParamPosteriorSmpl_afterThinning;
    for (int i = 0; i < ParamPosteriorSmpl.size(); i += n_lag) {
        ParamPosteriorSmpl_afterThinning.push_back(ParamPosteriorSmpl[i]);
    }
    ParamPosteriorSmpl = ParamPosteriorSmpl_afterThinning;
}

vector<Col<double>> BERGM_MCMC::getPosteriorSample() {
    return ParamPosteriorSmpl;
}

NetMCMCSampler_ERGM BERGM_MCMC::get_lastExchangeNetworkSampler() {
    return lastExchangeNetworkSampler;
}

void BERGM_MCMC::testOut() {
    int i = 0;
    while (i < ParamPosteriorSmpl.size()) {
        cout << "#" << i << " : " << ParamPosteriorSmpl[i].t() << endl;
        i++;
    }
}