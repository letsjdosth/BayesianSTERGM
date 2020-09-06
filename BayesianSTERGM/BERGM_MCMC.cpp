#pragma once
#include <vector>
#include <iostream>
#include <armadillo>

#include "Network.h"
#include "netMCMCSampler.h"
#include "BERGM_MCMC.h"

using namespace std;
using namespace arma;


//private 
void BERGM_MCMC::updateNetworkInfo() {
    n_Node = observedNet.get_n_Node();
    n_paramDim = ParamPosteriorSmpl[0].size();
}

Col<double> BERGM_MCMC::proposeParam(Col<double> mean, double covRate) {
    //from multivariate normal
    Mat<double> covariance(n_paramDim, n_paramDim, fill::eye);
    Col<double> proposedParam = mvnrnd(mean, covariance * covRate);
    return proposedParam;
}

Network BERGM_MCMC::genNetworkSampleByMCMC(Col<double> parameter, int m_MCMCiter) {
    //undirected graph
    Mat<int> initialNetStructure; // isolated graph로 시작 (random으로 뿌리면 더 좋을듯)
    initialNetStructure.zeros(n_Node, n_Node);
    Network initialNet(initialNetStructure, 0);

    netMCMCSampler MCMCsampler(parameter, initialNet);
    MCMCsampler.generateSample(m_MCMCiter);
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
    Col<double> model_delta = { (double)exchangeNet.get_n_Edge() - observedNet.get_n_Edge() }; // <-model specify. s(y')-s(y)
    Col<double> log_r_col = (lastParam - proposedParam).t() * model_delta;
    double res = log_r_col(0) + log_paramPriorPDF(proposedParam) - log_paramPriorPDF(lastParam);
    return res;
}

void BERGM_MCMC::sampler() {
    Col<double> lastParam = ParamPosteriorSmpl.back();
    Col<double> proposedParam = proposeParam(lastParam, 0.0025);
    Network exchangeNet = genNetworkSampleByMCMC(proposedParam, 10000);

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


void BERGM_MCMC::generateSample(int num_iter) {
    for (int i = 0; i < num_iter; i++) {
        if (i % 10 == 0) {
            cout << "MCMC : " << n_iterated << "/" << num_iter << endl;
        }
        sampler();
    }
    cout << "MCMC done: " << n_iterated << " posterior samples are generated." << endl;
}

void BERGM_MCMC::cutBurnIn(int n_burn_in) {
    ParamPosteriorSmpl.erase(ParamPosteriorSmpl.begin(), ParamPosteriorSmpl.begin() + n_burn_in + 1);
}

void BERGM_MCMC::testOut() {
    int i = 0;
    while (i < ParamPosteriorSmpl.size()) {
        cout << "#" << i << " : " << ParamPosteriorSmpl[i].t() << endl;
        i++;
    }
}