#pragma once
#include <vector>
#include <iostream>
#include <armadillo>

#include "Network.h"
#include "netMCMCSampler.h"
#include "ERGM_MCML.h"

using namespace std;
using namespace arma;

//private

void ERGM_MCML::updateNetworkInfo() {
    n_Node = observedNet.get_n_Node();
}

vector<Network> ERGM_MCML::genSampleByMCMC(int m_Smpl, int m_burnIn) {
    //undirected graph
    Col<double> lastParam = ParamSequence.back();
    Mat<int> initialNetStructure; // isolated graph로 시작 (random으로 뿌리면 더 좋을듯)
    initialNetStructure.zeros(n_Node, n_Node);
    Network initialNet(initialNetStructure, 0);

    netMCMCSampler MCMCsampler(lastParam, initialNet);
    MCMCsampler.generateSample(m_Smpl);
    MCMCsampler.cutBurnIn(m_burnIn);
    vector<Network> MCMCSampleVec = MCMCsampler.getMCMCSampleVec();
    return MCMCSampleVec;
}


Col<double> ERGM_MCML::genWeight(Col<double> lastParam, vector<Col<double>> MCMCSample_ModelVal) {
    Col<double> initParam = ParamSequence.front();
    Col<double> weight = zeros(MCMCSample_ModelVal.size());
    for (int i = 0; i < MCMCSample_ModelVal.size(); i++) {
        Col<double> beforeExp = (lastParam - initParam).t() * MCMCSample_ModelVal[i];
        weight[i] = exp(beforeExp(0));
    }
    double weightTotal = accu(weight);
    weight = weight / weightTotal;
    return weight;
}


Mat<double> ERGM_MCML::invInfoCal(Col<double> lastParam, vector<Col<double>> MCMCSample_ModelVal, Col<double> weight, Col<double> wZsum) {
    Mat<double> FisherInfo = zeros(lastParam.size(), lastParam.size());
    Mat<double> wZZsum = zeros(lastParam.size(), lastParam.size());

    for (int i = 0; i < MCMCSample_ModelVal.size(); i++) {
        wZZsum += MCMCSample_ModelVal[i] * (MCMCSample_ModelVal[i].t()) * weight[i];
    }
    FisherInfo = wZZsum - wZsum * (wZsum.t());
    Mat<double> invFisherInfo = inv(FisherInfo);
    return invFisherInfo;
}

Col<double> ERGM_MCML::NRupdate1Step() {
    Col<double> lastParam = ParamSequence.back();

    //MCMC
    int m_MCSample = 1000;
    vector<Network> MCMCSampleVec = genSampleByMCMC(10000, 9000);

    //make MCMCSample_ModelVal vector for each MCMC sample (in paper, Z_i)
    //NOW: model : n_Edge
    vector<Col<double>> MCMCSample_ModelVal; //Z_i vectors
    for (int i = 0; i < MCMCSampleVec.size(); i++) {
        Col<double> val = { (double)MCMCSampleVec[i].get_n_Edge() }; // <- model specify
        MCMCSample_ModelVal.push_back(val);
    }

    //make weight vector (in paper, w_i)
    Col<double> weight = genWeight(lastParam, MCMCSample_ModelVal);

    //Newton-Raphson
    Col<double> Observed_ModelVal = { (double)observedNet.get_n_Edge() };

    Col<double> wZsum = zeros(lastParam.size());
    for (int i = 0; i < MCMCSample_ModelVal.size(); i++) {
        wZsum += MCMCSample_ModelVal[i] * weight[i];
    }
    Mat<double> invFisherInfo = invInfoCal(lastParam, MCMCSample_ModelVal, weight, wZsum);


    Col<double> newParam = lastParam + invFisherInfo * (Observed_ModelVal - wZsum);
    return newParam;
}

//public

ERGM_MCML::ERGM_MCML(Col<double> initialParam, Network observed) {
    ParamSequence.push_back(initialParam);
    observedNet = observed;
    updateNetworkInfo();
}

void ERGM_MCML::RunOptimize() {
    bool eq = false;
    int runNWnum = 0;
    double epsilon_thres = 0.002;
    while (!eq) {
        cout << "N-R iter" << runNWnum << endl;
        Col<double> lastParam = ParamSequence.back();
        Col<double> newParam = NRupdate1Step();
        ParamSequence.push_back(newParam);
        eq = approx_equal(lastParam, newParam, "absdiff", epsilon_thres);
        cout << "proposed: " << newParam << endl;
        runNWnum++;
    }
    cout << "optimized! iter:" << runNWnum << endl;
}

Col<double> ERGM_MCML::getMCMLE() {
    return ParamSequence.back();
}

void ERGM_MCML::testOut() {
    vector<Network> MCMCSampleVec = genSampleByMCMC(10, 8);
    int i = 0;
    while (i < MCMCSampleVec.size()) {
        Network printedNet = MCMCSampleVec[i];
        cout << "#" << i << endl;
        printedNet.printSummary();
        i++;
    }

}
