#pragma once
#include <vector>
#include <iostream>
#include <armadillo>

#include "Network.h"
#include "NetMCMCSampler_ERGM.h"
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
    Mat<int> initialNetStructure; // isolated graph로 시작 (random으로 뿌리면 더 좋을듯) << 이거 고칠것!
    initialNetStructure.zeros(n_Node, n_Node);
    Network initialNet(initialNetStructure, 0);

    NetMCMCSampler_ERGM MCMCsampler(lastParam, initialNet);
    MCMCsampler.generateSample(m_Smpl);
    MCMCsampler.cutBurnIn(m_burnIn);
    latestStep_netMCSampler = MCMCsampler;
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


Col<double> ERGM_MCML::netOne_modelVal(Network net) {
    // make Z(y)
    //NOW: model : n_Edge
    Col<double> val = { (double)net.get_n_Edge(), (double)net.get_undirected_k_starDist(2) }; // <- model specify
    return val;
}


vector<Col<double>> ERGM_MCML::netVec_modelVal(vector<Network> netVec) {
    // make Z(y_i) vec
    vector<Col<double>> modelValVec;
    for (int i = 0; i < netVec.size(); i++) {
        Col<double> val = netOne_modelVal(netVec[i]);
        modelValVec.push_back(val);
    }
    return modelValVec;
}

Col<double> ERGM_MCML::NRupdate1Step() {
    Col<double> lastParam = ParamSequence.back();

    //MCMC
    int m_MCSample = 1000;
    vector<Network> MCMCSampleVec = genSampleByMCMC(10000, 9000);

    //make MCMCSample_ModelVal vector for each MCMC sample (in paper, Z_i)
    //NOW: model : n_Edge
    vector<Col<double>> MCMCSample_ModelVal = netVec_modelVal(MCMCSampleVec);

    //make weight vector (in paper, w_i)
    Col<double> weight = genWeight(lastParam, MCMCSample_ModelVal);

    //Newton-Raphson
    Col<double> Observed_ModelVal = netOne_modelVal(observedNet);

    Col<double> wZsum = zeros(lastParam.size());
    for (int i = 0; i < MCMCSample_ModelVal.size(); i++) {
        wZsum += MCMCSample_ModelVal[i] * weight[i];
    }
    Mat<double> invFisherInfo = invInfoCal(lastParam, MCMCSample_ModelVal, weight, wZsum);


    Col<double> newParam = lastParam + invFisherInfo * (Observed_ModelVal - wZsum);
    return newParam;
}

//diagnostics


double ERGM_MCML::diag_logLiklihoodRatio(Col<double> upperParam, Col<double> lowerParam) {
    //smallgamma_m(eta, eta0)
    Col<double> paramDiff = upperParam - lowerParam;

    vector<Network> lastMCNetVec = latestStep_netMCSampler.getMCMCSampleVec();
    vector<Col<double>> ModelValVec_MC = netVec_modelVal(lastMCNetVec);
    Col<double> ModelVal_obs = netOne_modelVal(observedNet);

    Col<double> ratio_obsTerm = paramDiff.t() * ModelVal_obs;
    double ratio = ratio_obsTerm(0);

    Col<double> ratio_MCTerm(ModelValVec_MC.size());
    for (int i = 0; i < ModelValVec_MC.size(); i++) {
        Col<double> prod_col = paramDiff.t() * ModelValVec_MC[i];
        ratio_MCTerm(i) = exp(prod_col(0));
    }
    ratio -= log(mean(ratio_MCTerm));

    return ratio;
}
double ERGM_MCML::diag_estimatedLikelihood() {
    //undirected
    Col<double> atTheParam = ParamSequence.back();
    Col<double> initParam = ParamSequence[0];

    double num_allCase = ((double)n_Node * ((double)n_Node - 1)) / 2;
    Col<double> zeroVec(atTheParam.size(), fill::zeros);
    double logLikelihood = diag_logLiklihoodRatio(atTheParam, initParam) - diag_logLiklihoodRatio(zeroVec, initParam);
    logLikelihood -= log(num_allCase);
    return logLikelihood;
}


vector<double> ERGM_MCML::autoCorr(Col<double> sampleSequence, int maxLag) {
    vector<double> autoCorrVec(maxLag + 1);
    autoCorrVec[0] = 1.0;
    Col<double> diffMean = sampleSequence - mean(sampleSequence);
    for (int i = 1; i < maxLag + 1; i++) {
        int num_pair = diffMean.size() - i;
        double cov_term = 0;
        for (int j = 0; j < num_pair; j++) {
            cov_term += (diffMean[j] * diffMean[j + i]);
        }
        cov_term /= num_pair;
        autoCorrVec[i] = (cov_term / var(sampleSequence));
    }
    return autoCorrVec;
}

double ERGM_MCML::diag_varianceLastMC(int estLag_K) {
    //choose K s.t. K<<m_num, so that autocorr_lagK is approximately zero for k>K

    vector<Network> lastMCNetVec = latestStep_netMCSampler.getMCMCSampleVec();
    int m_num = lastMCNetVec.size();
    vector<Col<double>> MCMCSample_ModelVal = netVec_modelVal(lastMCNetVec); //Zi

    Col<double> initParam = ParamSequence[0];
    Col<double> MCMLParam = ParamSequence.back();
    Col<double> Ui(m_num);
    for (int i = 0; i < m_num; i++) {
        Col<double> val = (MCMLParam - initParam).t() * MCMCSample_ModelVal[i];
        Ui(i) = exp(val(0));
    }
    double U_bar = mean(Ui);
    vector<double> U_autoCorr = autoCorr(Ui, estLag_K);
    double VAR_lastMC = 0;
    for (int k = 1; k < estLag_K; k++) {
        VAR_lastMC += (((double)m_num - k) * U_autoCorr[k]);
    }
    VAR_lastMC *= 2;
    VAR_lastMC += m_num; //k=0;
    VAR_lastMC /= (U_bar * U_bar * m_num * m_num);
    return VAR_lastMC;
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
    double epsilon_thres = 0.05;
    while (!eq) {
        cout << "N-R iter" << runNWnum << endl;
        Col<double> lastParam = ParamSequence.back();
        Col<double> newParam = NRupdate1Step();
        ParamSequence.push_back(newParam);
        eq = approx_equal(lastParam, newParam, "absdiff", epsilon_thres);
        cout << "proposed: " << newParam.t() << endl;
        runNWnum++;
    }
    cout << "optimized! iter:" << runNWnum << endl;
}

Col<double> ERGM_MCML::getMCMLE() {
    return ParamSequence.back();
}

NetMCMCSampler_ERGM ERGM_MCML::getLatestStep_netMCSampler() {
    return latestStep_netMCSampler;
}

void ERGM_MCML::print_checkConvergence() {
    cout << "if var_MC >> logLiklihood, then re-optimize with new initial condition." << endl;
    cout << "VAR_MC: " << diag_varianceLastMC(10) << " , ";
    cout << "logLiklihood(est): " << diag_estimatedLikelihood() << endl;
}

void ERGM_MCML::testOut() {
    vector<Network> MCMCSampleVec = genSampleByMCMC(10, 8);
    int i = 0;
    while (i < MCMCSampleVec.size()) {
        Network printedNet = MCMCSampleVec[i];
        cout << "#" << i << endl;
        printedNet.undirected_printSummary();
        i++;
    }

}
