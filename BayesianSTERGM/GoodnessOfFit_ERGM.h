#pragma once
#include <vector>
#include <iostream>

#include <armadillo>
#include "Network.h"
#include "netMCMCSampler.h"


class GoodnessOfFit_ERGM {
    //사용시: netMCMCSampler에서 model 맞추고 쓸 것
private:
    Network obsERGM;
    Col<double> fittedParam;
    vector<Network> gofSampleVec;
    int n_Node;

    vector<vector<double>> nodeDegreeDist_eachDegreeVec;
    vector<vector<double>> edgewiseSharedPartnerDist_eachDegreeVec;
    vector<vector<double>> userSpecific_eachVec;
    vector<Col<double>> summaryQuantile_nodeDegreeDist;
    vector<Col<double>> summaryQuantile_ESPDist;
    vector<Col<double>> summaryQuantile_userSpecific;
    //나중에 최소거리dist에 대해서도 하면 좋긴할듯

    void netMCMC(int n_iter, int n_burn_in);
    void make_diagStat();
    void make_diagSummary();

public:
    GoodnessOfFit_ERGM();
    GoodnessOfFit_ERGM(Network obsERGM, Col<double> fittedParam);
    void run(int n_iter, int n_burn_in);
    void printResult();
};

