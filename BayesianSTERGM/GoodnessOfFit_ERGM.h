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

    void netMCMC(int n_iter, int n_burn_in) {
        Mat<int> zeroMat(n_Node, n_Node, fill::zeros); // initial. zeromat 싫으면 STERGMnetMCSampler::genSymmetricMat을 가져올 것
        Network zeroNet(zeroMat, 0);
        netMCMCSampler gofERGMSampler(fittedParam, zeroNet);
        gofERGMSampler.generateSample(n_iter);
        gofERGMSampler.cutBurnIn(n_burn_in);
        gofSampleVec = gofERGMSampler.getMCMCSampleVec();
    }
    void make_diagStat() {
        for (int i = 0; i < gofSampleVec.size(); i++) {
            Network net = gofSampleVec[i];
            Col<int> netNodeDegreeDist = net.get_nodeDegreeDist(); //1차원 높게나옴(n_Node+1)
            Col<int> netESPDist = net.get_edgewiseSharedPartnerDist();
            vector<double> userSpecific = { //<-추가로 얻고싶은 netStat을 집어넣을것. 이후 생성자에서 추가netStat 개수 설정
                (double)net.get_n_Edge(),
                net.get_geoWeightedNodeDegree(0.3),
                net.get_geoWeightedESP(0.3)
            };

            for (int degree = 0; degree < n_Node; degree++) {
                nodeDegreeDist_eachDegreeVec[degree].push_back((double)netNodeDegreeDist(degree));
            }
            for (int degree = 0; degree < n_Node - 1; degree++) {
                edgewiseSharedPartnerDist_eachDegreeVec[degree].push_back((double)netESPDist(degree));
            }
            for (int i = 0; i < userSpecific.size(); i++) {
                userSpecific_eachVec[i].push_back(userSpecific[i]);
            }
        }
    }
    void make_diagSummary() {
        Col<double> quantilePts = { 0.1, 0.25, 0.5, 0.75, 0.9 };

        for (int deg = 0; deg < nodeDegreeDist_eachDegreeVec.size(); deg++) {
            Col<double> eachNodeDegree = Col<double>(nodeDegreeDist_eachDegreeVec[deg]);
            summaryQuantile_nodeDegreeDist.push_back(quantile(eachNodeDegree, quantilePts));
        }
        for (int deg = 0; deg < edgewiseSharedPartnerDist_eachDegreeVec.size(); deg++) {
            Col<double> eachESP = Col<double>(edgewiseSharedPartnerDist_eachDegreeVec[deg]);
            summaryQuantile_ESPDist.push_back(quantile(eachESP, quantilePts));
        }
        for (int i = 0; i < userSpecific_eachVec.size(); i++) {
            Col<double> each = Col<double>(userSpecific_eachVec[i]);
            summaryQuantile_userSpecific.push_back(quantile(each, quantilePts));
        }
    }


public:
    GoodnessOfFit_ERGM() {
        //빈 생성자
    }
    GoodnessOfFit_ERGM(Network obsERGM, Col<double> fittedParam) {
        this->obsERGM = obsERGM;
        this->fittedParam = fittedParam;
        this->n_Node = obsERGM.get_n_Node();
        this->nodeDegreeDist_eachDegreeVec.resize(n_Node);
        this->edgewiseSharedPartnerDist_eachDegreeVec.resize(n_Node - 1);
        this->userSpecific_eachVec.resize(3); // <- 여기에서 추가netStat 개수 설정!!
    }
    void run(int n_iter, int n_burn_in) {
        netMCMC(n_iter, n_burn_in);
        make_diagStat();
        make_diagSummary();
    }
    void printResult() {
        Col<int> obsNodeDegree = obsERGM.get_nodeDegreeDist();
        Col<int> obsESP = obsERGM.get_edgewiseSharedPartnerDist();
        for (int i = 0; i < summaryQuantile_nodeDegreeDist.size(); i++) {
            cout << "#node degree " << i << " :\n";
            cout << "data: " << obsNodeDegree(i) << "\n";
            cout << "at the param: " << summaryQuantile_nodeDegreeDist[i].t() << endl;
        }
        for (int i = 0; i < summaryQuantile_ESPDist.size(); i++) {
            cout << "#Edgewise Shared Partner  degree " << i << " :\n";
            cout << "data:" << obsESP(i) << "\n";
            cout << "at the param: " << summaryQuantile_ESPDist[i].t() << endl;
        }
        for (int i = 0; i < summaryQuantile_userSpecific.size(); i++) {
            cout << "#userSpecific index " << i << " :\n";
            cout << "data:" << "see yourself" << "\n";
            cout << "at the param: " << summaryQuantile_userSpecific[i].t() << endl;
        }
    }
};

