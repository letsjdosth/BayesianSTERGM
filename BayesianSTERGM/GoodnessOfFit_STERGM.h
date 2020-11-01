#pragma once
#include <vector>
#include <iostream>
#include <armadillo>

#include "Network.h"
#include "NetMCMCSampler_STERGM.h"

using namespace std;
using namespace arma;

class GoodnessOfFit_1time_STERGM {
private:
    Network obsStartNet;
    Network obsNextNet;
    Col<double> fittedParam_formation;
    Col<double> fittedParam_dissolution;
    int n_Node;

    vector<Network> gofSampleVec;
    vector<vector<double>> nodeDegreeDist_eachDegreeVec;
    vector<vector<double>> edgewiseSharedPartnerDist_eachDegreeVec;
    vector<vector<double>> userSpecific_eachVec;
    vector<Col<double>> summaryQuantile_nodeDegreeDist;
    vector<Col<double>> summaryQuantile_ESPDist;
    vector<Col<double>> summaryQuantile_userSpecific;

    void netGenerate(int num_GoF_smpl, int num_for_each_iter) {
        STERGMnet1TimeSampler_1EdgeMCMC stepGoF = STERGMnet1TimeSampler_1EdgeMCMC(fittedParam_formation, fittedParam_dissolution, obsStartNet);
        for (int i = 0; i < num_GoF_smpl; i++) {
            stepGoF.generateSample(num_for_each_iter);
            gofSampleVec.push_back(stepGoF.get_CombinedNetMCMCSample());
        }
    }

    void make_diagStat() {
        for (int i = 0; i < gofSampleVec.size(); i++) {
            Network net = gofSampleVec[i];
            //diag netstat 설정
            Col<int> netNodeDegreeDist = net.get_undirected_nodeDegreeDist(); //1차원 높게나옴(n_Node+1)
            Col<int> netESPDist = net.get_undirected_edgewiseSharedPartnerDist();
            vector<double> userSpecific = { //<-추가로 얻고싶은 netStat을 집어넣을것. 이후 생성자에서 추가netStat 개수 설정
                (double)net.get_n_Edge(),
                (double)net.get_undirected_k_starDist(2)
            };

            //diag netstat 계산
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
    GoodnessOfFit_1time_STERGM() {

    }
    GoodnessOfFit_1time_STERGM(Col<double> fittedParam_formation, Col<double> fittedParam_dissolution,
        Network obsStartNet, Network obsNextNet) {
        this->obsStartNet = obsStartNet;
        this->obsNextNet = obsNextNet;
        this->n_Node = obsStartNet.get_n_Node();
        this->fittedParam_formation = fittedParam_formation;
        this->fittedParam_dissolution = fittedParam_dissolution;
        this->nodeDegreeDist_eachDegreeVec.resize(n_Node);
        this->edgewiseSharedPartnerDist_eachDegreeVec.resize(n_Node - 1);
        this->userSpecific_eachVec.resize(2); // <- 여기에서 추가netStat 개수 설정!!
    }
    void run(int num_GoF_smpl, int num_for_each_iter) {
        netGenerate(num_GoF_smpl, num_for_each_iter);
        make_diagStat();
        make_diagSummary();
    }
    void printResult() {
        Col<int> obsNodeDegree = obsNextNet.get_undirected_nodeDegreeDist();
        Col<int> obsESP = obsNextNet.get_undirected_edgewiseSharedPartnerDist();
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

class GoodnessOfFit_STERGM {
private:
    vector<Network> obsNetSeq;
    Col<double> fittedParam_formation;
    Col<double> fittedParam_dissolution;
public:
    GoodnessOfFit_STERGM(Col<double> fittedParam_formation, Col<double> fittedParam_dissolution, vector<Network> obsNetSeq) {
        this->obsNetSeq = obsNetSeq;
        this->fittedParam_formation = fittedParam_formation;
        this->fittedParam_dissolution = fittedParam_dissolution;
    }
    void run(int startTime, int num_GoF_smpl, int num_for_each_iter) {
        GoodnessOfFit_1time_STERGM gofRunner = GoodnessOfFit_1time_STERGM(fittedParam_formation, fittedParam_dissolution,
            obsNetSeq[startTime], obsNetSeq[startTime + 1]);
        gofRunner.run(num_GoF_smpl, num_for_each_iter);
        gofRunner.printResult();
    }
};