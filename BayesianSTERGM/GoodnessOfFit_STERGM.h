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
    bool isDirected;

    Network gofSample;
    Col<int> netNodeDegreeDist;
    Col<int> netNodeInDegreeDist;
    Col<int> netNodeOutDegreeDist;
    Col<int> netESPDist;
    vector<double> userSpecific;

    void netGenerate(int num_for_each_iter) {
        STERGMnet1TimeSampler_1EdgeMCMC stepGoF = STERGMnet1TimeSampler_1EdgeMCMC(fittedParam_formation, fittedParam_dissolution, obsStartNet);
        stepGoF.generateSample(num_for_each_iter);
        gofSample = stepGoF.get_CombinedNetMCMCSample();
    }

    void undirected_make_diagStat() {
        //diag netstat 설정
        netNodeDegreeDist = gofSample.get_undirected_nodeDegreeDist(); //1차원 높게나옴(n_Node+1)
        netESPDist = gofSample.get_undirected_edgewiseSharedPartnerDist();
        userSpecific = { //<-추가로 얻고싶은 netStat을 집어넣을것. 이후 생성자에서 추가netStat 개수 설정
            (double)gofSample.get_n_Edge(),
            (double)gofSample.get_undirected_k_starDist(2)
        };
    }

    void directed_make_diagStat() {
        //diag netstat 설정
        netNodeInDegreeDist = gofSample.get_directed_nodeInDegreeDist(); //1차원 높게나옴(n_Node+1)
        netNodeOutDegreeDist = gofSample.get_directed_nodeOutDegreeDist(); //1차원 높게나옴(n_Node+1)
        netESPDist = gofSample.get_directed_edgewiseSharedPartnerDist();
        userSpecific = { //<-추가로 얻고싶은 netStat을 집어넣을것. 이후 생성자에서 추가netStat 개수 설정
            (double)gofSample.get_n_Edge(),
            (double)gofSample.get_directed_geoWeightedESP(0.5)
        };
    }

public:
    GoodnessOfFit_1time_STERGM() {

    }
    GoodnessOfFit_1time_STERGM(Col<double> fittedParam_formation, Col<double> fittedParam_dissolution,
        Network obsStartNet, Network obsNextNet) {
        this->obsStartNet = obsStartNet;
        this->obsNextNet = obsNextNet;
        this->fittedParam_formation = fittedParam_formation;
        this->fittedParam_dissolution = fittedParam_dissolution;
        this->isDirected = obsStartNet.is_directed_graph();
        this->n_Node = obsStartNet.get_n_Node();
        
    }
    void undirected_run(int num_for_each_iter) {
        netGenerate(num_for_each_iter);
        undirected_make_diagStat();
    }
    void directed_run(int num_for_each_iter) {
        netGenerate(num_for_each_iter);
        directed_make_diagStat();
    }
    
    Col<int> get_netNodeDegreeDist() {
        return netNodeDegreeDist;
    }
    Col<int> get_netNodeInDegreeDist() {
        return netNodeInDegreeDist;
    }
    Col<int> get_netNodeOutDegreeDist() {
        return netNodeOutDegreeDist;
    }
    Col<int> get_netESPDist() {
        return netESPDist;
    }
    vector<double> get_userSpecificStat() {
        return userSpecific;
    }
};


class GoodnessOfFit_STERGM {
private:
    vector<Network> obsNetSeq;
    vector<Col<double>> posteriorParam_formation;
    vector<Col<double>> posteriorParam_dissolution;

    int n_Node;
    bool isDirected;
    
    vector<vector<double>> nodeDegreeDist_eachDegreeVec;
    vector<vector<double>> nodeInDegreeDist_eachDegreeVec;
    vector<vector<double>> nodeOutDegreeDist_eachDegreeVec;
    vector<vector<double>> edgewiseSharedPartnerDist_eachDegreeVec;
    vector<vector<double>> userSpecific_eachVec;

    vector<Col<double>> summaryQuantile_nodeDegreeDist;
    vector<Col<double>> summaryQuantile_nodeInDegreeDist;
    vector<Col<double>> summaryQuantile_nodeOutDegreeDist;
    vector<Col<double>> summaryQuantile_ESPDist;
    vector<Col<double>> summaryQuantile_userSpecific;


    void undirected_cal(int startTime, int num_usingPosteriorSample, int num_for_each_iter) {
        for (int i = num_usingPosteriorSample; i > 0; i--) {
            GoodnessOfFit_1time_STERGM gofRunner = GoodnessOfFit_1time_STERGM(posteriorParam_formation[i], posteriorParam_dissolution[i],
                obsNetSeq[startTime], obsNetSeq[startTime + 1]);
            gofRunner.undirected_run(num_for_each_iter);

            //save
            Col<int> net_nodeDegreeDist = gofRunner.get_netNodeDegreeDist();
            Col<int> net_ESPdist = gofRunner.get_netESPDist();
            vector<double> net_userSpecific = gofRunner.get_userSpecificStat();
            for (int degree = 0; degree < n_Node; degree++) {
                nodeDegreeDist_eachDegreeVec[degree].push_back((double)net_nodeDegreeDist(degree));
            }
            for (int degree = 0; degree < n_Node - 1; degree++) {
                edgewiseSharedPartnerDist_eachDegreeVec[degree].push_back((double)net_ESPdist(degree));
            }
            for (int i = 0; i < net_userSpecific.size(); i++) {
                userSpecific_eachVec[i].push_back(net_userSpecific[i]);
            }
        }
    }

    void undirected_make_diagSummary() {
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

    void directed_cal(int startTime, int num_usingPosteriorSample, int num_for_each_iter) {
        for (int i = num_usingPosteriorSample; i > 0; i--) {
            GoodnessOfFit_1time_STERGM gofRunner = GoodnessOfFit_1time_STERGM(posteriorParam_formation[i], posteriorParam_dissolution[i],
                obsNetSeq[startTime], obsNetSeq[startTime + 1]);
            gofRunner.directed_run(num_for_each_iter);

            //save
            Col<int> net_nodeInDegreeDist = gofRunner.get_netNodeInDegreeDist();
            Col<int> net_nodeOutDegreeDist = gofRunner.get_netNodeOutDegreeDist();
            Col<int> net_ESPdist = gofRunner.get_netESPDist();
            vector<double> net_userSpecific = gofRunner.get_userSpecificStat();
            for (int degree = 0; degree < n_Node; degree++) {
                nodeInDegreeDist_eachDegreeVec[degree].push_back((double)net_nodeInDegreeDist(degree));
                nodeOutDegreeDist_eachDegreeVec[degree].push_back((double)net_nodeOutDegreeDist(degree));
            }
            for (int degree = 0; degree < n_Node - 1; degree++) {
                edgewiseSharedPartnerDist_eachDegreeVec[degree].push_back((double)net_ESPdist(degree));
            }
            for (int i = 0; i < net_userSpecific.size(); i++) {
                userSpecific_eachVec[i].push_back(net_userSpecific[i]);
            }
        }
    }

    void directed_make_diagSummary() {
        Col<double> quantilePts = { 0.1, 0.25, 0.5, 0.75, 0.9 };
        for (int deg = 0; deg < nodeInDegreeDist_eachDegreeVec.size(); deg++) {
            Col<double> eachNodeInDegree = Col<double>(nodeInDegreeDist_eachDegreeVec[deg]);
            Col<double> eachNodeOutDegree = Col<double>(nodeOutDegreeDist_eachDegreeVec[deg]);
            summaryQuantile_nodeInDegreeDist.push_back(quantile(eachNodeInDegree, quantilePts));
            summaryQuantile_nodeOutDegreeDist.push_back(quantile(eachNodeOutDegree, quantilePts));
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
    GoodnessOfFit_STERGM(vector<Col<double>> posteriorParam_formation, vector<Col<double>> posteriorParam_dissolution, vector<Network> obsNetSeq) {
        this->obsNetSeq = obsNetSeq;
        this->posteriorParam_formation = posteriorParam_formation;
        this->posteriorParam_dissolution = posteriorParam_dissolution;
        
        this->isDirected = obsNetSeq[0].is_directed_graph();
        this->n_Node = obsNetSeq[0].get_n_Node();
        if (isDirected) {
            this->nodeInDegreeDist_eachDegreeVec.resize(n_Node);
            this->nodeOutDegreeDist_eachDegreeVec.resize(n_Node);
        }
        else {
            this->nodeDegreeDist_eachDegreeVec.resize(n_Node);
        }
        this->edgewiseSharedPartnerDist_eachDegreeVec.resize(n_Node - 1);
        this->userSpecific_eachVec.resize(2); // <- 여기에서 추가netStat 개수 설정!!
    }

    void undirected_run(int startTime, int num_usingPosteriorSample, int num_for_each_iter) {
        undirected_cal(startTime, num_usingPosteriorSample, num_for_each_iter);
        undirected_make_diagSummary();
    }
    void directed_run(int startTime, int num_usingPosteriorSample, int num_for_each_iter) {
        directed_cal(startTime, num_usingPosteriorSample, num_for_each_iter);
        directed_make_diagSummary();
    }

    void undirected_printResult(int startTime) {
        Col<int> obsNodeDegree = obsNetSeq[startTime + 1].get_undirected_nodeDegreeDist();
        Col<int> obsESP = obsNetSeq[startTime + 1].get_undirected_edgewiseSharedPartnerDist();
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

    void directed_printResult(int startTime) {
        Col<int> obsNodeInDegree = obsNetSeq[startTime + 1].get_directed_nodeInDegreeDist();
        Col<int> obsNodeOutDegree = obsNetSeq[startTime + 1].get_directed_nodeOutDegreeDist();
        Col<int> obsESP = obsNetSeq[startTime + 1].get_directed_edgewiseSharedPartnerDist();
        for (int i = 0; i < summaryQuantile_nodeInDegreeDist.size(); i++) {
            cout << "#node in-degree " << i << " :\n";
            cout << "data: " << obsNodeInDegree(i) << "\n";
            cout << "at the param: " << summaryQuantile_nodeInDegreeDist[i].t() << endl;
        }
        for (int i = 0; i < summaryQuantile_nodeOutDegreeDist.size(); i++) {
            cout << "#node out-degree " << i << " :\n";
            cout << "data: " << obsNodeOutDegree(i) << "\n";
            cout << "at the param: " << summaryQuantile_nodeOutDegreeDist[i].t() << endl;
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