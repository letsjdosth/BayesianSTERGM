#pragma once
#include <fstream>
#include <vector>
#include <iostream>
#include <armadillo>

#include "Network.h"

using namespace std;
using namespace arma;


class Diagnostics_MCNetworkSample {
private:
    vector<Network> networkSampleVec;
    int n_Node;

    vector<vector<double>> nodeDegreeDist_eachDegreeVec;
    vector<vector<double>> edgewiseSharedPartnerDist_eachDegreeVec;
    vector<vector<double>> userSpecific_eachVec;
    vector<Col<double>> summaryQuantile_nodeDegreeDist;
    vector<Col<double>> summaryQuantile_ESPDist;
    vector<Col<double>> summaryQuantile_userSpecific;

    void make_diagStat() {
        for (int i = 0; i < networkSampleVec.size(); i++) {
            Network net = networkSampleVec[i];
            Col<int> netNodeDegreeDist = net.get_nodeDegreeDist(); //1차원 높게나옴(n_Node+1)
            Col<int> netESPDist = net.get_edgewiseSharedPartnerDist();
            vector<double> userSpecific = { //<-추가로 얻고싶은 netStat을 집어넣을것. 이후 생성자에서 추가netStat 개수 설정
                (double)net.get_n_Edge(),
                (double)net.get_k_starDist(2)
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
    Diagnostics_MCNetworkSample() {

    }
    Diagnostics_MCNetworkSample(vector<Network> networkSampleVec) {
        this->networkSampleVec = networkSampleVec;
        this->n_Node = networkSampleVec[0].get_n_Node();
        this->nodeDegreeDist_eachDegreeVec.resize(n_Node);
        this->edgewiseSharedPartnerDist_eachDegreeVec.resize(n_Node - 1);
        this->userSpecific_eachVec.resize(2); // <- 여기에서 추가netStat 개수 설정!!
        make_diagStat();
        make_diagSummary();
    }


    void printResult() {
        for (int i = 0; i < summaryQuantile_nodeDegreeDist.size(); i++) {
            cout << "#node degree " << i << " :\n";
            cout << "at the param: " << summaryQuantile_nodeDegreeDist[i].t() << endl;
        }
        for (int i = 0; i < summaryQuantile_ESPDist.size(); i++) {
            cout << "#Edgewise Shared Partner  degree " << i << " :\n";
            cout << "at the param: " << summaryQuantile_ESPDist[i].t() << endl;
        }
        for (int i = 0; i < summaryQuantile_userSpecific.size(); i++) {
            cout << "#userSpecific index " << i << " :\n";
            cout << "at the param: " << summaryQuantile_userSpecific[i].t() << endl;
        }
    }

    void writeToCsv_Sample(string filename) {
        ofstream file;
        file.open(filename);

        //make header
        for (int c = 0; c < nodeDegreeDist_eachDegreeVec.size(); c++) {
            file << "nodeDegree" << c << ",";
        }
        for (int c = 0; c < edgewiseSharedPartnerDist_eachDegreeVec.size(); c++) {
            file << "edgewiseSharedPartner" << c << ",";
        }
        for (int c = 0; c < userSpecific_eachVec.size(); c++) {
            if (c == userSpecific_eachVec.size() - 1) {
                file << "userSpecific" << c;
            }
            else {
                file << "userSpecific" << c << ",";
            }
        }
        file << "\n";

        //fill values
        for (int i = 0; i < networkSampleVec.size(); i++) {
            for (int c = 0; c < nodeDegreeDist_eachDegreeVec.size(); c++) {
                file << nodeDegreeDist_eachDegreeVec[c][i] << ",";
            }
            for (int c = 0; c < edgewiseSharedPartnerDist_eachDegreeVec.size(); c++) {
                file << edgewiseSharedPartnerDist_eachDegreeVec[c][i] << ",";
            }
            for (int c = 0; c < userSpecific_eachVec.size(); c++) {
                if (c == userSpecific_eachVec.size() - 1) {
                    file << userSpecific_eachVec[c][i];
                }
                else {
                    file << userSpecific_eachVec[c][i] << ",";
                }
            }
            file << "\n";
        }
        file.close();
    }

};
