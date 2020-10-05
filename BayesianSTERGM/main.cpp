#pragma once
#include <vector>
#include <iostream>
#include <fstream>
#include <armadillo>

#include "Network.h"
#include "netMCMCSampler.h"
#include "ERGM_MCML.h"
#include "BERGM_MCMC.h"
#include "MCdiagnostics.h"
#include "STERGMnetSampler.h"
#include "BSTERGM_MCMC.h"
#include "GoodnessOfFit_ERGM.h"
#include "GoodnessOfFit_STERGM.h"

using namespace std;
using namespace arma;

//model specification
//�ڵ�ȭ���ȵǾ.. �������θŹ��ٲ����(...)
//FOR ERGM:
// 1. netMCMCSampler.log_r �� model_delta(Col<double>)���� ���� ������ ���� '���� term'�� col�� element�� �߰�
// 2. ERGM_MCML.netOne_modelVal�� val�� �Ϲ� term�� col�� element�� �߰�
//FOR BERGM:
// 1. netMCMCSampler.log_r �� model_delta(Col<double>)���� ���� ������ ���� '���� term'�� col�� element�� �߰�
// 2. BERGM_MCMC.log_r�� model_delta(Col<double>)���� ���� ������ ���� '���� term'�� col�� element�� �߰�
// 3. �ʿ�� prior ���� (BERGM_MCMC.log_paramPriorPDF() ����)
//For BSTERGM:
// 1. STERGMnet1TimeMCSampler.log_accProb�� model_delta �� '���� term' ����
// 2. BSTERGM_MCMC�� log_r���� model delta 4���� '���� term' ����
// 3. �ʿ�� prior ���� (double log_paramPriorPDF(Col<double> param_formation, Col<double> param_dissolution) ����)

// MCMC ���ܹ��
//FOR BERGM/BSTERGM:
// �׳� posterior sample�� �����ڿ� �ѱ� �� (traceplot�� ���� write�Լ��� �Ἥ R���� �ҷ���)
//FOR netMCMC-DIAG:
// 1. vector<Col<double>> netMCMCSampler::getDiagStatVec() ��, netStat col�� ���ܿ�� �߰�. ���� main���� �� �Լ� ����
// 2. ����, MCdiagnostics �����ڿ� �������
// ���߿�: (BERGM/ERGM)���� ������ ���÷� ������ �����ϴ� �Լ� ����

// GoF ���
// For ERGM/BERGM:
// 1. netMCMCSample.log_r���� �� Ȯ�� (fitting���� �𵨰� ���ƾ� ��. ������ (������) 1���� �Ǿ��ִٸ� �ٲ��� �ʾƵ� ��
// 2. GoodnessOfFItERGM ������/make_diagStat�� �°� ����ְ�, run ���� printSummary ȣ�� (�⺻������ MC�� 0-MAT���� ������)
// For BSTERGM
// 1. STERGMnet1TimeSampler Ŭ������ log_accProb���� �� Ȯ�� (fitting���� �𵨰� ���ƾ� ��)
// 2. GoodnessOfFit_1time_STERGM ������/make_diagStat�� �߰��� ��������� �°� ����ְ�, GoodnessOfFit_STERGM �ν��Ͻ� �� 
// ���۽ð����� run


// B-STERGM����
// n_Node + k_starDist(2) -> MCMC ����
//����: 
// 1. STERGMnetMCSampler �˰���üũ�� �� ������� CPP �и�
// 2. BSTERGM_mcmc �˰���üũ�� ������� CPP �и�
// 3. GoodnessOfFit_STERGM �˰��� üũ �� ������� CPP �и�

class netMCMCSamplerDiagnostics {
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
            Col<int> netNodeDegreeDist = net.get_nodeDegreeDist(); //1���� ���Գ���(n_Node+1)
            Col<int> netESPDist = net.get_edgewiseSharedPartnerDist();
            vector<double> userSpecific = { //<-�߰��� ������ netStat�� ���������. ���� �����ڿ��� �߰�netStat ���� ����
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
    netMCMCSamplerDiagnostics() {

    }
    netMCMCSamplerDiagnostics(vector<Network> networkSampleVec) {
        this->networkSampleVec = networkSampleVec;
        this->n_Node = networkSampleVec[0].get_n_Node();
        this->nodeDegreeDist_eachDegreeVec.resize(n_Node);
        this->edgewiseSharedPartnerDist_eachDegreeVec.resize(n_Node - 1);
        this->userSpecific_eachVec.resize(3); // <- ���⿡�� �߰�netStat ���� ����!!
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


int main()
{
    //=================================================================================================
    // Sample symmetric matrix
    //Mat<int> AA = {
    //    {0,1,0,1,1, 0,0,1,1,0, 1,1,1,0,0, 1},
    //    {1,0,1,0,1, 1,1,0,0,0, 1,1,1,0,1, 0},
    //    {0,1,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 1},
    //    {1,0,0,0,0, 0,1,0,0,1, 0,0,0,1,1, 0},
    //    {1,1,0,0,0, 0,0,0,0,0, 1,1,1,0,0, 0},
    //    
    //    {0,1,0,0,0, 0,1,1,0,0, 0,0,1,0,0, 0},
    //    {0,1,0,1,0, 1,0,0,1,0, 0,0,0,1,0, 1},
    //    {1,0,0,0,0, 1,0,0,0,0, 1,1,0,1,1, 0},
    //    {1,0,0,0,0, 0,1,0,0,0, 0,0,1,1,1, 0},
    //    {0,0,0,1,0, 0,0,0,0,0, 1,0,1,0,0, 0},
    //    
    //    {1,1,0,0,1, 0,0,1,0,1, 0,0,1,0,0, 0},
    //    {1,1,0,0,1, 0,0,1,0,0, 0,0,1,1,1, 1},
    //    {1,1,0,0,1, 1,0,0,1,1, 1,1,0,1,0, 0},
    //    {0,0,0,1,0, 0,1,1,1,0, 0,1,1,0,1, 0},
    //    {0,1,0,1,0, 0,0,1,1,0, 0,1,0,1,0, 1},

    //    {1,0,1,0,0, 0,1,0,0,0, 0,1,0,0,1, 0}
    //};
    Mat<int> A = {
        {0,1,0,1,1},
        {1,0,1,0,1},
        {0,1,0,0,0},
        {1,0,0,0,0},
        {1,1,0,0,0} };
    Mat<int> B = {
        {0,1,0,1,1},
        {1,0,0,0,1},
        {0,0,0,0,1},
        {1,0,0,0,0},
        {1,1,1,0,0} };
    Mat<int> C = {
        {0,0,0,0,1},
        {0,0,1,1,1},
        {0,1,0,0,1},
        {0,1,0,0,1},
        {1,1,1,1,0} };
    Mat<int> floBusiness = {
        {0,1,1,0,0, 0,0,0,0,0, 0,0,0,0,0, 0},
        {1,0,1,1,0, 0,0,0,0,0, 0,0,0,0,0, 0},
        {1,1,0,1,1, 0,0,0,0,0, 0,0,0,0,0, 0},
        {0,1,1,0,1, 1,0,0,0,0, 0,0,0,0,0, 0},
        {0,0,1,1,0, 1,0,0,0,0, 0,0,0,0,0, 0},

        {0,0,0,1,1, 0,1,1,0,0, 0,0,0,0,0, 0},
        {0,0,0,0,0, 1,0,1,0,0, 0,0,0,0,0, 0},
        {0,0,0,0,0, 1,1,0,1,1, 1,0,0,0,0, 0},
        {0,0,0,0,0, 0,0,1,0,0, 0,0,0,0,0, 0},
        {0,0,0,0,0, 0,0,1,0,0, 0,0,0,0,0, 0},

        {0,0,0,0,0, 0,0,1,0,0, 0,0,0,0,0, 0},
        {0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0},
        {0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0},
        {0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0},
        {0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0},

        {0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0}
    };
    Network netA = Network(A, false);
    //netA.printSummary();
    Network netB = Network(B, false);
    Network netC = Network(C, false);
    vector<Network> netSeq = { netA, netB, netC };
    Network netFloBusiness = Network(floBusiness, false);
    //netFloBusiness.printSummary();
    //=================================================================================================
    //// stergm sampler : STERGMnet1TimeSampler test
    //Col<double> testParam1 = { 0.2, 0.1 };
    //Col<double> testParam2 = { -0.2,-0.1 };

    //STERGMnet1TimeSampler tsampler = STERGMnet1TimeSampler(testParam1, testParam2, netA);
    //tsampler.generateSample(1000);
    //tsampler.cutBurnIn(994);
    //tsampler.testOut();
    //=================================================================================================
    //// stergm sampler : STERGMnetSeqSampler test
    //Col<double> testParam1 = { 0.2, 0.1 };
    //Col<double> testParam2 = { -0.2,-0.1 };

    //STERGMnetSeqSampler Tsampler = STERGMnetSeqSampler(testParam1, testParam2, netA, 3);
    //Tsampler.generateSample(10, 1000);
    //Tsampler.printResult(0);
    //Tsampler.printResult(1);
    //Tsampler.printResult(2);
    //Tsampler.printResult(3);

    //=================================================================================================
    ////BSTERGM test
    //
    //Col<double> testParam1 = { 0.2, 0.1 };
    //Col<double> testParam2 = { -0.2,-0.1 };
    //BSTERGM_MCMC Bstergm = BSTERGM_MCMC(testParam1, testParam2, netSeq);
    //Bstergm.generateSample(500000);
    //Bstergm.cutBurnIn(200000);
    //Bstergm.thinning(1000);
    ////Bstergm.testOut();
    //
    //MCdiagnostics BstergmDiag1(Bstergm.getPosteriorSample_formation());
    //BstergmDiag1.print_mean(0);
    //Col<double> quantilePts = { 0.1, 0.25, 0.5, 0.75, 0.9 };
    //BstergmDiag1.print_quantile(0, quantilePts);
    //BstergmDiag1.print_autoCorr(0, 30);
    //BstergmDiag1.print_mean(1);
    //BstergmDiag1.print_quantile(1, quantilePts);
    //BstergmDiag1.print_autoCorr(1, 30);

    //MCdiagnostics BstergmDiag2(Bstergm.getPosteriorSample_dissolution());
    //BstergmDiag2.print_mean(0);
    //BstergmDiag2.print_quantile(0, quantilePts);
    //BstergmDiag2.print_autoCorr(0, 30);
    //BstergmDiag2.print_mean(1);
    //BstergmDiag2.print_quantile(1, quantilePts);
    //BstergmDiag2.print_autoCorr(1, 30);

    //BstergmDiag1.writeToCsv_Sample("formation.csv");
    //BstergmDiag2.writeToCsv_Sample("dissolution.csv");
    //

    //GoodnessOfFit_STERGM BstergmGoF (BstergmDiag1.get_mean(), BstergmDiag2.get_mean(), netSeq);
    //cout << "t=0 to t=1" << endl;
    //BstergmGoF.run(0, 30000);
    //cout << "\n\nt=1 to t=2" << endl;
    //BstergmGoF.run(1, 30000);
    //

    //=================================================================================================
    // netMCMCsampler test
    ////now n_edge -2.88316 k2-star 0.229221 (in degeneracy region)
    //netFloBusiness.printSummary();
    //Col<double> testParam = { -2.88316, 0.229221 };
    //netMCMCSampler sampler(testParam, netFloBusiness);
    //sampler.generateSample(100000);
    ////sampler.testOut();
    //sampler.cutBurnIn(50000);
    //vector<Col<double>> diagNetVec = sampler.getDiagStatVec();
    //for (int i = 0; i < diagNetVec.size(); i++) {
    //    cout << diagNetVec[i].t() << endl;
    //}

    //MCdiagnostics netMCMCDiag(diagNetVec);
    //Col<double> quantilePts = { 0.1, 0.25, 0.5, 0.75, 0.9 };
    //for (int idx = 0; idx < diagNetVec[0].size(); idx++) {
    //    netMCMCDiag.print_mean(idx);
    //    netMCMCDiag.print_quantile(idx, quantilePts);
    //    netMCMCDiag.print_autoCorr(idx, 50);
    //}
    //cout << netA.get_n_Edge() << " " << netA.get_geoWeightedNodeDegree(0.3) << " " << netA.get_geoWeightedESP(0.3) << endl;
 
    //netMCMCSamplerDiagnostics netDiag = netMCMCSamplerDiagnostics(sampler.getMCMCSampleVec());
    //netDiag.printResult();

    //=================================================================================================
    ////ERGM test
    ////Optimizer test
    //Col<double> initParam = { 0.01 , -0.01 };
    //ERGM_MCML OptimizerA(initParam, netA);
    //OptimizerA.RunOptimize();
    //OptimizerA.print_checkConvergence();
    //
    //Col<double> optimizedParam = OptimizerA.getMCMLE(); //{ 0.3908, -0.1190 } for n_edge  k-star(2)
    //Mat<int> zeroMat(5, 5, fill::zeros);
    //Network zeroNet(zeroMat, 0);
    //
    ////ERGM GoF
    //GoodnessOfFit_ERGM gofERGMdiag2 = GoodnessOfFit_ERGM(netA, OptimizerA.getMCMLE());
    //gofERGMdiag2.run(100000, 90000);
    //gofERGMdiag2.printResult();

    //=================================================================================================
    ////BERGM test
    //Col<double> initParam = { 0.01 , -0.01 };
    //BERGM_MCMC bergm(initParam, netFloBusiness);
    //bergm.generateSample(25000, 200);
    //bergm.cutBurnIn(10000);
    //bergm.thinning(30);
    //
    ////BERGM MCMCDIAG
    //MCdiagnostics bergmDiag(bergm.getPosteriorSample());
    //bergmDiag.print_mean(0);
    //Col<double> quantilePts = { 0.1, 0.25, 0.5, 0.75, 0.9 };
    //bergmDiag.print_quantile(0, quantilePts);
    //bergmDiag.print_autoCorr(0, 30);
    //bergmDiag.print_mean(1);
    //bergmDiag.print_quantile(1, quantilePts);
    //bergmDiag.print_autoCorr(1, 30);
    //bergmDiag.writeToCsv_Sample("bergmPosteriorSample.csv");

    ////BERGM GOF
    ////����
    ////SAMCMC n_edge : -2.733, k2-star : 0.198
    ////MCMLE n_edge : -3.191, k2-star : 0.412
    ////SAA n_edge : -2.842, k2-star : 0.283
    ////now n_edge -2.88316 k2-star 0.229221

    // Col<double> fittedParam = bergmDiag.get_mean();
    Col<double> fittedParam = { -2.88316, 0.229221 }; // ����

    GoodnessOfFit_ERGM gofBERGMdiagF = GoodnessOfFit_ERGM(netFloBusiness, fittedParam);
    gofBERGMdiagF.run(50000, 25000);
    gofBERGMdiagF.printResult();
    netFloBusiness.printSummary();

    ////BERGM LAST Exchange Sampler diag
    //netMCMCSampler lastExNetSampler = bergm.get_lastExchangeNetworkSampler();
    //netMCMCSamplerDiagnostics lastExNetDiag = netMCMCSamplerDiagnostics(lastExNetSampler.getMCMCSampleVec());
    //lastExNetDiag.writeToCsv_Sample("lastExNetSamplerNetworkStats.csv");
    //// lastExNetDiag.printResult();

    return 0;
}
