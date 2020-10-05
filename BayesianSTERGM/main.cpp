#pragma once
#include <vector>
#include <iostream>
#include <armadillo>

#include "Network.h"
#include "netMCMCSampler.h"
#include "ERGM_MCML.h"
#include "BERGM_MCMC.h"
#include "MCdiagnostics.h"
#include "STERGMnetSampler.h"
#include "BSTERGM_MCMC.h"
#include "GoodnessOfFit_ERGM.h"

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

    void netGenerate(int num_smpl) {
        STERGMnet1TimeSampler stepGoF = STERGMnet1TimeSampler(fittedParam_formation, fittedParam_dissolution, obsStartNet);
        for (int i = 0; i < num_smpl; i++) {
            stepGoF.generateSample();
            gofSampleVec.push_back(stepGoF.get_CombinedNetMCMCSample());
        }
    }
    
    void make_diagStat() {
        for (int i = 0; i < gofSampleVec.size(); i++) {
            Network net = gofSampleVec[i];
            //diag netstat ����
            Col<int> netNodeDegreeDist = net.get_nodeDegreeDist(); //1���� ���Գ���(n_Node+1)
            Col<int> netESPDist = net.get_edgewiseSharedPartnerDist();
            vector<double> userSpecific = { //<-�߰��� ������ netStat�� ���������. ���� �����ڿ��� �߰�netStat ���� ����
                (double)net.get_n_Edge(),
                net.get_geoWeightedNodeDegree(0.3),
                net.get_geoWeightedESP(0.3)
            };

            //diag netstat ���
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
    void make_diagSummary(){
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
        this->obsStartNet= obsStartNet;
        this->obsNextNet= obsNextNet;
        this->n_Node = obsStartNet.get_n_Node();
        this->fittedParam_formation = fittedParam_formation;
        this->fittedParam_dissolution = fittedParam_dissolution;
        this->nodeDegreeDist_eachDegreeVec.resize(n_Node);
        this->edgewiseSharedPartnerDist_eachDegreeVec.resize(n_Node - 1);
        this->userSpecific_eachVec.resize(3); // <- ���⿡�� �߰�netStat ���� ����!!
    }
    void run(int n_smpl) {
        netGenerate(n_smpl);
        make_diagStat();
        make_diagSummary();
    }
    void printResult() {
        Col<int> obsNodeDegree = obsNextNet.get_nodeDegreeDist();
        Col<int> obsESP = obsNextNet.get_edgewiseSharedPartnerDist();
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
    void run(int startTime, int n_smpl) {
        GoodnessOfFit_1time_STERGM gofRunner = GoodnessOfFit_1time_STERGM(fittedParam_formation, fittedParam_dissolution,
            obsNetSeq[startTime], obsNetSeq[startTime + 1]);
        gofRunner.run(n_smpl);
        gofRunner.printResult();
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
    Col<double> testParam = { 0.2, 0.1 };
    netMCMCSampler sampler(testParam, netA);
    sampler.generateSample(100000);
    //sampler.testOut();
    sampler.cutBurnIn(98000);
    cout << "after burnin" << endl;
    //sampler.testOut();
    vector<Col<double>> diagNetVec = sampler.getDiagStatVec();
    for (int i = 0; i < diagNetVec.size(); i++) {
        cout << diagNetVec[i].t() << endl;
    }

    MCdiagnostics netMCMCDiag(diagNetVec);
    Col<double> quantilePts = { 0.1, 0.25, 0.5, 0.75, 0.9 };
    for (int idx = 0; idx < diagNetVec[0].size(); idx++) {
        netMCMCDiag.print_mean(idx);
        netMCMCDiag.print_quantile(idx, quantilePts);
        netMCMCDiag.print_autoCorr(idx, 50);
    }
    cout << netA.get_n_Edge() << " " << netA.get_geoWeightedNodeDegree(0.3) << " " << netA.get_geoWeightedESP(0.3) << endl;
 
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
    //bergm.generateSample(10000, 1000);
    //bergm.cutBurnIn(5000);
    //bergm.thinning(25);
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
    //// {-2.19095, 0.0456346} (3000�� ��������)
    //GoodnessOfFit_ERGM gofBERGMdiagF = GoodnessOfFit_ERGM(netFloBusiness, bergmDiag.get_mean());
    //gofBERGMdiagF.run(10000, 1000);
    //gofBERGMdiagF.printResult();
    //netFloBusiness.printSummary();

    return 0;
}
