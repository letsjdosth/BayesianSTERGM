#pragma once
#include <vector>
#include <iostream>
#include <armadillo>

#include "Network.h"
#include "NetMCMCSampler_ERGM.h"
#include "ERGM_MCML.h"
#include "BERGM_MCMC.h"
#include "GoodnessOfFit_ERGM.h"

#include "NetMCMCSampler_STERGM.h"
#include "BSTERGM_MCMC.h"
#include "GoodnessOfFit_STERGM.h"

#include "Diagnostics_MCParamSample.h"
#include "Diagnostics_MCNetworkSample.h"


using namespace std;
using namespace arma;

//model specification
//�ڵ�ȭ���ȵǾ.. �������θŹ��ٲ����(...)
//FOR ERGM:
// 1. NetMCMCSampler_ERGM.log_r �� model_delta(Col<double>)���� ���� ������ ���� '���� term'�� col�� element�� �߰�
// 2. ERGM_MCML.netOne_modelVal�� val�� �Ϲ� term�� col�� element�� �߰�
//FOR BERGM:
// 1. NetMCMCSampler_ERGM.log_r �� model_delta(Col<double>)���� ���� ������ ���� '���� term'�� col�� element�� �߰�
// 2. BERGM_MCMC.log_r�� model_delta(Col<double>)���� ���� ������ ���� '���� term'�� col�� element�� �߰�
// 3. �ʿ�� prior ���� (BERGM_MCMC.log_paramPriorPDF() ����)
//For BSTERGM:
// 1. STERGMnet1TimeMCSampler.log_accProb�� model_delta �� '���� term' ����
// 2. BSTERGM_MCMC�� log_r���� model delta 4���� '���� term' ����
// 3. �ʿ�� prior ���� (double log_paramPriorPDF(Col<double> param_formation, Col<double> param_dissolution) ����)

// MCMC ���ܹ��
//FOR BERGM/BSTERGM:
// �׳� posterior sample�� Diagnostics_MCParamSample �����ڿ� �ѱ� �� (traceplot�� ���� write�Լ��� �Ἥ R���� �ҷ���)
//FOR netMCMC-DIAG:
// 1. (vector<Col<double>> NetMCMCSampler_ERGM::getDiagStatVec() ��, netStat col�� ���ܿ�� �߰�. ���� main���� �� �Լ� ����
// 2. ����, Diagnostics_MCParamSample �����ڿ� �������)
// 3. Diagnostics_MCNetworkSample�� network vector ������� (BERGM�� �ش� get �Լ� �����ص�)
//(��������: 1,2�� �۾��� 3�� Ŭ�������� ��ü������ �ϵ��� ������. ���� �� �������̽��� �����ɵ�)


// GoF ���
// For ERGM/BERGM:
// 1. netMCMCSample.log_r���� �� Ȯ�� (fitting���� �𵨰� ���ƾ� ��. ������ (������) 1���� �Ǿ��ִٸ� �ٲ��� �ʾƵ� ��
// 2. GoodnessOfFItERGM ������/make_diagStat�� �°� ����ְ�, run ���� printSummary ȣ�� (�⺻������ MC�� 0-MAT���� ������)
// For BSTERGM
// 1. STERGMnet1TimeSampler Ŭ������ log_accProb���� �� Ȯ�� (fitting���� �𵨰� ���ƾ� ��)
// 2. GoodnessOfFit_1time_STERGM ������/make_diagStat�� �߰��� ��������� �°� ����ְ�, GoodnessOfFit_STERGM �ν��Ͻ� �� 
// ���۽ð����� run


//����: 
// 1. STERNET.org�� stergm ����ü��, formation�� dissolution�� ���� �ٸ� ���� ���� �Ǿ�����. �̰� BSTERGM���� �����غ��� ������ ������
// 2. BSTERGM prediction-simulation(at fitted param or each param sample value) 
// -> long-term dynamic �̾Ƽ� �� netStat ts.plot�׸��� �� ������� duration/incidence/prevalence ���
// (prevalence ~ incidence * duration (�ٻ�������) <- �ƴϸ�׳� sample���� ���� ����ص��ɵ�

// ��Ÿ: NetMCMCSampler_STERGM ����? ���� ���ÿ����� ���ʿ� x 
// (������ �ǻ� indep mcmc��. edge�� formation/dissolution���� �ϳ��� �ְ��Ÿ� �Ҽ����ְڴµ�...)

//��Ÿ����
// 1. STERGMnetMCSampler �˰���üũ�� �� ������� CPP �и�
// 2. BSTERGM_mcmc �˰���üũ�� ������� CPP �и�
// 3. GoodnessOfFit_STERGM �˰��� üũ �� ������� CPP �и�



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
    //Col<double> testParam1 = { 1, 1 };
    //Col<double> testParam2 = { -1,-1 };

    //STERGMnet1TimeSampler tsampler = STERGMnet1TimeSampler(testParam1, testParam2, netA);
    //tsampler.generateSample();
    //cout << netA.get_netStructure() << endl;
    //cout << tsampler.get_FormationNetMCMCSample().get_netStructure() << endl;
    //cout << tsampler.get_DissolutionNetMCMCSample().get_netStructure() << endl;
    //cout << tsampler.get_CombinedNetMCMCSample().get_netStructure() << endl;
    
    //=================================================================================================
    //// stergm sampler : STERGMnetSeqSampler test
    //Col<double> testParam1 = { 0.2, 0.1 };
    //Col<double> testParam2 = { -0.2,-0.1 };

    //STERGMnetSeqSampler Tsampler = STERGMnetSeqSampler(testParam1, testParam2, 5, netA);
    //Tsampler.generateSample(4);
    //Tsampler.printResult(0);
    //Tsampler.printResult(1);
    //Tsampler.printResult(2);
    //Tsampler.printResult(3);

    //=================================================================================================
    //BSTERGM test
    
    Col<double> testParam1 = { 0.2, 0.1 };
    Col<double> testParam2 = { -0.2,-0.1 };
    BSTERGM_MCMC Bstergm = BSTERGM_MCMC(testParam1, testParam2, netSeq);
    Bstergm.generateSample(500000);
    Bstergm.cutBurnIn(200000);
    Bstergm.thinning(1000);
    //Bstergm.testOut();
    
    Diagnostics_MCParamSample BstergmDiag1(Bstergm.getPosteriorSample_formation());
    BstergmDiag1.print_mean(0);
    Col<double> quantilePts = { 0.1, 0.25, 0.5, 0.75, 0.9 };
    BstergmDiag1.print_quantile(0, quantilePts);
    BstergmDiag1.print_autoCorr(0, 30);
    BstergmDiag1.print_mean(1);
    BstergmDiag1.print_quantile(1, quantilePts);
    BstergmDiag1.print_autoCorr(1, 30);

    Diagnostics_MCParamSample BstergmDiag2(Bstergm.getPosteriorSample_dissolution());
    BstergmDiag2.print_mean(0);
    BstergmDiag2.print_quantile(0, quantilePts);
    BstergmDiag2.print_autoCorr(0, 30);
    BstergmDiag2.print_mean(1);
    BstergmDiag2.print_quantile(1, quantilePts);
    BstergmDiag2.print_autoCorr(1, 30);

    BstergmDiag1.writeToCsv_Sample("formation.csv");
    BstergmDiag2.writeToCsv_Sample("dissolution.csv");
    

    GoodnessOfFit_STERGM BstergmGoF (BstergmDiag1.get_mean(), BstergmDiag2.get_mean(), netSeq);
    cout << "t=0 to t=1" << endl;
    BstergmGoF.run(0, 30000);
    cout << "\n\nt=1 to t=2" << endl;
    BstergmGoF.run(1, 30000);
    



    //=================================================================================================
    //=================================================================================================
    // NetMCMCsampler_ERGM test
    ////now n_edge -2.88316 k2-star 0.229221 (in degeneracy region)
    //netFloBusiness.printSummary();
    //Col<double> testParam = { -2.88316, 0.229221 };
    //NetMCMCSampler_ERGM sampler(testParam, netFloBusiness);
    //sampler.generateSample(100000);
    ////sampler.testOut();
    //sampler.cutBurnIn(50000);
    //vector<Col<double>> diagNetVec = sampler.getDiagStatVec();
    //for (int i = 0; i < diagNetVec.size(); i++) {
    //    cout << diagNetVec[i].t() << endl;
    //}

    //Diagnostics_MCParamSample netMCMCDiag(diagNetVec);
    //Col<double> quantilePts = { 0.1, 0.25, 0.5, 0.75, 0.9 };
    //for (int idx = 0; idx < diagNetVec[0].size(); idx++) {
    //    netMCMCDiag.print_mean(idx);
    //    netMCMCDiag.print_quantile(idx, quantilePts);
    //    netMCMCDiag.print_autoCorr(idx, 50);
    //}
    //cout << netA.get_n_Edge() << " " << netA.get_geoWeightedNodeDegree(0.3) << " " << netA.get_geoWeightedESP(0.3) << endl;

    //Diagnostics_MCNetworkSample netDiag = Diagnostics_MCNetworkSample(sampler.getMCMCSampleVec());
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
    //Diagnostics_MCParamSample bergmDiag(bergm.getPosteriorSample());
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

    // Col<double> fittedParam = bergmDiag.get_mean(); //�ڵ� (�����ϳ��� �ּ�����)
    //Col<double> fittedParam = { -2.88316, 0.229221 }; // ����

    //GoodnessOfFit_ERGM gofBERGMdiagF = GoodnessOfFit_ERGM(netFloBusiness, fittedParam);
    //gofBERGMdiagF.run(50000, 25000);
    //gofBERGMdiagF.printResult();
    //netFloBusiness.printSummary();

    ////BERGM LAST Exchange Sampler diag
    //NetMCMCSampler_ERGM lastExNetSampler = bergm.get_lastExchangeNetworkSampler();
    //Diagnostics_MCNetworkSample lastExNetDiag = Diagnostics_MCNetworkSample(lastExNetSampler.getMCMCSampleVec());
    //lastExNetDiag.writeToCsv_Sample("lastExNetSamplerNetworkStats.csv");
    //// lastExNetDiag.printResult();

    return 0;
}
