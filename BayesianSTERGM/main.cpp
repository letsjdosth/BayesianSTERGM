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
#include "GoodnessOfFit_STERGM.h"

using namespace std;
using namespace arma;

//model specification
//자동화가안되어서.. 수동으로매번바꿔야함(...)
//FOR ERGM:
// 1. netMCMCSampler.log_r 의 model_delta(Col<double>)에서 모델항 각각에 대한 '차이 term'을 col의 element로 추가
// 2. ERGM_MCML.netOne_modelVal의 val에 일반 term을 col의 element로 추가
//FOR BERGM:
// 1. netMCMCSampler.log_r 의 model_delta(Col<double>)에서 모델항 각각에 대한 '차이 term'을 col의 element로 추가
// 2. BERGM_MCMC.log_r의 model_delta(Col<double>)에서 모델항 각각에 대한 '차이 term'을 col의 element로 추가
// 3. 필요시 prior 조정 (BERGM_MCMC.log_paramPriorPDF() 구현)
//For BSTERGM:
// 1. STERGMnet1TimeMCSampler.log_accProb의 model_delta 항 '차이 term' 수정
// 2. BSTERGM_MCMC의 log_r에서 model delta 4개항 '차이 term' 수정
// 3. 필요시 prior 조정 (double log_paramPriorPDF(Col<double> param_formation, Col<double> param_dissolution) 구현)

// MCMC 진단방법
//FOR BERGM/BSTERGM:
// 그냥 posterior sample을 생성자에 넘길 것 (traceplot은 이후 write함수로 써서 R에서 불러서)
//FOR netMCMC-DIAG:
// 1. vector<Col<double>> netMCMCSampler::getDiagStatVec() 의, netStat col에 진단요소 추가. 이후 main에서 이 함수 실행
// 2. 이후, MCdiagnostics 생성자에 집어넣자
// 나중에: (BERGM/ERGM)에서 마지막 샘플러 꺼내서 조사하는 함수 구현

// GoF 방법
// For ERGM/BERGM:
// 1. netMCMCSample.log_r에서 모델 확인 (fitting시의 모델과 같아야 함. 위에서 (각각의) 1번이 되어있다면 바꾸지 않아도 됨
// 2. GoodnessOfFItERGM 생성자/make_diagStat에 맞게 집어넣고, run 이후 printSummary 호출 (기본적으로 MC는 0-MAT에서 시작함)
// For BSTERGM
// 1. STERGMnet1TimeSampler 클래스의 log_accProb에서 모델 확인 (fitting시의 모델과 같아야 함)
// 2. GoodnessOfFit_1time_STERGM 생성자/make_diagStat에 추가로 보고싶은것 맞게 집어넣고, GoodnessOfFit_STERGM 인스턴스 찍어서 
// 시작시간별로 run


// B-STERGM문제
// n_Node + k_starDist(2) -> MCMC 터짐
//할일: 
// 1. STERGMnetMCSampler 알고리즘체크후 후 헤더에서 CPP 분리
// 2. BSTERGM_mcmc 알고리즘체크후 헤더에서 CPP 분리


class netMCMCSamplerDiagnostics {

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
    //// netMCMCsampler test
    //Col<double> testParam = { 0.2, 0.1 };
    //netMCMCSampler sampler(testParam, netA);
    //sampler.generateSample(100000);
    ////sampler.testOut();
    //sampler.cutBurnIn(98000);
    //cout << "after burnin" << endl;
    ////sampler.testOut();
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
    //BERGM test
    Col<double> initParam = { 0.01 , -0.01 };
    BERGM_MCMC bergm(initParam, netFloBusiness);
    bergm.generateSample(30000, 100);
    bergm.cutBurnIn(10000);
    bergm.thinning(20);
    
    //BERGM MCMCDIAG
    MCdiagnostics bergmDiag(bergm.getPosteriorSample());
    bergmDiag.print_mean(0);
    Col<double> quantilePts = { 0.1, 0.25, 0.5, 0.75, 0.9 };
    bergmDiag.print_quantile(0, quantilePts);
    bergmDiag.print_autoCorr(0, 30);
    bergmDiag.print_mean(1);
    bergmDiag.print_quantile(1, quantilePts);
    bergmDiag.print_autoCorr(1, 30);
    bergmDiag.writeToCsv_Sample("bergmPosteriorSample.csv");

    //BERGM GOF
    // {-2.19095, 0.0456346} (3000개 돌렸을때)
    GoodnessOfFit_ERGM gofBERGMdiagF = GoodnessOfFit_ERGM(netFloBusiness, bergmDiag.get_mean());
    gofBERGMdiagF.run(10000, 1000);
    gofBERGMdiagF.printResult();
    netFloBusiness.printSummary();

    //BERGM LAST Exchange Sampler diag
    netMCMCSampler lastExNetSampler = bergm.get_lastExchangeNetworkSampler();
    //음...
    

    return 0;
}
