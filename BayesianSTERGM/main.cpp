#pragma once
#include <vector>
#include <iostream>
#include <armadillo>

#include "Network.h"
#include "netMCMCSampler.h"
#include "ERGM_MCML.h"
#include "BERGM_MCMC.h"
#include "MCdiagnostics.h"
#include "STERGMnetMCSampler.h"

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
//For STERGM:
// 1. STERGMnet1TimeMCSampler.log_r의 model_delta_formation, model_delta_dissolution 모델항 '차이 term' 수정


//diagnostics 방법
//FOR netMCMC-DIAG:
// 1. vector<Col<double>> netMCMCSampler::getDiagStatVec() 의, netStat col에 진단요소 추가. 이후 main에서 이 함수 실행
// 2. 이후, MCdiagnostics 생성자에 집어넣자
// 나중에: (BERGM/ERGM)에서 마지막 샘플러 꺼내서 조사하는 함수 구현

//할일: thinning 구현 (ㅠㅠfor문돌리면 쉽긴한데.. 보다 똑똑하게 어떻게 방법없는지 찾기)
// traceplot


class BSTERGM {
private:
public:

};


int main()
{
    //Mat<int> A = {
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

    
    Network netA = Network(A, false);
    //netA.printSummary();
    
    
    
    //// stergm sampler : STERGMnet1TimeMCSampler test
    //Col<double> testParam1 = { 0.2, 0.1 };
    //Col<double> testParam2 = { -0.2,-0.1 };

    //STERGMnet1TimeMCSampler tsampler = STERGMnet1TimeMCSampler(testParam1, testParam2, netA);
    //tsampler.generateSample(1000);
    //tsampler.cutBurnIn(994);
    //tsampler.testOut();

    // stergm sampler : STERGMnetMCSampler
    Col<double> testParam1 = { 0.2, 0.1 };
    Col<double> testParam2 = { -0.2,-0.1 };

    STERGMnetMCSampler Tsampler = STERGMnetMCSampler(testParam1, testParam2, netA, 3);
    Tsampler.generateSample(10, 1000);
    Tsampler.printResult(0);
    Tsampler.printResult(1);
    Tsampler.printResult(2);
    Tsampler.printResult(3);
    

    //// MCMCsampler test
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
 
    //ERGM test
    //Optimizer test
    /*Col<double> initParam = { 0.0 , 0.0 };
    ERGM_MCML OptimizerA(initParam, netA);
    OptimizerA.RunOptimize();
    OptimizerA.printDiagnosticVal();
    */


    ////BERGM test
    /*Col<double> initParam = { 0.0 , 0.0};
    BERGM_MCMC bergm(initParam, netA);
    bergm.generateSample(500, 100);
    bergm.cutBurnIn(250);
    

    MCdiagnostics bergmDiag(bergm.getPosteriorSample());
    bergmDiag.print_mean(0);
    Col<double> quantilePts = { 0.1, 0.25, 0.5, 0.75, 0.9 };
    bergmDiag.print_quantile(0, quantilePts);
    bergmDiag.print_autoCorr(0, 30);
    bergmDiag.print_mean(1);
    bergmDiag.print_quantile(1, quantilePts);
    bergmDiag.print_autoCorr(1, 30);*/
    
    

    return 0;
}
