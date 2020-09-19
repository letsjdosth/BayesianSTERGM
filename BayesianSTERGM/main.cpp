#pragma once
#include <vector>
#include <iostream>
#include <armadillo>

#include "Network.h"
#include "netMCMCSampler.h"
#include "ERGM_MCML.h"
#include "BERGM_MCMC.h"
#include "MCdiagnostics.h"

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

//diagnostics ���
//FOR netMCMC-DIAG:
// 1. vector<Col<double>> netMCMCSampler::getDiagStatVec() ��, netStat col�� ���ܿ�� �߰�. ���� main���� �� �Լ� ����
// 2. ����, MCdiagnostics �����ڿ� �������
// ���߿�: (BERGM/ERGM)���� ������ ���÷� ������ �����ϴ� �Լ� ����

//����: thinning ���� (�Ф�for�������� �����ѵ�.. ���� �ȶ��ϰ� ��� ��������� ã��)
// traceplot

class temporalNetMCMCSampler{
private:

public:

};


int main()
{
    Mat<int> A = {
        {0,1,0,1,1, 0,0,1,1,0, 1,1,1,0,0, 1},
        {1,0,1,0,1, 1,1,0,0,0, 1,1,1,0,1, 0},
        {0,1,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 1},
        {1,0,0,0,0, 0,1,0,0,1, 0,0,0,1,1, 0},
        {1,1,0,0,0, 0,0,0,0,0, 1,1,1,0,0, 0},
        
        {0,1,0,0,0, 0,1,1,0,0, 0,0,1,0,0, 0},
        {0,1,0,1,0, 1,0,0,1,0, 0,0,0,1,0, 1},
        {1,0,0,0,0, 1,0,0,0,0, 1,1,0,1,1, 0},
        {1,0,0,0,0, 0,1,0,0,0, 0,0,1,1,1, 0},
        {0,0,0,1,0, 0,0,0,0,0, 1,0,1,0,0, 0},
        
        {1,1,0,0,1, 0,0,1,0,1, 0,0,1,0,0, 0},
        {1,1,0,0,1, 0,0,1,0,0, 0,0,1,1,1, 1},
        {1,1,0,0,1, 1,0,0,1,1, 1,1,0,1,0, 0},
        {0,0,0,1,0, 0,1,1,1,0, 0,1,1,0,1, 0},
        {0,1,0,1,0, 0,0,1,1,0, 0,1,0,1,0, 1},

        {1,0,1,0,0, 0,1,0,0,0, 0,1,0,0,1, 0}
    };

    Network netA = Network(A, false);
    // netA.printSummary();


    //// MCMCsampler test
    //Col<double> testParam = { 1, 2 };
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

    //ERGM test
    //Optimizer test
    /*Col<double> initParam = { 0.0 , 0.0 };
    ERGM_MCML OptimizerA(initParam, netA);
    OptimizerA.RunOptimize();
    OptimizerA.printDiagnosticVal();
    */


    ////BERGM test
    Col<double> initParam = { 0.0 , 0.0};
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
    bergmDiag.print_autoCorr(1, 30);
    
    return 0;
}
