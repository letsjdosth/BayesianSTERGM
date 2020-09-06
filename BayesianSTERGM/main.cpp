#pragma once
#include <vector>
#include <iostream>
#include <armadillo>

#include "Network.h"
#include "netMCMCSampler.h"
#include "ERGM_MCML.h"
#include "BERGM_MCMC.h"

using namespace std;
using namespace arma;

//model specification
//�ڵ�ȭ���ȵǾ.. �������θŹ��ٲ����(...)
//FOR ERGM:
// 1. netMCMCSampler.log_r �� model_delta(Col<double>)���� ���� ������ ���� '���� term'�� col�� element�� �߰�
// 2. ERGM_MCML.NRupdate1Step��, MCMCSampleVec�� ����� for�� �� val(Col<double>)�� �Ϲ� term�� col�� element�� �߰�
//FOR BERGM:
// 1. netMCMCSampler.log_r �� model_delta(Col<double>)���� ���� ������ ���� '���� term'�� col�� element�� �߰�
// 2. BERGM_MCMC.log_r�� model_delta(Col<double>)���� ���� ������ ���� '���� term'�� col�� element�� �߰�
// 3. �ʿ�� prior ���� (BERGM_MCMC.log_paramPriorPDF() ����)

int main()
{
    Mat<int> A = { {0,1,1,0,0,1},
                    {1,0,1,0,0,0},
                    {1,1,0,0,1,0},
                    {0,0,0,0,0,1},
                    {0,0,1,0,0,1},
                    {1,0,0,1,1,0} }
    ;

    Network netA = Network(A, false);
    netA.printSummary();


    ////MCMCsampler test
    //Col<double> testParam = { 0.1 };
    //netMCMCSampler sampler(testParam, netA);
    //sampler.generateSample(10);
    //sampler.testOut();
    //sampler.cutBurnIn(8);
    //cout << "after burnin" << endl;
    //sampler.testOut();

    ////Optimizer test
    //Col<double> initParam = { 0.0 };
    //ERGM_MCML OptimizerA(initParam, netA);
    //OptimizerA.RunOptimize();

    //Col<double> testvec = { 1,2,3,4,5 };
    //testvec = testvec / 5;
    //cout << testvec << endl;

    //BERGM test
    Col<double> initParam = { 0.0 };
    BERGM_MCMC bergm(initParam, netA);
    bergm.generateSample(1000);
    bergm.cutBurnIn(990);
    bergm.testOut();

    return 0;
}
