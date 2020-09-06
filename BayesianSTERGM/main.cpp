#pragma once
#include <vector>
#include <iostream>
#include <armadillo>

#include "Network.h"
#include "netMCMCSampler.h"
#include "ERGM_MCML.h"


using namespace std;
using namespace arma;


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


    //MCMCsampler test
    Col<double> testParam = { 0.1 };
    netMCMCSampler sampler(testParam, netA);
    sampler.generateSample(10);
    sampler.testOut();
    sampler.cutBurnIn(8);
    cout << "after burnin" << endl;
    sampler.testOut();

    //Optimizer test
    Col<double> initParam = { 0.0 };
    ERGM_MCML OptimizerA(initParam, netA);
    OptimizerA.RunOptimize();

    //Col<double> testvec = { 1,2,3,4,5 };
    //testvec = testvec / 5;
    //cout << testvec << endl;



    return 0;
}
