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
//자동화가안되어서.. 수동으로매번바꿔야함(...)
//FOR ERGM:
// 1. netMCMCSampler.log_r 의 model_delta(Col<double>)에서 모델항 각각에 대한 '차이 term'을 col의 element로 추가
// 2. ERGM_MCML.NRupdate1Step의, MCMCSampleVec을 만드는 for문 안 val(Col<double>)에 일반 term을 col의 element로 추가
//FOR BERGM:
// 1. netMCMCSampler.log_r 의 model_delta(Col<double>)에서 모델항 각각에 대한 '차이 term'을 col의 element로 추가
// 2. BERGM_MCMC.log_r의 model_delta(Col<double>)에서 모델항 각각에 대한 '차이 term'을 col의 element로 추가
// 3. 필요시 prior 조정 (BERGM_MCMC.log_paramPriorPDF() 구현)


class MCdiagnostics {
private:
    vector<Col<double>> MCSampleVec;
    int n_sample;
    int n_dim;

    vector<Col<double>> MCDimSepVec;
    vector<double> MCDimMean; //Col<double>이 나을수도 있겠음..
    vector<double> MCDimVar;


    void MCdimSeparation() {
        for (int i = 0; i < n_dim; i++) {
            Col<double> nth_dim_for_each_sample(n_sample);
            for (int j = 0; j < n_sample; j++) {
                nth_dim_for_each_sample(j) = MCSampleVec[j](i);
            }
            MCDimSepVec.push_back(nth_dim_for_each_sample);
        }
    }

    void MCdimStatisticCal() {
        MCDimMean.clear();
        MCDimVar.clear();
        for (int i = 0; i < n_dim; i++) {
            Col<double> dimSepSample = MCDimSepVec[i];
            MCDimMean.push_back(mean(dimSepSample));
            MCDimVar.push_back(var(dimSepSample));
        }
        
    }

    vector<double> autoCorr(int dim_idx, int maxLag) {
        Col<double> sampleSequence = MCDimSepVec[dim_idx];
        vector<double> autoCorrVec(maxLag+1);
        autoCorrVec[0] = 1.0;
        Col<double> diffMean = sampleSequence - MCDimMean[dim_idx];
        for (int i = 1; i < maxLag + 1; i++) {
            int num_pair = diffMean.size() - i;
            double cov_term = 0;
            for (int j = 0; j < num_pair; j++) {
                cov_term += (diffMean[j] * diffMean[j + i]);
            }
            cov_term /= num_pair;
            autoCorrVec[i] = (cov_term/MCDimVar[dim_idx]);
        }
        return autoCorrVec;
    }

    Col<double> smplQuantile(int dim_idx, Col<double> prob_pts) {
        Col<double> sampleSequence = MCDimSepVec[dim_idx];
        return quantile(sampleSequence, prob_pts);
    }


public:
    MCdiagnostics(vector<Col<double>> MCsample) {
        MCSampleVec = MCsample;
        n_sample = MCsample.size();
        n_dim = MCsample[0].size();
        MCdimSeparation();
        MCdimStatisticCal();
    }

    vector<double> get_autoCorr(int dim_idx, int maxLag) {
        return autoCorr(dim_idx, maxLag);
    }

    void print_mean(int dim_idx) {
        cout << "mean(dim #" << dim_idx << ") : " << MCDimMean[dim_idx] << endl;
    }

    void print_autoCorr(int dim_idx, int maxLag) {
        vector<double> nowAutoCorr = autoCorr(dim_idx, maxLag);
        cout << "auotcorrelation: ";
        for (int i = 0; i < maxLag + 1; i++) {
            cout << nowAutoCorr[i] << "  ";
        }
        cout << endl;
    }

    void print_quantile(int dim_idx, Col<double> prob_pts){
        Col<double> quantileVec = smplQuantile(dim_idx, prob_pts);
        cout << "quantile: ";
        for (int i = 0; i < prob_pts.size(); i++) {
            cout << quantileVec[i] << "  ";
        }
        cout << endl;
    }

};

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

    //ERGM test
    //Optimizer test
    /*Col<double> initParam = { 0.0 , 0.0 };
    ERGM_MCML OptimizerA(initParam, netA);
    OptimizerA.RunOptimize();*/


    ////BERGM test
    Col<double> initParam = { 0.0 , 0.0};
    BERGM_MCMC bergm(initParam, netA);
    bergm.generateSample(3000, 3000);
    bergm.cutBurnIn(1000);
    

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
