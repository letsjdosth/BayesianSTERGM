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


class MCdiagnostics {
private:
    vector<Col<double>> MCSampleVec;
    int n_sample;
    int n_dim;

    vector<Col<double>> MCDimSepVec;
    vector<double> MCDimMean; //Col<double>�� �������� �ְ���..
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
