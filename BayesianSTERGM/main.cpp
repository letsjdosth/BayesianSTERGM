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
//�ڵ�ȭ���ȵǾ.. �������θŹ��ٲ����(...)
//FOR ERGM:
// 1. netMCMCSampler.log_r �� model_delta(Col<double>)���� ���� ������ ���� '���� term'�� col�� element�� �߰�
// 2. ERGM_MCML.netOne_modelVal�� val�� �Ϲ� term�� col�� element�� �߰�
//FOR BERGM:
// 1. netMCMCSampler.log_r �� model_delta(Col<double>)���� ���� ������ ���� '���� term'�� col�� element�� �߰�
// 2. BERGM_MCMC.log_r�� model_delta(Col<double>)���� ���� ������ ���� '���� term'�� col�� element�� �߰�
// 3. �ʿ�� prior ���� (BERGM_MCMC.log_paramPriorPDF() ����)
//For BSTERGM:
// 1. STERGMnet1TimeMCSampler.log_r�� model_delta_formation, model_delta_dissolution ���� '���� term' ����


//diagnostics ���
//FOR ERGM/BERGM/BSTERGM:
// �׳� posterior �����ڿ� �ѱ� ��
//FOR netMCMC-DIAG:
// 1. vector<Col<double>> netMCMCSampler::getDiagStatVec() ��, netStat col�� ���ܿ�� �߰�. ���� main���� �� �Լ� ����
// 2. ����, MCdiagnostics �����ڿ� �������
// ���߿�: (BERGM/ERGM)���� ������ ���÷� ������ �����ϴ� �Լ� ����

//����: thinning ���� (�Ф�for�������� �����ѵ�.. ���� �ȶ��ϰ� ��� ��������� ã��)
// traceplot


class BSTERGM_MCMC {
private:
    vector<Network> observedSeq;
    vector<Network> observed_FormationSeq;
    vector<Network> observed_DissolutionSeq;
    vector<Col<double>> paramVec_formation;
    vector<Col<double>> paramVec_dissolution;
    
    int T_time;
    int n_Node;
    int n_paramDim;
    int n_accepted;
    int n_iterated;
    
    void dissociateObservedSeq() {
        observed_FormationSeq.push_back(observedSeq[0]);
        observed_DissolutionSeq.push_back(observedSeq[0]);

        for (int i = 1; i < observedSeq.size(); i++) {
            Mat<int> obs_now = observedSeq[i].get_netStructure();
            Mat<int> obsFormationSt = observedSeq[i-1].get_netStructure();
            Mat<int> obsDissolutionSt= observedSeq[i-1].get_netStructure();
            
            //���߿� ���������� ���ĺ� �� (formationSt�� or��. dissolution�� �𸣰���)
            for (int r = 1; r < n_Node; r++) {
                for (int c = 0; c < r; c++) {
                    if (obs_now(r, c) == 1) {
                        obsFormationSt(r, c) = 1;
                        obsFormationSt(c, r) = 1;
                    }
                    if (obs_now(r, c) == 0) {
                        obsDissolutionSt(r, c) = 0;
                        obsDissolutionSt(c, r) = 0;
                    }
                }
            }
            Network obsFormationNet = Network(obsFormationSt, 0);
            Network obsDissolutionNet = Network(obsDissolutionSt, 0);
            observed_FormationSeq.push_back(obsFormationNet);
            observed_DissolutionSeq.push_back(obsDissolutionNet);
        }
    }
    double log_paramPriorPDF(Col<double> param_formation, Col<double> param_dissolution) {
        //NOW: model parameter prior : 1
        return 0;
    }
    Col<double> proposeParam(Col<double> lastParam) {
        Mat<double> proposalCov(n_paramDim, n_paramDim, fill::eye);
        Col<double> res = mvnrnd(lastParam, proposalCov);
        return res;
    }
    STERGMnetMCSampler getSampler_ExchangeNetSeqByMCMC(Col<double> param_formation, Col<double> param_dissolution) {
        //initial : obs�� ù time������ ����
        STERGMnetMCSampler exchangeSampler = STERGMnetMCSampler(param_formation, param_dissolution, T_time, observedSeq[0], true);
        exchangeSampler.generateSample(1, 1);//���� 1 ����. �ڴ� indepMCMC�� depMCMC�� ��ġ�� ���� ���� ��
        return exchangeSampler;
    }
    double log_r(Col<double> param_lastFormation, Col<double> param_lastDissolution,
        Col<double> param_newFormation, Col<double> param_newDissolution,
        vector<Network> exchange_combinedSeq, vector<Network> exchange_FormationSeq, vector<Network> exchange_DissolutionSeq) {
        double log_r_val = 0;
        Row<double> model_delta_exchangeFormation_1(n_paramDim, fill::zeros);
        Row<double> model_delta_exchangeDissolution_2(n_paramDim, fill::zeros);
        Row<double> model_delta_obsFormation_3(n_paramDim, fill::zeros);
        Row<double> model_delta_obsDissolution_4(n_paramDim, fill::zeros);
        //cout << T_time << exchange_combinedSeq.size() << endl; //T_time,T_time+1(����: sampler���� ������ �ٿ�������)
        //cout << T_time << observedSeq.size() << observed_FormationSeq.size() << endl; //��� T_time

        // now model
        // n_Edge, k_starDist(2)
        for (int t = 1; t < T_time + 1; t++) { //exchange_~Seq�� sampler���� ����� initial y0(=obs ù��)�� �پ�����.
            model_delta_exchangeFormation_1 += {
                (double)exchange_FormationSeq[t].get_n_Edge() - exchange_combinedSeq[t - 1].get_n_Edge(),
                    (double)exchange_FormationSeq[t].get_k_starDist(2) - exchange_combinedSeq[t - 1].get_k_starDist(2)
            };
            model_delta_exchangeDissolution_2 += {
                (double)exchange_DissolutionSeq[t].get_n_Edge() - exchange_combinedSeq[t - 1].get_n_Edge(),
                    (double)exchange_DissolutionSeq[t].get_k_starDist(2) - exchange_combinedSeq[t - 1].get_k_starDist(2)
            };
        }
        for (int t = 1; t < T_time; t++) { 
            //������ sampler ������ obs ù���� �����Ƿ�, delta ù���� 0��. �ش����� ���� �����
            model_delta_obsFormation_3 += {
                (double)observed_FormationSeq[t].get_n_Edge() - observedSeq[t - 1].get_n_Edge(),
                    (double)observed_FormationSeq[t].get_k_starDist(2) - observedSeq[t - 1].get_k_starDist(2)
            };
            model_delta_obsDissolution_4 += {
                (double)observed_DissolutionSeq[t].get_n_Edge() - observedSeq[t - 1].get_n_Edge(),
                    (double)observed_DissolutionSeq[t].get_k_starDist(2) - observedSeq[t - 1].get_k_starDist(2)
            };
        }

        log_r_val += dot(param_lastFormation - param_newFormation, model_delta_exchangeFormation_1 - model_delta_obsFormation_3);
        log_r_val += dot(param_lastDissolution - param_newDissolution, model_delta_exchangeDissolution_2 - model_delta_obsDissolution_4);
        log_r_val += log_paramPriorPDF(param_newFormation, param_newDissolution);
        log_r_val -= log_paramPriorPDF(param_lastFormation, param_lastDissolution);
        return log_r_val;
    }
    void sampler() {
        Col<double> param_lastFormation = paramVec_formation.back();
        Col<double> param_lastDissolution = paramVec_dissolution.back();
        
        //proposal
        Col<double> param_newFormation = proposeParam(param_lastFormation);
        Col<double> param_newDissolution = proposeParam(param_lastDissolution);

        //exchange sample
        STERGMnetMCSampler exSeqGenerator = getSampler_ExchangeNetSeqByMCMC(param_newFormation, param_newDissolution);
        vector<Network> exchange_combinedSeq = exSeqGenerator.get_LastCombinedSeq();
        vector<Network> exchange_formationSeq = exSeqGenerator.get_LastFormationSeq();
        vector<Network> exchange_dissolutionSeq = exSeqGenerator.get_LastDissolutionSeq();
        //exSeqGenerator.printResult(0);

        double log_unif_sample = log(randu());
        double log_r_val = log_r(param_lastFormation, param_lastDissolution, param_newFormation, param_newDissolution,
            exchange_combinedSeq, exchange_formationSeq, exchange_dissolutionSeq);
        
        // cout << log_unif_sample << " " << log_r_val << endl; // for test
        if (log_unif_sample < log_r_val) {
            //accept
            paramVec_formation.push_back(param_newFormation);
            paramVec_dissolution.push_back(param_newDissolution);
            n_accepted++;
            n_iterated++;
        }
        else {
            //reject
            paramVec_formation.push_back(param_lastFormation);
            paramVec_dissolution.push_back(param_lastDissolution);
            n_iterated++;
        }

    }

public:
    BSTERGM_MCMC() {
        //�� ������
    }
    BSTERGM_MCMC(Col<double> initialParam_formation, Col<double> initialParam_dissolution, vector<Network> observedSeq) {
        this->observedSeq = observedSeq;
        paramVec_formation.push_back(initialParam_formation);
        paramVec_dissolution.push_back(initialParam_dissolution);
        T_time = observedSeq.size();
        n_Node = observedSeq[0].get_n_Node();
        n_paramDim = initialParam_dissolution.size();
        n_accepted = 0;
        n_iterated = 0;
        
        dissociateObservedSeq();
    }

    void generateSample(int num_mainMCiter) {
        for (int i = 0; i < num_mainMCiter; i++) {
            if (i % 500 == 0) {
                cout << "MCMC : " << n_iterated << "/" << num_mainMCiter << endl;
            }
            sampler();
        }
        cout << "MCMC done: " << n_iterated << " posterior samples are generated." << endl;
        cout << "accepted: " << n_accepted << " :: acc.rate: " << (double)n_accepted / n_iterated << endl;
    }

    void cutBurnIn(int n_burn_in) {
        paramVec_formation.erase(paramVec_formation.begin(), paramVec_formation.begin() + n_burn_in + 1);
        paramVec_dissolution.erase(paramVec_dissolution.begin(), paramVec_dissolution.begin() + n_burn_in + 1);
    }
    void thinning(int n_lag) {
        vector<Col<double>> paramVec_formation_afterThinning;
        vector<Col<double>> paramVec_dissolution_afterThinning;
        for (int i = 0; i < paramVec_formation.size(); i += n_lag) {
            paramVec_formation_afterThinning.push_back(paramVec_formation[i]);
            paramVec_dissolution_afterThinning.push_back(paramVec_dissolution[i]);
        }
        paramVec_formation = paramVec_formation_afterThinning;
        paramVec_dissolution = paramVec_dissolution_afterThinning;
    }

    vector<Col<double>> getPosteriorSample_formation() {
        return paramVec_formation;
    }
    vector<Col<double>> getPosteriorSample_dissolution() {
        return paramVec_dissolution;
    }


    void testOut() {
        //// dissociate test
        //for (int t = 0; t < T_time; t++) {
        //    cout << "=====================\nt=" << t << endl;
        //    cout << "formation:\n" << observed_FormationSeq[t].get_netStructure() << endl;
        //    cout << "dissolution:\n" << observed_DissolutionSeq[t].get_netStructure() << endl;
        //    cout << "observed:\n" << observedSeq[t].get_netStructure() << endl;
        //}
    }
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

    //// stergm sampler : STERGMnetMCSampler
    //Col<double> testParam1 = { 0.2, 0.1 };
    //Col<double> testParam2 = { -0.2,-0.1 };

    //STERGMnetMCSampler Tsampler = STERGMnetMCSampler(testParam1, testParam2, netA, 3);
    //Tsampler.generateSample(10, 1000);
    //Tsampler.printResult(0);
    //Tsampler.printResult(1);
    //Tsampler.printResult(2);
    //Tsampler.printResult(3);


    //BSERGM test
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

    Network netB = Network(B, false);
    Network netC = Network(C, false);
    vector<Network> netSeq = { netA, netB, netC };
    Col<double> testParam1 = { 0.2, 0.1 };
    Col<double> testParam2 = { -0.2,-0.1 };
    BSTERGM_MCMC Bstergm = BSTERGM_MCMC(testParam1, testParam2, netSeq);
    Bstergm.generateSample(300000);
    Bstergm.cutBurnIn(100000);
    Bstergm.thinning(500);
    //Bstergm.testOut();
    
    MCdiagnostics BstergmDiag1(Bstergm.getPosteriorSample_formation());
    BstergmDiag1.print_mean(0);
    Col<double> quantilePts = { 0.1, 0.25, 0.5, 0.75, 0.9 };
    BstergmDiag1.print_quantile(0, quantilePts);
    BstergmDiag1.print_autoCorr(0, 30);
    BstergmDiag1.print_mean(1);
    BstergmDiag1.print_quantile(1, quantilePts);
    BstergmDiag1.print_autoCorr(1, 30);

    MCdiagnostics BstergmDiag2(Bstergm.getPosteriorSample_dissolution());
    BstergmDiag2.print_mean(0);
    BstergmDiag2.print_quantile(0, quantilePts);
    BstergmDiag2.print_autoCorr(0, 30);
    BstergmDiag2.print_mean(1);
    BstergmDiag2.print_quantile(1, quantilePts);
    BstergmDiag2.print_autoCorr(1, 30);

    BstergmDiag1.writeToCsv_Sample("formation.csv");
    BstergmDiag2.writeToCsv_Sample("dissolution.csv");

    //=====================================

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
