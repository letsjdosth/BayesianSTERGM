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
//For BSTERGM:
// 1. STERGMnet1TimeMCSampler.log_r의 model_delta_formation, model_delta_dissolution 모델항 '차이 term' 수정


//diagnostics 방법
//FOR ERGM/BERGM/BSTERGM:
// 그냥 posterior 생성자에 넘길 것
//FOR netMCMC-DIAG:
// 1. vector<Col<double>> netMCMCSampler::getDiagStatVec() 의, netStat col에 진단요소 추가. 이후 main에서 이 함수 실행
// 2. 이후, MCdiagnostics 생성자에 집어넣자
// 나중에: (BERGM/ERGM)에서 마지막 샘플러 꺼내서 조사하는 함수 구현

//할일: thinning 구현 (ㅠㅠfor문돌리면 쉽긴한데.. 보다 똑똑하게 어떻게 방법없는지 찾기)
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
            
            //나중에 논리연산으로 고쳐볼 것 (formationSt는 or임. dissolution은 모르겠음)
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
        //initial : obs의 첫 time값에서 시작
        STERGMnetMCSampler exchangeSampler = STERGMnetMCSampler(param_formation, param_dissolution, T_time, observedSeq[0], true);
        exchangeSampler.generateSample(1, 1);//앞은 1 고정. 뒤는 indepMCMC를 depMCMC로 고치면 많이 돌릴 것
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
        //cout << T_time << exchange_combinedSeq.size() << endl; //T_time,T_time+1(이유: sampler에서 초항을 붙여나왔음)
        //cout << T_time << observedSeq.size() << observed_FormationSeq.size() << endl; //모두 T_time

        // now model
        // n_Edge, k_starDist(2)
        for (int t = 1; t < T_time + 1; t++) { //exchange_~Seq엔 sampler에서 사용한 initial y0(=obs 첫항)가 붙어있음.
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
            //어차피 sampler 초항을 obs 첫값을 썼으므로, delta 첫항은 0임. 해당항을 빼고 계산함
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
        //빈 생성자
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
