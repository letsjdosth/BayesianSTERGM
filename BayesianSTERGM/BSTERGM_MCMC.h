#pragma once
#include <vector>
#include <iostream>
#include <armadillo>

#include "NetMCMCSampler_STERGM.h"
#include "Network.h"

using namespace std;
using namespace arma;


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
            Mat<int> obsFormationSt = observedSeq[i - 1].get_netStructure();
            Mat<int> obsDissolutionSt = observedSeq[i - 1].get_netStructure();

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
    Col<double> proposeParam(Col<double> lastParam, double varRate) {
        Mat<double> proposalCov(n_paramDim, n_paramDim, fill::eye);
        Col<double> res = mvnrnd(lastParam, proposalCov * varRate);
        return res;
    }
    STERGMnetSeqSampler getSampler_ExchangeNetSeqByMCMC(Col<double> param_formation, Col<double> param_dissolution) {
        //initial : obs의 첫 time값에서 시작
        STERGMnetSeqSampler exchangeSampler = STERGMnetSeqSampler(param_formation, param_dissolution, T_time, observedSeq[0]);
        exchangeSampler.generateSample(1);
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
        //cout << T_time << exchange_combinedSeq.size() << endl; //T_time,T_time
        //cout << T_time << observedSeq.size() << observed_FormationSeq.size() << endl; //모두 T_time

        // now model
        // n_Edge, k_starDist(2)
        for (int t = 1; t < T_time; t++) {
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
        double varRate = 0.1;
        Col<double> param_newFormation = proposeParam(param_lastFormation, varRate);
        Col<double> param_newDissolution = proposeParam(param_lastDissolution, varRate);

        //exchange sample
        STERGMnetSeqSampler exSeqGenerator = getSampler_ExchangeNetSeqByMCMC(param_newFormation, param_newDissolution);
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





class BSTERGM_MCMC_RandomLag {
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

    STERGMnet1TimeSampler_1EdgeMCMC lastExchangeNetworkSampler;

    void dissociateObservedSeq() {
        observed_FormationSeq.push_back(observedSeq[0]);
        observed_DissolutionSeq.push_back(observedSeq[0]);

        for (int i = 1; i < observedSeq.size(); i++) {
            Mat<int> obs_now = observedSeq[i].get_netStructure();
            Mat<int> obsFormationSt = observedSeq[i - 1].get_netStructure();
            Mat<int> obsDissolutionSt = observedSeq[i - 1].get_netStructure();

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
    Col<double> proposeParam(Col<double> lastParam, double varRate) {
        Mat<double> proposalCov(n_paramDim, n_paramDim, fill::eye);
        Col<double> res = mvnrnd(lastParam, proposalCov * varRate);
        return res;
    }

    double log_r(Col<double> param_lastFormation, Col<double> param_lastDissolution,
        Col<double> param_newFormation, Col<double> param_newDissolution,
        int lag_startTimePoint, Network exchange_Formation, Network exchange_Dissolution) {

        double log_r_val = 0;
        Row<double> model_delta_exchangeFormation_1(n_paramDim, fill::zeros);
        Row<double> model_delta_exchangeDissolution_2(n_paramDim, fill::zeros);
        Row<double> model_delta_obsFormation_3(n_paramDim, fill::zeros);
        Row<double> model_delta_obsDissolution_4(n_paramDim, fill::zeros);
        //cout << T_time << exchange_combinedSeq.size() << endl; //T_time,T_time
        //cout << T_time << observedSeq.size() << observed_FormationSeq.size() << endl; //모두 T_time

        // now model
        // n_Edge, k_starDist(2)
        model_delta_exchangeFormation_1 += {
            (double)exchange_Formation.get_n_Edge(), // - observedSeq[lag_startTimePoint].get_n_Edge(),
                (double)exchange_Formation.get_k_starDist(2) // - observedSeq[lag_startTimePoint].get_k_starDist(2)
        };
        model_delta_exchangeDissolution_2 += {
            (double)exchange_Dissolution.get_n_Edge(), // - observedSeq[lag_startTimePoint].get_n_Edge(),
                (double)exchange_Dissolution.get_k_starDist(2) // - observedSeq[lag_startTimePoint].get_k_starDist(2)
        };
        model_delta_obsFormation_3 += {
            (double)observed_FormationSeq[lag_startTimePoint+1].get_n_Edge(), // - observedSeq[lag_startTimePoint].get_n_Edge(),
                (double)observed_FormationSeq[lag_startTimePoint+1].get_k_starDist(2) // - observedSeq[lag_startTimePoint].get_k_starDist(2)
        };
        model_delta_obsDissolution_4 += {
            (double)observed_DissolutionSeq[lag_startTimePoint+1].get_n_Edge(), // - observedSeq[lag_startTimePoint].get_n_Edge(),
                (double)observed_DissolutionSeq[lag_startTimePoint+1].get_k_starDist(2) // - observedSeq[lag_startTimePoint].get_k_starDist(2)
        };

        log_r_val += dot(param_lastFormation - param_newFormation, model_delta_exchangeFormation_1 - model_delta_obsFormation_3);
        log_r_val += dot(param_lastDissolution - param_newDissolution, model_delta_exchangeDissolution_2 - model_delta_obsDissolution_4);
        log_r_val += log_paramPriorPDF(param_newFormation, param_newDissolution);
        log_r_val -= log_paramPriorPDF(param_lastFormation, param_lastDissolution);
        return log_r_val;
    }
    void sampler(int num_exchangeMCiter) {
        Col<double> param_lastFormation = paramVec_formation.back();
        Col<double> param_lastDissolution = paramVec_dissolution.back();

        //proposal
        double varRate = 0.01;
        Col<double> param_newFormation = proposeParam(param_lastFormation, varRate);
        Col<double> param_newDissolution = proposeParam(param_lastDissolution, varRate);

        //select timeLag
        int lag_startTimePoint = randi<int>(distr_param(0, T_time - 2));

        //exchange sample
        STERGMnet1TimeSampler_1EdgeMCMC exSeqGenerator = STERGMnet1TimeSampler_1EdgeMCMC(param_newFormation, param_newDissolution, observedSeq[lag_startTimePoint]);
        exSeqGenerator.generateSample(num_exchangeMCiter);
        lastExchangeNetworkSampler = exSeqGenerator;
        Network exchange_formation = exSeqGenerator.get_FormationNetMCMCSample();
        Network exchange_dissolution = exSeqGenerator.get_DissolutionNetMCMCSample();

        //MCMC accept - reject
        double log_unif_sample = log(randu());
        double log_r_val = log_r(param_lastFormation, param_lastDissolution, param_newFormation, param_newDissolution,
            lag_startTimePoint, exchange_formation, exchange_dissolution);
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
    BSTERGM_MCMC_RandomLag() {
        //빈 생성자
    }
    BSTERGM_MCMC_RandomLag(Col<double> initialParam_formation, Col<double> initialParam_dissolution, vector<Network> observedSeq) {
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

    void generateSample(int num_mainMCiter, int num_exchangeMCiter) {
        for (int i = 0; i < num_mainMCiter; i++) {
            if (i % 500 == 0) {
                cout << "MCMC : " << n_iterated << "/" << num_mainMCiter << endl;
            }
            sampler(num_exchangeMCiter);
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
    STERGMnet1TimeSampler_1EdgeMCMC get_lastExchangeNetworkSampler() {
        return lastExchangeNetworkSampler;
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
