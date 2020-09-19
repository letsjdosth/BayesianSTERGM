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
//자동화가안되어서.. 수동으로매번바꿔야함(...)
//FOR ERGM:
// 1. netMCMCSampler.log_r 의 model_delta(Col<double>)에서 모델항 각각에 대한 '차이 term'을 col의 element로 추가
// 2. ERGM_MCML.netOne_modelVal의 val에 일반 term을 col의 element로 추가
//FOR BERGM:
// 1. netMCMCSampler.log_r 의 model_delta(Col<double>)에서 모델항 각각에 대한 '차이 term'을 col의 element로 추가
// 2. BERGM_MCMC.log_r의 model_delta(Col<double>)에서 모델항 각각에 대한 '차이 term'을 col의 element로 추가
// 3. 필요시 prior 조정 (BERGM_MCMC.log_paramPriorPDF() 구현)

//diagnostics 방법
//FOR netMCMC-DIAG:
// 1. vector<Col<double>> netMCMCSampler::getDiagStatVec() 의, netStat col에 진단요소 추가. 이후 main에서 이 함수 실행
// 2. 이후, MCdiagnostics 생성자에 집어넣자
// 나중에: (BERGM/ERGM)에서 마지막 샘플러 꺼내서 조사하는 함수 구현

//할일: thinning 구현 (ㅠㅠfor문돌리면 쉽긴한데.. 보다 똑똑하게 어떻게 방법없는지 찾기)
// traceplot



class STERGMnet1TimeMCSampler{
private:
    Network lastTimeNet;
    vector<Network> formation_MCMCSampleVec;
    vector<Network> dissolution_MCMCSampleVec;
    vector<Network> combined_MCMCSampleVec;
    Col<double> formation_Param;
    Col<double> dissolution_Param;
    int n_Node;
    int n_accepted;
    int n_iterated;

    
    Mat<int> genSymmetricMat(int dim) {
        Mat<int> res(dim, dim, fill::zeros);
        Col<double> randmem((dim - 1) * dim / 2, fill::randu);
        randmem = round(randmem);
        int i = 0;
        for (int r = 1; r < dim; r++) {
            for (int c = 0; c < r; c++) {
                double val = randmem(i);
                res(r, c) = val;
                res(c, r) = val;
                i++;
            }
        }
        return res;
    }

    Network proposal_Formation() {
        //하나씩해야할지도모름..
        //undirected
        Mat<int> lastSt = lastTimeNet.get_netStructure();
        Mat<int> mixingProposalSt = genSymmetricMat(n_Node);
        Mat<int> res = conv_to<Mat<int>>::from(lastSt || mixingProposalSt);
        // cout << (lastSt + mixingProposalSt) << endl;
        Network resNet = Network(res, 0);
        return resNet;
    }
    Network proposal_Dissolution() {
        //undirected
        Mat<int> lastSt = lastTimeNet.get_netStructure();
        Mat<int> mixingProposalSt = genSymmetricMat(n_Node);
        Mat<int> res = conv_to<Mat<int>>::from(lastSt && mixingProposalSt);
        // cout << (lastSt +mixingProposalSt) << endl;
        Network resNet = Network(res, 0);
        return resNet;
    }

    double log_r(Network newProposedFormation, Network newProposedDissolution,
                Network lastProposedFormation, Network lastProposedDissolution)
    {
        //NOW: model : n_Edge + get_k_starDist(2)
        double res;

        //Col<double> model_delta_NewFormation = { (double)newProposedFormation.get_n_Edge() - lastTimeNet.get_n_Edge(),
        //                            (double)newProposedFormation.get_k_starDist(2) - lastTimeNet.get_k_starDist(2) }; // <-model specify
        //Col<double> model_delta_NewDissolution = { (double)newProposedDissolution.get_n_Edge() - lastTimeNet.get_n_Edge(),
        //                            (double)newProposedDissolution.get_k_starDist(2) - lastTimeNet.get_k_starDist(2) }; // <-model specify
        //Col<double> model_delta_LastFormation = { (double)lastProposedFormation.get_n_Edge() - lastTimeNet.get_n_Edge(),
        //                            (double)lastProposedFormation.get_k_starDist(2) - lastTimeNet.get_k_starDist(2) }; // <-model specify
        //Col<double> model_delta_LastDissolution = { (double)lastProposedDissolution.get_n_Edge() - lastTimeNet.get_n_Edge(),
        //                            (double)lastProposedDissolution.get_k_starDist(2) - lastTimeNet.get_k_starDist(2) }; // <-model specify
        //res = dot(formation_Param, model_delta_NewFormation) + dot(dissolution_Param, model_delta_NewDissolution)
        //    - dot(formation_Param, model_delta_LastFormation) - dot(dissolution_Param, model_delta_LastDissolution);

        Col<double> model_delta_Formation = { (double)newProposedFormation.get_n_Edge() - lastProposedFormation.get_n_Edge(),
                                            (double)newProposedFormation.get_k_starDist(2) - lastProposedFormation.get_k_starDist(2) };
        Col<double> model_delta_Dissolution = { (double)newProposedDissolution.get_n_Edge() - lastProposedDissolution.get_n_Edge(),
                                            (double)newProposedDissolution .get_k_starDist(2) - lastProposedDissolution.get_k_starDist(2)};
        res = dot(formation_Param, model_delta_Formation) + dot(dissolution_Param, model_delta_Dissolution);
        return res;
    }

    void sampler(){
        Network lastFormation = formation_MCMCSampleVec.back();
        Network lastDissolution = dissolution_MCMCSampleVec.back();
        Network lastProposedCombinedNet = combined_MCMCSampleVec.back();
        
        Network newProposedFormation = proposal_Formation();
        Network newProposedDissolution = proposal_Dissolution();
        Mat<int> yCombinedSt = newProposedFormation.get_netStructure() - (lastTimeNet.get_netStructure() - newProposedDissolution.get_netStructure());
        Network newProposedCombinedNet = Network(yCombinedSt, 0);

        // accept-reject
        double log_unif_sample = log(randu());
        double log_r_val = log_r(newProposedFormation, newProposedDissolution, lastFormation, lastDissolution);
        if (log_unif_sample < log_r_val) {
            //accept
            formation_MCMCSampleVec.push_back(newProposedFormation);
            dissolution_MCMCSampleVec.push_back(newProposedDissolution);
            combined_MCMCSampleVec.push_back(newProposedCombinedNet);
            n_accepted++;
            n_iterated++;
        }
        else {
            //reject
            formation_MCMCSampleVec.push_back(lastFormation);
            dissolution_MCMCSampleVec.push_back(lastDissolution);
            combined_MCMCSampleVec.push_back(lastProposedCombinedNet);
            n_iterated++;
        }
    }
    
public:
    STERGMnet1TimeMCSampler() {
        //empty constructor
    }
    STERGMnet1TimeMCSampler(Col<double> formationParam, Col<double> dissolutionParam, Network lastTimeNet) {
        this->formation_Param = formationParam;
        this->dissolution_Param = dissolutionParam;
        this->lastTimeNet = lastTimeNet;
        formation_MCMCSampleVec.push_back(lastTimeNet); //initial net을 따로 받아도 될듯
        dissolution_MCMCSampleVec.push_back(lastTimeNet); //initial net을 따로 받아도 될듯
        combined_MCMCSampleVec.push_back(lastTimeNet); //initial net을 따로 받아도 될듯
        n_Node = lastTimeNet.get_n_Node();
    }
    
    void generateSample(int num_iter) {
        for (int i = 0; i < num_iter; i++) {
            sampler();
        }
    }

    Network getTemporalNetMCMCSample() {
        return combined_MCMCSampleVec.back();
    }

    void cutBurnIn(int n_burn_in) {
        formation_MCMCSampleVec.erase(formation_MCMCSampleVec.begin(), formation_MCMCSampleVec.begin() + n_burn_in + 1);
        dissolution_MCMCSampleVec.erase(dissolution_MCMCSampleVec.begin(), dissolution_MCMCSampleVec.begin() + n_burn_in + 1);
        combined_MCMCSampleVec.erase(combined_MCMCSampleVec.begin(), combined_MCMCSampleVec.begin() + n_burn_in + 1);
    }

    void testOut() {
        int i = 0;
        while (i < combined_MCMCSampleVec.size()) {
            cout << "#" << i << endl;
            combined_MCMCSampleVec[i].printSummary();
            i++;
        }
        cout << "n_accepted : " << n_accepted << ", n_iterated: " << n_iterated << endl;
        cout << "acc rate : " << (double)n_accepted / n_iterated << endl;
    }
};

class STERGMnetMCSampler {
private:
    int T_time;
    Network initialNet;
    vector<vector<Network>> SeqVec;
    Col<double> formation_Param;
    Col<double> dissolution_Param;

    void sequenceSampler(int num_each_iter_per_time) {
        vector<Network> oneSeq = { initialNet };
        for (int i = 0; i < T_time; i++) {
            STERGMnet1TimeMCSampler sampler = STERGMnet1TimeMCSampler(formation_Param, dissolution_Param, oneSeq.back());
            sampler.generateSample(num_each_iter_per_time);
            oneSeq.push_back(sampler.getTemporalNetMCMCSample());
        }
        SeqVec.push_back(oneSeq);
    }

public:
    STERGMnetMCSampler() {
        // 빈 생성자
    }
    STERGMnetMCSampler(Col<double> formationParam, Col<double> dissolutionParam, Network initial, int T_time) {
        this->T_time = T_time;
        this->formation_Param = formationParam;
        this->dissolution_Param = dissolutionParam;
        initialNet = initial;
    }
    void generateSample(int n_Seq, int num_each_iter_per_time) {
        for (int j = 0; j < n_Seq; j++) {
            sequenceSampler(num_each_iter_per_time);
        }
        cout << "SeqVec size" << SeqVec.size() << endl;
    }
    void printResult(int idx) {
        vector<Network> outNets = SeqVec[idx];
        cout << "Sample Sequence #" << idx << endl;
        for (int i = 0; i < outNets.size(); i++) {
            cout << "t=" << i << endl;
            //cout << outNets[i].get_netStructure() << endl;
            cout << "n_edge:" << outNets[i].get_n_Edge() << endl;
        }
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
