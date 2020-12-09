#pragma once
#include <vector>
#include <iostream>
#include <armadillo>

#include "Network.h"

using namespace std;
using namespace arma;



class STERGMnet1TimeSampler_1EdgeMCMC {
private:
    Network lastTimeNet;
    Col<double> formation_Param;
    Col<double> dissolution_Param;
    int n_Node;
    bool isDirected;

    vector<Network> MCproposedNetVec;

    Network next_combinedSample;
    Network next_formationSample;
    Network next_dissolutionSample;

    pair<int, int> selectRandom2Edges(int n_Node) {
        int randNode1 = randi<int>(distr_param(0, n_Node - 1));
        int randNode2 = randi<int>(distr_param(0, n_Node - 1));
        while (randNode1 == randNode2) {
            randNode2 = randi<int>(distr_param(0, n_Node - 1));
        }
        pair<int, int> res = { randNode1, randNode2 };
        return res;

    }

    pair<Network, int> proposeNet(Network& lastNet) {
        pair<int, int> changeEdgeIndex = selectRandom2Edges(n_Node);
        Mat<int> proposalNetStructure = lastNet.get_netStructure();
        int Y_ij = proposalNetStructure(changeEdgeIndex.first, changeEdgeIndex.second); //기존값

        proposalNetStructure(changeEdgeIndex.first, changeEdgeIndex.second) = 1 - Y_ij;
        
        if (!isDirected) {
            proposalNetStructure(changeEdgeIndex.second, changeEdgeIndex.first) = 1 - Y_ij;
        }
        Network proposalNet = Network(proposalNetStructure, isDirected);

        pair<Network, int> res = { proposalNet, Y_ij }; // Y_ij==1 : dissolution, Y_ij==0 : formation
        return res;
    }

    double log_r(Network& last, Network& proposed, bool isDissolution) {
        //NOW MODEL
        Col<double> model_delta = { (double)proposed.get_n_Edge() - last.get_n_Edge(),
                                     (double)proposed.get_undirected_geoWeightedESP(0.5) - last.get_undirected_geoWeightedESP (0.5) 
        }; // <-model specify
        Col<double> param;
        if (isDissolution) {
            param = dissolution_Param;
        }
        else {
            param = formation_Param;
        }
        double res = dot(param, model_delta);
        return res;
    }

    void sampler() {
        Network lastIterNet = MCproposedNetVec.back();
        pair<Network, int> proposedNet = proposeNet(lastIterNet);

        double log_r_val = log_r(lastIterNet, proposedNet.first, proposedNet.second);
        double log_unif_sample = log(randu());

        if (log_unif_sample < log_r_val) {
            MCproposedNetVec.push_back(proposedNet.first);
        }
        else {
            MCproposedNetVec.push_back(lastIterNet);
        }
    }

    void compute_nextNet() {
        Mat<int> combinedSt = MCproposedNetVec.back().get_netStructure();
        Mat<int> formationSt = lastTimeNet.get_netStructure();
        Mat<int> dissolutionSt = lastTimeNet.get_netStructure();

        //나중에 논리연산으로 고쳐볼 것 (formationSt는 or임. dissolution은 모르겠음)
        for (int r = 0; r < n_Node; r++) {
            for (int c = 0; c < n_Node; c++) {
                if (combinedSt(r, c) == 1) {
                    formationSt(r, c) = 1;
                }
                if (combinedSt(r, c) == 0) {
                    dissolutionSt(r, c) = 0;
                }
            }
        }
        Network formationNet = Network(formationSt, isDirected);
        Network dissolutionNet = Network(dissolutionSt, isDirected);
        next_formationSample = formationNet;
        next_dissolutionSample = dissolutionNet;
        next_combinedSample = MCproposedNetVec.back();
    }

public:
    STERGMnet1TimeSampler_1EdgeMCMC() {
        //empty constructor
    }
    STERGMnet1TimeSampler_1EdgeMCMC(Col<double> formationParam, Col<double> dissolutionParam, Network lastTimeNet) {
        this->formation_Param = formationParam;
        this->dissolution_Param = dissolutionParam;
        this->lastTimeNet = lastTimeNet;
        this->MCproposedNetVec.push_back(lastTimeNet);
        this->n_Node = lastTimeNet.get_n_Node();
        this->isDirected = lastTimeNet.is_directed_graph();
    }

    void generateSample(int num_iter) {
        for (int i = 0; i < num_iter; i++) {
            sampler();
        }
        // cout << "MCMC done: " << n_iterated << " networks are generated." << endl;
        compute_nextNet();
    }

    void cutBurnIn(int n_burn_in) {
        MCproposedNetVec.erase(MCproposedNetVec.begin(), MCproposedNetVec.begin() + n_burn_in + 1);
    }
    Network get_CombinedNetMCMCSample() {
        return next_combinedSample;
    }
    Network get_FormationNetMCMCSample() {
        return next_formationSample;
    }
    Network get_DissolutionNetMCMCSample() {
        return next_dissolutionSample;
    }
    vector<Network> get_MCMCSampleVec() {
        return MCproposedNetVec;
    }
};



//class STERGMnet1TimeSampler_indepMC {
//private:
//    Network lastTimeNet;
//    Col<double> formation_Param;
//    Col<double> dissolution_Param;
//    int n_Node;
//
//    Network next_combinedSample;
//    Network next_formationSample;
//    Network next_dissolutionSample;
//
//    Mat<int> genSymmetricMat(int dim) {
//        Mat<int> res(dim, dim, fill::zeros);
//        Col<double> randmem((dim - 1) * dim / 2, fill::randu);
//        randmem = round(randmem);
//        int i = 0;
//        for (int r = 1; r < dim; r++) {
//            for (int c = 0; c < r; c++) {
//                double val = randmem(i);
//                res(r, c) = val;
//                res(c, r) = val;
//                i++;
//            }
//        }
//        return res;
//    }
//
//    Network proposal_Formation() {
//        //undirected
//        Mat<int> lastSt = lastTimeNet.get_netStructure();
//        Mat<int> mixingProposalSt = genSymmetricMat(n_Node);
//        Mat<int> res = conv_to<Mat<int>>::from(lastSt || mixingProposalSt);
//        // cout << (lastSt + mixingProposalSt) << endl;
//        Network resNet = Network(res, 0);
//        return resNet;
//    }
//    Network proposal_Dissolution() {
//        //undirected
//        Mat<int> lastSt = lastTimeNet.get_netStructure();
//        Mat<int> mixingProposalSt = genSymmetricMat(n_Node);
//        Mat<int> res = conv_to<Mat<int>>::from(lastSt && mixingProposalSt);
//        // cout << (lastSt +mixingProposalSt) << endl;
//        Network resNet = Network(res, 0);
//        return resNet;
//    }
//
//    double log_accProb(Network proposed, bool isDissolution) {
//        //NOW MODEL : n_edge + k2stardist
//        Col<double> model_delta = { (double)proposed.get_n_Edge() - lastTimeNet.get_n_Edge(),
//                                    (double)proposed.get_undirected_k_starDist(2) - lastTimeNet.get_undirected_k_starDist(2) }; // <-model specify
//        Col<double> param;
//        if (isDissolution) {
//            param = dissolution_Param;
//        }
//        else {
//            param = formation_Param;
//        }
//        double exponent = dot(param, model_delta);
//        double res = exponent - log(1 + exp(exponent));
//        return res;
//    }
//
//    void sampler() {
//        Network newProposedFormation = proposal_Formation();
//        Network newProposedDissolution = proposal_Dissolution();
//
//        // new code
//        // formation-dissolution accept-reject
//        double log_unif_sample1 = log(randu());
//        double log_unif_sample2 = log(randu());
//        double log_accProb_formation = log_accProb(newProposedFormation, 0);
//        double log_accProb_dissolution = log_accProb(newProposedDissolution, 1);
//        Network newFormation;
//        Network newDissolution;
//        if (log_unif_sample1 < log_accProb_formation) {
//            newFormation = newProposedFormation;
//        }
//        else {
//            newFormation = lastTimeNet;
//        }
//        if (log_unif_sample2 < log_accProb_dissolution) {
//            newDissolution = newProposedDissolution;
//        }
//        else {
//            newDissolution = lastTimeNet;
//        }
//        Mat<int> yCombinedSt = newFormation.get_netStructure() - (lastTimeNet.get_netStructure() - newDissolution.get_netStructure());
//        next_combinedSample = Network(yCombinedSt, 0);
//        next_formationSample = newFormation;
//        next_dissolutionSample = newDissolution;
//    }
//
//public:
//    STERGMnet1TimeSampler_indepMC() {
//        //empty constructor
//    }
//    STERGMnet1TimeSampler_indepMC(Col<double> formationParam, Col<double> dissolutionParam, Network lastTimeNet) {
//        this->formation_Param = formationParam;
//        this->dissolution_Param = dissolutionParam;
//        this->lastTimeNet = lastTimeNet;
//        n_Node = lastTimeNet.get_n_Node();
//    }
//
//    void generateSample() {
//        sampler();
//    }
//
//    Network get_CombinedNetMCMCSample() {
//        return next_combinedSample;
//    }
//    Network get_FormationNetMCMCSample() {
//        return next_formationSample;
//    }
//    Network get_DissolutionNetMCMCSample() {
//        return next_dissolutionSample;
//    }
//};



//class STERGMnetSeqSampler {
//private:
//    int T_time;
//    Network initialNet;
//    vector<vector<Network>> combined_SeqVec;
//    vector<vector<Network>> formation_SeqVec;
//    vector<vector<Network>> dissolution_SeqVec;
//
//    Col<double> formation_Param;
//    Col<double> dissolution_Param;
//
//    void sequenceSampler() {
//        vector<Network> formation_oneSeq;
//        vector<Network> dissolution_oneSeq;
//        vector<Network> combined_oneSeq;
//        
//        formation_oneSeq.push_back(initialNet);
//        dissolution_oneSeq.push_back(initialNet);
//        combined_oneSeq.push_back(initialNet);
//        
//        
//        for (int i = 1; i < T_time; i++) {
//            //sampler 교체시 여기 2줄 변경
//            STERGMnet1TimeSampler_indepMC sampler = STERGMnet1TimeSampler_indepMC(formation_Param, dissolution_Param, combined_oneSeq.back());
//            sampler.generateSample();
//
//            formation_oneSeq.push_back(sampler.get_FormationNetMCMCSample());
//            dissolution_oneSeq.push_back(sampler.get_DissolutionNetMCMCSample());
//            combined_oneSeq.push_back(sampler.get_CombinedNetMCMCSample());
//
//        }
//        formation_SeqVec.push_back(formation_oneSeq);
//        dissolution_SeqVec.push_back(dissolution_oneSeq);
//        combined_SeqVec.push_back(combined_oneSeq);
//    }
//
//public:
//    STERGMnetSeqSampler() {
//        // 빈 생성자
//    }
//    STERGMnetSeqSampler(Col<double> formationParam, Col<double> dissolutionParam, int T_time, Network initialNet) {
//        //예: 한 sample은 t=0,1,...T_time-1 (총 T_time개)의 Network로 이루어짐.
//        //
//        this->T_time = T_time;
//        this->formation_Param = formationParam;
//        this->dissolution_Param = dissolutionParam;
//        this->initialNet = initialNet;
//    }
//    void generateSample(int n_Seq) {
//        for (int j = 0; j < n_Seq; j++) {
//            sequenceSampler();
//        }
//        // cout << "SeqVec size" << combined_SeqVec.size() << endl;
//    }
//    vector<vector<Network>> get_FormationSeqVec() {
//        return formation_SeqVec;
//    }
//    vector<vector<Network>> get_DissolutionSeqVec() {
//        return dissolution_SeqVec;
//    }
//    vector<vector<Network>> get_CombinedSeqVec() {
//        return combined_SeqVec;
//    }
//
//    vector<Network> get_LastFormationSeq() {
//        return formation_SeqVec.back();
//    }
//    vector<Network> get_LastDissolutionSeq() {
//        return dissolution_SeqVec.back();
//    }
//    vector<Network> get_LastCombinedSeq() {
//        return combined_SeqVec.back();
//    }
//
//    void printResult(int sampleSeqIdx) {
//        vector<Network> outNets = combined_SeqVec[sampleSeqIdx];
//        vector<Network> outFormationNets = formation_SeqVec[sampleSeqIdx];
//        vector<Network> outDissolutionNets = dissolution_SeqVec[sampleSeqIdx];
//        cout << "Sample Sequence #" << sampleSeqIdx << endl;
//        for (int i = 0; i < outNets.size(); i++) {
//            cout << "t=" << i << endl;
//            cout << "formation:\n" << outFormationNets[i].get_netStructure() << endl;
//            cout << "dissolution\n" << outDissolutionNets[i].get_netStructure() << endl;
//            cout << "combined\n" << outNets[i].get_netStructure() << endl;
//            //cout << "n_edge:" << outNets[i].get_n_Edge() << endl;
//        }
//    }
//
//
//};