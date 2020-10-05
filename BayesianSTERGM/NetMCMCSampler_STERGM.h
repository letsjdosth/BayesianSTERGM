#pragma once
#include <vector>
#include <iostream>
#include <armadillo>

#include "Network.h"

using namespace std;
using namespace arma;


class STERGMnet1TimeSampler {
private:
    Network lastTimeNet;
    Col<double> formation_Param;
    Col<double> dissolution_Param;
    int n_Node;

    Network next_combinedSample;
    Network next_formationSample;
    Network next_dissolutionSample;

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

    double log_accProb(Network proposed, bool isFormation) {
        //NOW MODEL : n_edge + k2stardist
        Col<double> model_delta = { (double)proposed.get_n_Edge() - lastTimeNet.get_n_Edge(),
                                    (double)proposed.get_k_starDist(2) - lastTimeNet.get_k_starDist(2) }; // <-model specify
        Col<double> param;
        if (isFormation) {
            param = formation_Param;
        }
        else {
            param = dissolution_Param;
        }
        double exponent = dot(param, model_delta);
        double res = exponent - log(1 + exp(exponent));
        return res;
    }

    void sampler() {
        Network newProposedFormation = proposal_Formation();
        Network newProposedDissolution = proposal_Dissolution();

        // new code
        // formation-dissolution accept-reject
        double log_unif_sample1 = log(randu());
        double log_unif_sample2 = log(randu());
        double log_accProb_formation = log_accProb(newProposedFormation, 1);
        double log_accProb_dissolution = log_accProb(newProposedDissolution, 0);
        Network newFormation;
        Network newDissolution;
        if (log_unif_sample1 < log_accProb_formation) {
            newFormation = newProposedFormation;
        }
        else {
            newFormation = lastTimeNet;
        }
        if (log_unif_sample2 < log_accProb_dissolution) {
            newDissolution = newProposedDissolution;
        }
        else {
            newDissolution = lastTimeNet;
        }
        Mat<int> yCombinedSt = newFormation.get_netStructure() - (lastTimeNet.get_netStructure() - newDissolution.get_netStructure());
        next_combinedSample = Network(yCombinedSt, 0);
        next_formationSample = newFormation;
        next_dissolutionSample = newDissolution;
    }

public:
    STERGMnet1TimeSampler() {
        //empty constructor
    }
    STERGMnet1TimeSampler(Col<double> formationParam, Col<double> dissolutionParam, Network lastTimeNet) {
        this->formation_Param = formationParam;
        this->dissolution_Param = dissolutionParam;
        this->lastTimeNet = lastTimeNet;
        n_Node = lastTimeNet.get_n_Node();
    }

    void generateSample() {
        sampler();
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
};

class STERGMnetSeqSampler {
private:
    int T_time;
    Network initialNet;
    vector<vector<Network>> combined_SeqVec;
    vector<vector<Network>> formation_SeqVec;
    vector<vector<Network>> dissolution_SeqVec;

    Col<double> formation_Param;
    Col<double> dissolution_Param;

    void sequenceSampler() {
        vector<Network> formation_oneSeq;
        vector<Network> dissolution_oneSeq;
        vector<Network> combined_oneSeq;
        
        formation_oneSeq.push_back(initialNet);
        dissolution_oneSeq.push_back(initialNet);
        combined_oneSeq.push_back(initialNet);
        
        
        for (int i = 1; i < T_time; i++) {
            STERGMnet1TimeSampler sampler = STERGMnet1TimeSampler(formation_Param, dissolution_Param, combined_oneSeq.back());
            sampler.generateSample();

            formation_oneSeq.push_back(sampler.get_FormationNetMCMCSample());
            dissolution_oneSeq.push_back(sampler.get_DissolutionNetMCMCSample());
            combined_oneSeq.push_back(sampler.get_CombinedNetMCMCSample());

        }
        formation_SeqVec.push_back(formation_oneSeq);
        dissolution_SeqVec.push_back(dissolution_oneSeq);
        combined_SeqVec.push_back(combined_oneSeq);
    }

public:
    STERGMnetSeqSampler() {
        // ºó »ý¼ºÀÚ
    }
    STERGMnetSeqSampler(Col<double> formationParam, Col<double> dissolutionParam, int T_time, Network initialNet) {
        //¿¹: sampleÀº t=0,1,...T_time-1 (ÃÑ T_time°³)ÀÓ.
        //
        this->T_time = T_time;
        this->formation_Param = formationParam;
        this->dissolution_Param = dissolutionParam;
        this->initialNet = initialNet;
    }
    void generateSample(int n_Seq) {
        for (int j = 0; j < n_Seq; j++) {
            sequenceSampler();
        }
        // cout << "SeqVec size" << combined_SeqVec.size() << endl;
    }
    vector<vector<Network>> get_FormationSeqVec() {
        return formation_SeqVec;
    }
    vector<vector<Network>> get_DissolutionSeqVec() {
        return dissolution_SeqVec;
    }
    vector<vector<Network>> get_CombinedSeqVec() {
        return combined_SeqVec;
    }

    vector<Network> get_LastFormationSeq() {
        return formation_SeqVec.back();
    }
    vector<Network> get_LastDissolutionSeq() {
        return dissolution_SeqVec.back();
    }
    vector<Network> get_LastCombinedSeq() {
        return combined_SeqVec.back();
    }

    void printResult(int idx) {
        vector<Network> outNets = combined_SeqVec[idx];
        vector<Network> outFormationNets = formation_SeqVec[idx];
        vector<Network> outDissolutionNets = dissolution_SeqVec[idx];
        cout << "Sample Sequence #" << idx << endl;
        for (int i = 0; i < outNets.size(); i++) {
            cout << "t=" << i << endl;
            cout << "formation:\n" << outFormationNets[i].get_netStructure() << endl;
            cout << "dissolution\n" << outDissolutionNets[i].get_netStructure() << endl;
            cout << "combined\n" << outNets[i].get_netStructure() << endl;
            //cout << "n_edge:" << outNets[i].get_n_Edge() << endl;
        }
    }


};