#pragma once
#include <vector>
#include <iostream>
#include <armadillo>

#include "Network.h"

using namespace std;
using namespace arma;


class STERGMnet1TimeMCSampler {
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
                                            (double)newProposedDissolution.get_k_starDist(2) - lastProposedDissolution.get_k_starDist(2) };
        res = dot(formation_Param, model_delta_Formation) + dot(dissolution_Param, model_delta_Dissolution);
        return res;
    }

    void sampler() {
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
        n_accepted = 0;
        n_iterated = 0;
    }

    void generateSample(int num_iter) {
        for (int i = 0; i < num_iter; i++) {
            sampler();
        }
    }

    Network get_CombinedNetMCMCSample() {
        return combined_MCMCSampleVec.back();
    }
    Network get_FormationNetMCMCSample() {
        return formation_MCMCSampleVec.back();
    }
    Network get_DissolutionNetMCMCSample() {
        return dissolution_MCMCSampleVec.back();
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
    bool resultInitInclude;
    int T_time;
    Network initialNet;
    vector<vector<Network>> combined_SeqVec;
    vector<vector<Network>> formation_SeqVec;
    vector<vector<Network>> dissolution_SeqVec;

    Col<double> formation_Param;
    Col<double> dissolution_Param;

    void sequenceSampler(int num_each_iter_per_time) {
        vector<Network> formation_oneSeq;
        vector<Network> dissolution_oneSeq;
        vector<Network> combined_oneSeq;
        
        int i;
        if (resultInitInclude) {
            formation_oneSeq.push_back(initialNet);
            dissolution_oneSeq.push_back(initialNet);
            combined_oneSeq.push_back(initialNet);
            i = 0;
        }
        else {
            // i=0, 
            STERGMnet1TimeMCSampler firstSampler = STERGMnet1TimeMCSampler(formation_Param, dissolution_Param, initialNet);
            firstSampler.generateSample(num_each_iter_per_time);
            formation_oneSeq.push_back(firstSampler.get_FormationNetMCMCSample());
            dissolution_oneSeq.push_back(firstSampler.get_DissolutionNetMCMCSample());
            combined_oneSeq.push_back(firstSampler.get_CombinedNetMCMCSample());
            
            // next i's
            i = 1;
        }

        for (i; i < T_time; i++) {
            STERGMnet1TimeMCSampler sampler = STERGMnet1TimeMCSampler(formation_Param, dissolution_Param, combined_oneSeq.back());
            sampler.generateSample(num_each_iter_per_time);

            formation_oneSeq.push_back(sampler.get_FormationNetMCMCSample());
            dissolution_oneSeq.push_back(sampler.get_DissolutionNetMCMCSample());
            combined_oneSeq.push_back(sampler.get_CombinedNetMCMCSample());

        }
        formation_SeqVec.push_back(formation_oneSeq);
        dissolution_SeqVec.push_back(dissolution_oneSeq);
        combined_SeqVec.push_back(combined_oneSeq);
    }

public:
    STERGMnetMCSampler() {
        // 빈 생성자
    }
    STERGMnetMCSampler(Col<double> formationParam, Col<double> dissolutionParam, int T_time, Network initial, bool resultInitInclude) {
        //예: sample은 t=0,1,...T_time-1 (총 T_time개)임.
        //
        this->T_time = T_time;
        this->resultInitInclude = resultInitInclude;
        this->formation_Param = formationParam;
        this->dissolution_Param = dissolutionParam;
        initialNet = initial;
    }
    void generateSample(int n_Seq, int num_each_iter_per_time) {
        for (int j = 0; j < n_Seq; j++) {
            sequenceSampler(num_each_iter_per_time);
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