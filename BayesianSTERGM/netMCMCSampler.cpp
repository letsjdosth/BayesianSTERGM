#pragma once
#include "netMCMCSampler.h"
#include "Network.h"
#include <vector>
#include <armadillo>


pair<int, int> netMCMCSampler::selectRandom2Edges(int n_Node) {
    int randNode1 = randi<int>(distr_param(0, n_Node - 1));
    int randNode2 = randi<int>(distr_param(0, n_Node - 1));
    while (randNode1 == randNode2) {
        randNode2 = randi<int>(distr_param(0, n_Node - 1));
    }
    pair<int, int> res = { randNode1, randNode2 };
    return res;

}

pair<Network, int> netMCMCSampler::proposeNet(Network lastNet) {
    int n_node = lastNet.get_n_Node();
    pair<int, int> changeEdgeIndex = selectRandom2Edges(n_node);
    Mat<int> proposalNetStructure = lastNet.get_netStructure();
    int Y_ij = proposalNetStructure(changeEdgeIndex.first, changeEdgeIndex.second);

    proposalNetStructure(changeEdgeIndex.first, changeEdgeIndex.second) = 1 - Y_ij;
    proposalNetStructure(changeEdgeIndex.second, changeEdgeIndex.first) = 1 - Y_ij;
    Network proposalNet = Network(proposalNetStructure, false);

    pair<Network, int> res = { proposalNet, Y_ij };
    return res;
}

double netMCMCSampler::log_r(Network lastNet, pair<Network, int> proposedNetPair) {
    //model specify �и��� �� ������ �����ҵ� (���?)
    //NOW: model : n_Edge
    Col<double> model_delta = { (double)proposedNetPair.first.get_n_Edge() - lastNet.get_n_Edge(),
                                (double)proposedNetPair.first.get_n_triangle() - lastNet.get_n_triangle()}; // <-model specify
    Col<double> log_r_col = (given_param * model_delta);
    double res = log_r_col(0);
    if (proposedNetPair.second == 1) res *= -1;
    return res;
}

void netMCMCSampler::sampler() {
    Network lastNet = MCMCSampleVec.back();
    pair<Network, int> proposedNetPair = proposeNet(lastNet);
    double log_unif_sample = log(randu());
    double log_r_val = log_r(lastNet, proposedNetPair);
    if (log_unif_sample < log_r_val) {
        //accept
        MCMCSampleVec.push_back(proposedNetPair.first);
        n_accepted++;
        n_iterated++;
    }
    else {
        //reject
        MCMCSampleVec.push_back(lastNet);
        n_iterated++;
    }
}



//public
netMCMCSampler::netMCMCSampler(Col<double> param, Network initialNet) {
    given_param = param.t();
    MCMCSampleVec.push_back(initialNet);
    //model�� �޵��� ���߿�
}
netMCMCSampler::netMCMCSampler(Row<double> param, Network initialNet) {
    given_param = param;
    MCMCSampleVec.push_back(initialNet);
    //model�� �޵��� ���߿�
}
netMCMCSampler::netMCMCSampler() {
    //����Ѱ�
}


void netMCMCSampler::generateSample(int num_iter) {
    for (int i = 0; i < num_iter; i++) {
        sampler();
    }
    // cout << "MCMC done: " << n_iterated << " networks are generated." << endl;
}

void netMCMCSampler::cutBurnIn(int n_burn_in) {
    MCMCSampleVec.erase(MCMCSampleVec.begin(), MCMCSampleVec.begin() + n_burn_in + 1);
}

vector<Network> netMCMCSampler::getMCMCSampleVec() {
    return MCMCSampleVec;
}

vector<Col<double>> netMCMCSampler::getDiagStatVec() {
    // output candid: edge, kstar, triangle, geoWeightedNodeDegree, geoWeightedESP, geoWeightedDSP
    
    vector<Col<double>> res;
    for (int i = 0; i < MCMCSampleVec.size(); i++) {
        Network net = MCMCSampleVec[i];
        Col<double> netStat = { //�ʿ�� �� candid�� �� �߰�
            (double)net.get_n_Edge(),
            (double)net.get_n_triangle()
        };
        res.push_back(netStat);
    }
    return res;
}

void netMCMCSampler::testOut() {
    int i = 0;
    while (i < MCMCSampleVec.size()) {
        Network printedNet = MCMCSampleVec[i];
        cout << "#" << i << endl;
        printedNet.printSummary();
        i++;
    }

}