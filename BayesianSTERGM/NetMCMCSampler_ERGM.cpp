#pragma once
#include "NetMCMCSampler_ERGM.h"
#include "Network.h"
#include <vector>
#include <armadillo>


pair<int, int> NetMCMCSampler_ERGM::selectRandom2Edges(int n_Node) {
    int randNode1 = randi<int>(distr_param(0, n_Node - 1));
    int randNode2 = randi<int>(distr_param(0, n_Node - 1));
    while (randNode1 == randNode2) {
        randNode2 = randi<int>(distr_param(0, n_Node - 1));
    }
    pair<int, int> res = { randNode1, randNode2 };
    return res;

}

pair<Network, int> NetMCMCSampler_ERGM::proposeNet(Network lastNet) {
    int n_node = lastNet.get_n_Node();
    bool isDirected = lastNet.is_directed_graph();
    pair<int, int> changeEdgeIndex = selectRandom2Edges(n_node);
    Mat<int> proposalNetStructure = lastNet.get_netStructure();
    int Y_ij = proposalNetStructure(changeEdgeIndex.first, changeEdgeIndex.second); //기존값

    proposalNetStructure(changeEdgeIndex.first, changeEdgeIndex.second) = 1 - Y_ij;
    if (!isDirected) {
        proposalNetStructure(changeEdgeIndex.second, changeEdgeIndex.first) = 1 - Y_ij;
    }
    Network proposalNet = Network(proposalNetStructure, isDirected);

    pair<Network, int> res = { proposalNet, Y_ij };
    return res;
}

double NetMCMCSampler_ERGM::log_r(Network lastNet, pair<Network, int> proposedNetPair) {
    //model specify 분리할 수 있으면 좋긴할듯 (어떻게?)
    //NOW: model : n_Edge
    Col<double> model_delta = { (double)proposedNetPair.first.get_n_Edge() - lastNet.get_n_Edge(),
                                (double)proposedNetPair.first.get_undirected_k_starDist(2) - lastNet.get_undirected_k_starDist(2) }; // <-model specify
    Col<double> log_r_col = (given_param * model_delta);
    double res = log_r_col(0);
    return res;
}

void NetMCMCSampler_ERGM::sampler() {
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
NetMCMCSampler_ERGM::NetMCMCSampler_ERGM(Col<double> param, Network initialNet) {
    given_param = param.t();
    MCMCSampleVec.push_back(initialNet);
    //model도 받도록 나중에
}
NetMCMCSampler_ERGM::NetMCMCSampler_ERGM(Row<double> param, Network initialNet) {
    given_param = param;
    MCMCSampleVec.push_back(initialNet);
    //model도 받도록 나중에
}
NetMCMCSampler_ERGM::NetMCMCSampler_ERGM() {
    //비워둘것
}


void NetMCMCSampler_ERGM::generateSample(int num_iter) {
    for (int i = 0; i < num_iter; i++) {
        sampler();
    }
    // cout << "MCMC done: " << n_iterated << " networks are generated." << endl;
}

void NetMCMCSampler_ERGM::cutBurnIn(int n_burn_in) {
    MCMCSampleVec.erase(MCMCSampleVec.begin(), MCMCSampleVec.begin() + n_burn_in + 1);
}

vector<Network> NetMCMCSampler_ERGM::getMCMCSampleVec() {
    return MCMCSampleVec;
}

vector<Col<double>> NetMCMCSampler_ERGM::getDiagStatVec() {
    // output candid: edge, kstar, triangle, geoWeightedNodeDegree, geoWeightedESP, geoWeightedDSP

    vector<Col<double>> res;
    for (int i = 0; i < MCMCSampleVec.size(); i++) {
        Network net = MCMCSampleVec[i];
        Col<double> netStat = { //필요시 위 candid들 더 추가
            (double)net.get_n_Edge(),
            //(double)net.get_k_starDist(2),
            //(double)net.get_triangleDist(1),
            net.get_undirected_geoWeightedNodeDegree(0.3),
            net.get_undirected_geoWeightedESP(0.3)
        };
        res.push_back(netStat);
    }
    return res;
}

void NetMCMCSampler_ERGM::testOut() {
    int i = 0;
    while (i < MCMCSampleVec.size()) {
        Network printedNet = MCMCSampleVec[i];
        cout << "#" << i << endl;
        printedNet.undirected_printSummary();
        i++;
    }
}