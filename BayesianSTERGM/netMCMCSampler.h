#pragma once
#include <vector>
#include <armadillo>
#include "Network.h"

using namespace std;
using namespace arma;


class netMCMCSampler {
//undirected
private:
    Row<double> given_param;
    vector<Network> MCMCSampleVec;
    int n_iter;

    int n_accepted;
    int n_iterated;

    pair<int, int> selectRandom2Edges(int n_Node);
    pair<Network, int> proposeNet(Network lastNet);
    double log_r(Network lastNet, pair<Network, int> proposedNetPair);
    void sampler();

public:
    netMCMCSampler(Col<double> param, Network initialNet);
    netMCMCSampler(Row<double> param, Network initialNet);
    netMCMCSampler();
    void generateSample(int num_iter);
    void cutBurnIn(int n_burn_in);
    vector<Network> getMCMCSampleVec();
    void testOut();
};
