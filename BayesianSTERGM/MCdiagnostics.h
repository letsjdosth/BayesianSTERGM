#pragma once
#include <vector>
#include <iostream>
#include <fstream>
#include <armadillo>

using namespace std;
using namespace arma;


class MCdiagnostics {
private:
    vector<Col<double>> MCSampleVec;
    int n_sample;
    int n_dim;

    vector<Col<double>> MCDimSepVec;
    vector<double> MCDimMean; //Col<double>이 나을수도 있겠음..
    vector<double> MCDimVar;


    void MCdimSeparation();
    void MCdimStatisticCal();
    vector<double> autoCorr(int dim_idx, int maxLag);
    Col<double> smplQuantile(int dim_idx, Col<double> prob_pts);

public:
    MCdiagnostics(vector<Col<double>> MCsample);
    vector<double> get_autoCorr(int dim_idx, int maxLag);
    void print_mean(int dim_idx);
    void print_autoCorr(int dim_idx, int maxLag);
    void print_quantile(int dim_idx, Col<double> prob_pts);
    void writeToCsv_Sample(string filename);
};