#pragma once
#include <vector>
#include <iostream>
#include <armadillo>
#include "MCdiagnostics.h"

using namespace std;
using namespace arma;

// private

void MCdiagnostics::MCdimSeparation() {
    for (int i = 0; i < n_dim; i++) {
        Col<double> nth_dim_for_each_sample(n_sample);
        for (int j = 0; j < n_sample; j++) {
            nth_dim_for_each_sample(j) = MCSampleVec[j](i);
        }
        MCDimSepVec.push_back(nth_dim_for_each_sample);
    }
}

void MCdiagnostics::MCdimStatisticCal() {
    MCDimMean.clear();
    MCDimVar.clear();
    for (int i = 0; i < n_dim; i++) {
        Col<double> dimSepSample = MCDimSepVec[i];
        MCDimMean.push_back(mean(dimSepSample));
        MCDimVar.push_back(var(dimSepSample));
    }

}

vector<double> MCdiagnostics::autoCorr(int dim_idx, int maxLag) {
    Col<double> sampleSequence = MCDimSepVec[dim_idx];
    vector<double> autoCorrVec(maxLag + 1);
    autoCorrVec[0] = 1.0;
    Col<double> diffMean = sampleSequence - MCDimMean[dim_idx];
    for (int i = 1; i < maxLag + 1; i++) {
        int num_pair = diffMean.size() - i;
        double cov_term = 0;
        for (int j = 0; j < num_pair; j++) {
            cov_term += (diffMean[j] * diffMean[j + i]);
        }
        cov_term /= num_pair;
        autoCorrVec[i] = (cov_term / MCDimVar[dim_idx]);
    }
    return autoCorrVec;
}

Col<double> MCdiagnostics::smplQuantile(int dim_idx, Col<double> prob_pts) {
    Col<double> sampleSequence = MCDimSepVec[dim_idx];
    return quantile(sampleSequence, prob_pts);
}


// public

MCdiagnostics::MCdiagnostics(vector<Col<double>> MCsample) {
    MCSampleVec = MCsample;
    n_sample = MCsample.size();
    n_dim = MCsample[0].size();
    MCdimSeparation();
    MCdimStatisticCal();
}

vector<double> MCdiagnostics::get_autoCorr(int dim_idx, int maxLag) {
    return autoCorr(dim_idx, maxLag);
}

void MCdiagnostics::print_mean(int dim_idx) {
    cout << "mean(dim #" << dim_idx << ") : " << MCDimMean[dim_idx] << endl;
}

void MCdiagnostics::print_autoCorr(int dim_idx, int maxLag) {
    vector<double> nowAutoCorr = autoCorr(dim_idx, maxLag);
    cout << "auotcorrelation: ";
    for (int i = 0; i < maxLag + 1; i++) {
        cout << nowAutoCorr[i] << "  ";
    }
    cout << endl;
}

void MCdiagnostics::print_quantile(int dim_idx, Col<double> prob_pts) {
    Col<double> quantileVec = smplQuantile(dim_idx, prob_pts);
    cout << "quantile: ";
    for (int i = 0; i < prob_pts.size(); i++) {
        cout << quantileVec[i] << "  ";
    }
    cout << endl;
}
