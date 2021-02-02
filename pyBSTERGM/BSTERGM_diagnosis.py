import csv

import numpy as np
import matplotlib.pyplot as plt

from BSTERGM import BSTERGM

class BSTERGM_posterior_work:
    def __init__(self):
        self.initial_formation_param = np.array(0)
        self.initial_dissolution_param = np.array(0)
        self.MC_formation_samples = []
        self.MC_dissolution_samples = []

    def read_from_csv(self, file_name, formation_param_dim, dissolution_param_dim):
        '''caution:
        Returned BSTERGM_Read instance has only propoerties as following:
            self.initial_formation_param(if the sample is cut burn-in, this value may not the real initial value.)
            self.initial_dissolution_param(same as above)
            self.MC_formation_samples
            self.MC_dissolution_samples
        thus, DO NOT use this instance to rerun MCMC or other works,
        except getting plots or summary statistics, diagonostics, GOF, etc.
        '''
        with open("pyBSTERGM/" + file_name + '.csv', 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for csv_row in reader:
                csv_row = [float(elem) for elem in csv_row]
                self.MC_formation_samples.append(np.array(csv_row[0:formation_param_dim]))
                self.MC_dissolution_samples.append(np.array(csv_row[formation_param_dim : formation_param_dim + dissolution_param_dim]))

        self.initial_formation_param = self.MC_formation_samples[0]
        self.initial_dissolution_param = self.MC_dissolution_samples[0]

    def read_from_BSTERGM_instance(self):
        pass
    
    def MC_sample_trace(self):
        formation_trace = []
        dissolution_trace = []
        
        for _ in range(len(self.initial_formation_param)):
            formation_trace.append([])
        for _ in range(len(self.initial_dissolution_param)):
            dissolution_trace.append([])
        for formation_sample in self.MC_formation_samples:
            for i, param_val in enumerate(formation_sample):
                formation_trace[i].append(param_val)
        for dissolution_sample in self.MC_dissolution_samples:
            for i, param_val in enumerate(dissolution_sample):
                dissolution_trace[i].append(param_val)
        return formation_trace, dissolution_trace


    def show_traceplot(self, show=True):
        formation_trace, dissolution_trace = self.MC_sample_trace()
        grid_column = 1
        # grid_row = int(len(netStat)/2+0.51)
        grid_row = len(self.initial_formation_param) + len(self.initial_dissolution_param)
        plt.figure(figsize=(5*grid_column, 3*grid_row))
        for i, paramSeq in enumerate(formation_trace):
            plt.subplot(grid_row, grid_column, i+1)
            plt.plot(range(len(paramSeq)), paramSeq)
        for i, paramSeq in enumerate(dissolution_trace):
            plt.subplot(grid_row, grid_column, len(self.initial_formation_param)+i+1)
            plt.plot(range(len(paramSeq)), paramSeq)

        if show:
            plt.show()

    def show_histogram(self, bins=100 ,show=True):
        formation_trace, dissolution_trace = self.MC_sample_trace()
        grid_column = 1
        # grid_row = int(len(netStat)/2+0.51)
        grid_row = len(self.initial_formation_param) + len(self.initial_dissolution_param)
        plt.figure(figsize=(5*grid_column, 3*grid_row))
        for i, paramSeq in enumerate(formation_trace):
            plt.subplot(grid_row, grid_column, i+1)
            plt.hist(paramSeq, bins=bins, density=True)
        for i, paramSeq in enumerate(dissolution_trace):
            plt.subplot(grid_row, grid_column, len(self.initial_formation_param)+i+1)
            plt.hist(paramSeq, bins=bins, density=True)

        if show:
            plt.show()

    def get_autocorr(self, trace, maxLag):
        acf = []
        trace_mean = np.mean(trace)
        trace = [elem - trace_mean  for elem in trace]
        n_var = sum([elem**2 for elem in trace])
        for k in range(maxLag+1):
            N = len(trace)-k
            n_cov_term = 0
            for i in range(N):
                n_cov_term += trace[i]*trace[i+k]
            acf.append(n_cov_term / n_var)
        return acf

    def show_acfplot(self, maxLag=50, show=True):
        formation_trace, dissolution_trace = self.MC_sample_trace()
        grid_column = 1
        # grid_row = int(len(netStat)/2+0.51)
        grid_row = len(self.initial_formation_param) + len(self.initial_dissolution_param)
        grid = [i for i in range(maxLag+1)]

        plt.figure(figsize=(5*grid_column, 3*grid_row))
        for i, paramSeq in enumerate(formation_trace):
            plt.subplot(grid_row, grid_column, i+1)
            nowseq_acf = self.get_autocorr(paramSeq, maxLag)
            plt.ylim([-1,1])
            plt.bar(grid, nowseq_acf, width=0.3)
            plt.axhline(0, color="black", linewidth=0.8)

        for i, paramSeq in enumerate(dissolution_trace):
            plt.subplot(grid_row, grid_column, len(self.initial_formation_param)+i+1)
            nowseq_acf = self.get_autocorr(paramSeq, maxLag)
            plt.ylim([-1,1])
            plt.bar(grid, nowseq_acf, width=0.3)
            plt.axhline(0, color="black", linewidth=0.8)

        if show:
            plt.show()


class BSTERGM_latest_exchangeSampler_work:
    def __init__(self):
        self.netstat_vec = []
    
    def read_from_csv(self, file_name):
        with open("pyBSTERGM/" + file_name + '.csv', 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for csv_row in reader:
                csv_row = [float(elem) for elem in csv_row]
                self.netstat_vec.append(csv_row)

    def netstat_trace(self):
        trace = []
        
        for _ in range(len(self.netstat_vec[0])):
            trace.append([])
        for sample in self.netstat_vec:
            for i, param_val in enumerate(sample):
                trace[i].append(param_val)

        return trace

    def show_traceplot(self, show=True):
        trace = self.netstat_trace()
        grid_column = 1
        # grid_row = int(len(netStat)/2+0.51)
        grid_row = len(trace)
        plt.figure(figsize=(5*grid_column, 3*grid_row))
        for i, statSeq in enumerate(trace):
            plt.subplot(grid_row, grid_column, i+1)
            plt.plot(range(len(statSeq)), statSeq)
        
        if show:
            plt.show()



if __name__ == "__main__":
    reader_inst = BSTERGM_posterior_work()
    reader_inst.read_from_csv("friendship_results/friendship_1chain", 2, 2)
    # print(reader_inst.MC_formation_samples[0:10])
    reader_inst.MC_dissolution_samples = reader_inst.MC_dissolution_samples[10000::10]
    reader_inst.MC_formation_samples = reader_inst.MC_formation_samples[10000::10]
    print(np.mean(reader_inst.MC_sample_trace()[0][0]), np.mean(reader_inst.MC_sample_trace()[0][1]),
        np.mean(reader_inst.MC_sample_trace()[1][0]), np.mean(reader_inst.MC_sample_trace()[1][1]))
    reader_inst.show_traceplot()
    reader_inst.show_histogram()
    reader_inst.show_acfplot()

    netstat_reader_inst = BSTERGM_latest_exchangeSampler_work()
    netstat_reader_inst.read_from_csv("friendship_results/friendship_1chain_NetworkStat")
    netstat_reader_inst.show_traceplot()