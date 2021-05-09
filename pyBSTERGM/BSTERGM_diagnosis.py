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

    def read_from_BSTERGM_csv(self, file_name, formation_param_dim, dissolution_param_dim):
        '''caution:
        Returned BSTERGM_posterior_work instance by using this function has propoerties merely as following:
            self.initial_formation_param(if the sample is cut burn-in, this value may not the real initial value.)
            self.initial_dissolution_param(same as above)
            self.MC_formation_samples
            self.MC_dissolution_samples
        Thus, DO NOT use this instance to replace original BSTERGM object.
        '''
        with open("pyBSTERGM/" + file_name + '.csv', 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for csv_row in reader:
                csv_row = [float(elem) for elem in csv_row]
                self.MC_formation_samples.append(np.array(csv_row[0:formation_param_dim]))
                self.MC_dissolution_samples.append(np.array(csv_row[formation_param_dim : formation_param_dim + dissolution_param_dim]))

        self.initial_formation_param = self.MC_formation_samples[0]
        self.initial_dissolution_param = self.MC_dissolution_samples[0]

    def read_from_BERGM_csv(self, file_name, param_dim, is_formation):
        pass

    def read_from_BSTERGM_instance(self, BSTERGM_obj):
        pass

    def read_from_BERGM_instance(self, BERGM_obj, is_formation):
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


    def show_traceplot(self, mean_hline=False, show=True):
        formation_trace, dissolution_trace = self.MC_sample_trace()
        grid_column = 1
        # grid_row = int(len(netStat)/2+0.51)
        grid_row = len(self.initial_formation_param) + len(self.initial_dissolution_param)
        plt.figure(figsize=(5*grid_column, 3*grid_row))
        for i, paramSeq in enumerate(formation_trace):
            plt.subplot(grid_row, grid_column, i+1)
            plt.plot(range(len(paramSeq)), paramSeq)
            if mean_hline:
                plt.axhline(np.mean(paramSeq), color='red', linewidth=1.5)
        for i, paramSeq in enumerate(dissolution_trace):
            plt.subplot(grid_row, grid_column, len(self.initial_formation_param)+i+1)
            plt.plot(range(len(paramSeq)), paramSeq)
            if mean_hline:
                plt.axhline(np.mean(paramSeq), color='red', linewidth=1.5)
        if show:
            plt.show()

    def show_histogram(self, bins=100, mean_vline=False, formation_mark=None, dissolution_mark=None, show=True):
        formation_trace, dissolution_trace = self.MC_sample_trace()
        grid_column = 2
        # grid_row = int(len(netStat)/2+0.51)
        grid_row = (len(self.initial_formation_param) + len(self.initial_dissolution_param))/2
        plt.figure(figsize=(5*grid_column, 3*grid_row))
        for i, paramSeq in enumerate(formation_trace):
            plt.subplot(grid_row, grid_column, i+1)
            plt.hist(paramSeq, bins=bins, density=True)
            plt.ylabel('formation'+str(i))
            if formation_mark is not None:
                plt.axvline(formation_mark[i], color='red', linewidth=1.5)
            if mean_vline:
                plt.axvline(np.mean(paramSeq), color='black', linewidth=1.5)

        for i, paramSeq in enumerate(dissolution_trace):
            plt.subplot(grid_row, grid_column, len(self.initial_formation_param)+i+1)
            plt.hist(paramSeq, bins=bins, density=True)
            plt.ylabel('dissolution'+str(i))
            if dissolution_mark is not None:
                plt.axvline(dissolution_mark[i], color='red', linewidth=1.5)
            if mean_vline:
                plt.axvline(np.mean(paramSeq), color='black', linewidth=1.5)

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

    def show_traceplot(self, hline_vec=None, show=True):
        trace = self.netstat_trace()
        grid_column = 1
        # grid_row = int(len(netStat)/2+0.51)
        grid_row = len(trace)
        plt.figure(figsize=(5*grid_column, 3*grid_row))
        for i, statSeq in enumerate(trace):
            plt.subplot(grid_row, grid_column, i+1)
            plt.plot(range(len(statSeq)), statSeq)
            if hline_vec is not None:
                plt.axhline(hline_vec[i], color='red', linewidth=1.5)

        if show:
            plt.show()



if __name__ == "__main__":
    #samplk joint
    #good: 2/3/4 chain / others: bad
    reader_inst_samplk_vig = BSTERGM_posterior_work()
    reader_inst_samplk_vig.read_from_BSTERGM_csv("example_results_joint/samplk_jointtimelag_normPrior_vignettesEx_4chain", 4, 4)
    reader_inst_samplk_vig.MC_formation_samples = reader_inst_samplk_vig.MC_formation_samples[10000::20]
    reader_inst_samplk_vig.MC_dissolution_samples = reader_inst_samplk_vig.MC_dissolution_samples[10000::20]
    reader_inst_samplk_vig.show_traceplot()
    reader_inst_samplk_vig.show_histogram(formation_mark=[-3.5586, 2.2624, -0.4994, 0.2945],
        dissolution_mark=[-0.1164, 1.5791, -1.6957, 0.6847])
    reader_inst_samplk_vig.show_acfplot()
    
    netstat_reader_inst_samplk_vig = BSTERGM_latest_exchangeSampler_work()
    netstat_reader_inst_samplk_vig.read_from_csv("example_results_joint/samplk_jointtimelag_normPrior_vignettesEx_4chain_NetworkStat")
    netstat_reader_inst_samplk_vig.show_traceplot()

    #friendship joint
    #good: 2 or 5 (-_-)
    reader_inst_friendship_KHEx = BSTERGM_posterior_work()
    reader_inst_friendship_KHEx.read_from_BSTERGM_csv("example_results_joint/friendship_jointtimelag_normPrior_KHEx_2chain", 8, 8)
    reader_inst_friendship_KHEx.MC_formation_samples = reader_inst_friendship_KHEx.MC_formation_samples[1000::20]
    reader_inst_friendship_KHEx.MC_dissolution_samples = reader_inst_friendship_KHEx.MC_dissolution_samples[1000::20]
    reader_inst_friendship_KHEx.show_traceplot()
    reader_inst_friendship_KHEx.show_histogram(formation_mark=[-3.336, 0.480, 0.973, -0.358, 0.650, 1.384, 0.886, -0.389],
        dissolution_mark=[-1.132, 0.122, 1.168, -0.577, 0.451, 2.682, 1.121, -1.016])
    reader_inst_friendship_KHEx.show_acfplot()
    
    netstat_reader_inst_friendship_KHEx = BSTERGM_latest_exchangeSampler_work()
    netstat_reader_inst_friendship_KHEx.read_from_csv("example_results_joint/friendship_jointtimelag_normPrior_KHEx_2chain_NetworkStat")
    netstat_reader_inst_friendship_KHEx.show_traceplot()

