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

    def read_from_BERGM_csv(self, formation_file_name, dissolution_file_name):
        with open("pyBSTERGM/" + formation_file_name + '.csv', 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for csv_row in reader:
                csv_row = [float(elem) for elem in csv_row]
                self.MC_formation_samples.append(np.array(csv_row))
        with open("pyBSTERGM/" + dissolution_file_name + '.csv', 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for csv_row in reader:
                csv_row = [float(elem) for elem in csv_row]
                self.MC_dissolution_samples.append(np.array(csv_row))

        self.initial_formation_param = self.MC_formation_samples[0]
        self.initial_dissolution_param = self.MC_dissolution_samples[0]

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


    def print_summary(self):
        formation_trace, dissolution_trace = self.MC_sample_trace()
        formation_means = []
        formation_sds = []
        dissolution_means = []
        dissolution_sds = []
        for samples in formation_trace:
            formation_means.append(np.mean(samples))
            formation_sds.append(np.std(samples))
        for samples in dissolution_trace:
            dissolution_means.append(np.mean(samples))
            dissolution_sds.append(np.std(samples))
        print("formation")
        for i, mean,sd in zip(range(len(formation_trace)), formation_means, formation_sds):
            print("f",i, "\t", round(mean,3), " & ", round(sd,3))
        
        print("dissolution")
        for i, mean,sd in zip(range(len(dissolution_trace)), dissolution_means, dissolution_sds):
            print("f",i, "\t", round(mean,3), " & ", round(sd,3))

    def show_traceplot(self, mean_hline=False, layout=None, show=True):
        formation_trace, dissolution_trace = self.MC_sample_trace()
        
        grid_row, grid_column = (0,0)
        if layout is not None:
            grid_row, grid_column = layout
        else:
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

    def show_histogram(self, bins=100, mean_vline=False, formation_mark=None, dissolution_mark=None, layout=None, show=True):
        formation_trace, dissolution_trace = self.MC_sample_trace()
        
        grid_row, grid_column = (0,0)
        if layout is not None:
            grid_row, grid_column = layout
        else:
            grid_column = 2
            # grid_row = int(len(netStat)/2+0.51)
            grid_row = (len(self.initial_formation_param) + len(self.initial_dissolution_param))//2
        
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

    def show_acfplot(self, maxLag=50, layout=None, show=True):
        formation_trace, dissolution_trace = self.MC_sample_trace()
        
        
        grid_row, grid_column = (0,0)
        if layout is not None:
            grid_row, grid_column = layout
        else:
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
    #example:
    reader_inst_tailorshop_edgeGWESP = BSTERGM_posterior_work()
    reader_inst_tailorshop_edgeGWESP.read_from_BERGM_csv("example_results_tailorshop/tailorshop_t01_normPrior_edgeGWESP_0chain_formation",
                                                        "example_results_tailorshop/tailorshop_t01_normPrior_edgeGWESP_2chain_dissolution")
    reader_inst_tailorshop_edgeGWESP_conti = BSTERGM_posterior_work()
    reader_inst_tailorshop_edgeGWESP_conti.read_from_BERGM_csv("example_results_tailorshop/tailorshop_jointly_normPrior_edgeGWESP_conti_0chain_formation",
                                                        "example_results_tailorshop/tailorshop_jointly_normPrior_edgeGWESP_conti_3chain_dissolution")
    reader_inst_tailorshop_edgeGWESP.MC_formation_samples = reader_inst_tailorshop_edgeGWESP.MC_formation_samples + reader_inst_tailorshop_edgeGWESP_conti.MC_formation_samples
    reader_inst_tailorshop_edgeGWESP.MC_dissolution_samples = reader_inst_tailorshop_edgeGWESP.MC_dissolution_samples + reader_inst_tailorshop_edgeGWESP_conti.MC_dissolution_samples


    reader_inst_tailorshop_edgeGWESP.MC_formation_samples = reader_inst_tailorshop_edgeGWESP.MC_formation_samples[10000::40]
    reader_inst_tailorshop_edgeGWESP.MC_dissolution_samples = reader_inst_tailorshop_edgeGWESP.MC_dissolution_samples[10000::40]
    reader_inst_tailorshop_edgeGWESP.show_traceplot()
    reader_inst_tailorshop_edgeGWESP.show_histogram(formation_mark=[-2.5621, 0.8827],
        dissolution_mark=[-0.1878, 0.5118])
    reader_inst_tailorshop_edgeGWESP.show_acfplot()

    netstat_reader_inst_tailorshop_edgeGWESP_f = BSTERGM_latest_exchangeSampler_work()
    netstat_reader_inst_tailorshop_edgeGWESP_f.read_from_csv("example_results_tailorshop/tailorshop_jointly_normPrior_edgeGWESP_conti_0chain_formation_NetworkStat")
    netstat_reader_inst_tailorshop_edgeGWESP_f.show_traceplot()
    netstat_reader_inst_tailorshop_edgeGWESP_d = BSTERGM_latest_exchangeSampler_work()
    netstat_reader_inst_tailorshop_edgeGWESP_d.read_from_csv("example_results_tailorshop/tailorshop_jointly_normPrior_edgeGWESP_conti_3chain_dissolution_NetworkStat")
    netstat_reader_inst_tailorshop_edgeGWESP_d.show_traceplot()

