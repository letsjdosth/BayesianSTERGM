import time
import csv

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

from network import UndirectedNetwork, DirectedNetwork
from network_sampler import NetworkSampler


class BERGM:
    def __init__(self, model_fn, initial_param, obs_network, rng_seed=2021, is_formation=None, constraint_net=None, pid=None):
        #constraint: ['isformation',y0]
        self.obs_network = obs_network
        self.node_num = obs_network.node_num
        self.isDirected = False
        if isinstance(obs_network, DirectedNetwork):
            self.isDirected = True
        else:
            self.isDirected = False
        
        self.initial_param = initial_param
        self.MC_sample = [initial_param]
        
        self.model = model_fn
        
        self.random_seed = rng_seed
        self.random_gen = np.random.default_rng(seed=rng_seed)

        self.pid = None
        if pid is not None:
            self.pid = pid
        
        self.latest_exchange_sampler = None

        self.is_formation = is_formation
        self.constraint_net = constraint_net


    def propose_param(self, last_param, cov_rate):
        cov_rate = np.array([cov_rate for _ in range(len(self.initial_param))])
        cov_mat = np.diag(cov_rate)
        return self.random_gen.multivariate_normal(last_param, cov_mat)
    
    def get_exchange_sampler(self, exchange_iter, proposed_param, rng_seed):
        exchange_sampler = NetworkSampler(self.model, proposed_param,
            self.obs_network, is_formation=self.is_formation, constraint_net=self.constraint_net, rng_seed=rng_seed)
        exchange_sampler.run(exchange_iter)
        return exchange_sampler

    def log_prior(self, last_param, proposed_param):
        # 0: normal(0,100) or 1: unif
        # proposed - last
        prior_select = 0

        if prior_select==0:
            dim = len(last_param)
            zero_mean = [0 for _ in range(dim)]
            cov = np.identity(dim) * 100
            proposed_prior = multivariate_normal.pdf(proposed_param, zero_mean, cov)
            if proposed_prior == 0:
                raise ZeroDivisionError("divide by zero encountered in log in calculating log_prior")
            last_prior = multivariate_normal.pdf(last_param, zero_mean, cov)
            log_prior_val = 0
            log_prior_val += np.log(proposed_prior) #underflow warning (it is why there are above 'if' sentences)
            log_prior_val -= np.log(last_prior)
            return log_prior_val

        elif prior_select==1:
            return 0


    def log_r(self, last_param, proposed_param, exchange_sample):
        netStat_diff = self.model(self.obs_network) -  self.model(exchange_sample)
        param_diff = proposed_param - last_param
        log_r_val = np.dot(netStat_diff, param_diff) + self.log_prior(last_param, proposed_param)
        return log_r_val


    def sampler(self, exchange_iter, proposal_cov_rate, rng_seed):
        last_param = self.MC_sample[-1]
        proposed_param = self.propose_param(last_param, proposal_cov_rate)

        #exchange
        self.latest_exchange_sampler = self.get_exchange_sampler(exchange_iter, proposed_param, rng_seed=rng_seed)
        exchange_sample = self.latest_exchange_sampler.network_samples[-1]

        log_r_val = self.log_r(last_param, proposed_param, exchange_sample)

        unif_sample = self.random_gen.random()
        if np.log(unif_sample) < log_r_val:
            self.MC_sample.append(proposed_param)
        else:
            self.MC_sample.append(last_param)


    def run(self, iter, exchange_iter, proposal_cov_rate):
        start_time = time.time()
        for i in range(iter):
            rng_seed = self.random_seed + i
            self.sampler(exchange_iter, proposal_cov_rate, rng_seed)
            if i%200==0:
                if self.pid is not None:
                    print("pid:",self.pid, " iter: ", i, "/", iter, " time elapsed(second):", round(time.time()-start_time,1))
                else:
                    print("iter: ", i, "/", iter, " time elapsed(second):", round(time.time()-start_time,1))
                
        if self.pid is not None:
            print("pid:",self.pid," iter:", iter,"/",iter, "time elapsed(second):", round(time.time()-start_time,1))
        else:
            print(iter,"/",iter, "time elapsed(second):", round(time.time()-start_time,1))

    # ===============================================

    def MC_sample_trace(self):
        trace = []
        for _ in range(len(self.initial_param)):
            trace.append([])

        for sample in self.MC_sample:
            for i, param_val in enumerate(sample):
                trace[i].append(param_val)
        return trace


    def show_traceplot(self, show=True):
        trace = self.MC_sample_trace()
        grid_column = 1
        # grid_row = int(len(netStat)/2+0.51)
        grid_row = len(self.initial_param)
        plt.figure(figsize=(5*grid_column, 3*grid_row))
        for i, paramSeq in enumerate(trace):
            plt.subplot(grid_row, grid_column, i+1)
            plt.plot(range(len(paramSeq)), paramSeq)
        
        if show:
            plt.show()

    def show_latest_exchangeSampler_netStat_traceplot(self, show=True):
        self.latest_exchange_sampler.show_traceplot()
        
    def write_posterior_samples(self, filename: str):
        # print(self.MC_dissolution_samples)
        with open("pyBERGM/" + filename + '.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for sample in self.MC_sample:
                csv_row = sample.tolist()
                writer.writerow(csv_row)

    def write_latest_exchangeSampler_netStat(self, filename: str):
        netStat = self.latest_exchange_sampler.netStat_trace()
        netStat_list = (np.array(netStat).T).tolist()
    
        with open("pyBERGM/" + filename + '.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for csv_row in netStat_list:
                writer.writerow(csv_row)

    
    def show_histogram(self, bins=100, param_mark=None, show=True):
        trace = self.MC_sample_trace()
        grid_column = 1
        # grid_row = int(len(netStat)/2+0.51)
        grid_row = len(self.initial_param)
        plt.figure(figsize=(5*grid_column, 3*grid_row))
        for i, paramSeq in enumerate(trace):
            plt.subplot(grid_row, grid_column, i+1)
            plt.hist(paramSeq, bins=bins, density=True)
            plt.ylabel('parameter'+str(i))
            if param_mark is not None:
                plt.axvline(param_mark[i], color='red', linewidth=1.5)
        if show:
            plt.show()


if __name__ == "__main__":
    import data_tailor 
    sociational_interactions = [
        UndirectedNetwork(np.array(data_tailor.KAPFTS1)),
        UndirectedNetwork(np.array(data_tailor.KAPFTS2))
    ]

    def model_netStat_edgeonly(network):
        model = []
        model.append(network.statCal_edgeNum())
        return np.array(model)


    def model_netStat_edgeGWESP(network):
        model = []
        model.append(network.statCal_edgeNum())
        model.append(network.statCal_geoWeightedESP(0.25))
        return np.array(model)

    # time 0
    # BERGM_sampler00 = BERGM(model_netStat_edgeonly, np.array([0]), sociational_interactions[0])
    # BERGM_sampler00.run(10000, 30, 0.01)
    # BERGM_sampler00.show_traceplot()
    # BERGM_sampler00.show_histogram(param_mark=[-1.30559])

    # BERGM_sampler01 = BERGM(model_netStat_edgeGWESP, np.array([0,0]), sociational_interactions[0])
    # BERGM_sampler01.run(10000, 30, 0.01, True)
    # BERGM_sampler01.show_traceplot()
    # BERGM_sampler01.show_histogram(param_mark=[-3.8272, 1.6318])


    # time 1
    # BERGM_sampler10 = BERGM(model_netStat_edgeonly, np.array([0]), sociational_interactions[1])
    # BERGM_sampler10.run(10000, 30, 0.01, True)
    # BERGM_sampler10.show_traceplot()
    # BERGM_sampler10.show_histogram(param_mark=[-0.84280])

    # BERGM_sampler11 = BERGM(model_netStat_edgeGWESP, np.array([0,0]), sociational_interactions[0])
    # BERGM_sampler11.run(300, 100, 0.05, True)
    # BERGM_sampler11.show_traceplot()
    # BERGM_sampler11.show_histogram(param_mark=[-2.9464, 1.3795])
    # BERGM_sampler11.show_latest_exchangeSampler_netStat_traceplot()


    def dissociate_network(last_network, now_network, isDirected):
        y_last_structure = last_network.structure
        y_now_structure = now_network.structure
        y_plus = y_last_structure.copy()
        y_minus = y_last_structure.copy()
        for row in range(last_network.node_num):
            for col in range(last_network.node_num):
                if y_now_structure[row,col]==1:
                    y_plus[row,col]=1
                if y_now_structure[row,col]==0:
                    y_minus[row,col]=0
        result = 0
        if isDirected:
            result = (DirectedNetwork(y_plus), DirectedNetwork(y_minus))
        else:
            result = (UndirectedNetwork(y_plus), UndirectedNetwork(y_minus))
        return result
    
    y_plus, y_minus = dissociate_network(sociational_interactions[0], sociational_interactions[1], False)

    #y+
    BERGM_sampler20 = BERGM(model_netStat_edgeonly, np.array([0]), y_plus)
    BERGM_sampler20.run(10000, 30, 0.01)
    # BERGM_sampler20.MC_formation_samples = BERGM_sampler20.MC_formation_samples[2000::2]
    # BERGM_sampler20.MC_dissolution_samples = BERGM_sampler20.MC_dissolution_samples[2000::2]
    print("mean:", np.mean(BERGM_sampler20.MC_sample_trace()[0]))
    BERGM_sampler20.show_traceplot()
    BERGM_sampler20.show_histogram(param_mark=[-0.51011])
    print("degenerated case edge num:", 39*38/2)
    BERGM_sampler20.show_latest_exchangeSampler_netStat_traceplot()

    # BERGM_sampler21 = BERGM(model_netStat_edgeGWESP, np.array([-1,1]), y_plus)
    # BERGM_sampler21.run(1000, 100, 0.01)
    # BERGM_sampler21.show_traceplot()
    # BERGM_sampler21.show_histogram(param_mark=[-1.7853, 0.9149])
    # BERGM_sampler21.show_latest_exchangeSampler_netStat_traceplot()

    #y+ constraint
    BERGM_sampler20c_f = BERGM(model_netStat_edgeonly, np.array([0]), y_plus, is_formation=True, constraint_net=sociational_interactions[0])
    BERGM_sampler20c_f.run(10000, 30, 0.01)
    # BERGM_sampler20c_f.MC_formation_samples = BERGM_sampler20.MC_formation_samples[2000::2]
    # BERGM_sampler20c_f.MC_dissolution_samples = BERGM_sampler20.MC_dissolution_samples[2000::2]
    print("mean:", np.mean(BERGM_sampler20c_f.MC_sample_trace()[0]))
    BERGM_sampler20c_f.show_traceplot()
    BERGM_sampler20c_f.show_histogram(param_mark=[-1.3502])
    print("constraint(lower bound) netStat:", model_netStat_edgeonly(sociational_interactions[0]))
    print("degenerated case edge num:", 39*38/2)
    BERGM_sampler20c_f.show_latest_exchangeSampler_netStat_traceplot()


    # BERGM_sampler21c_f = BERGM(model_netStat_edgeGWESP, np.array([-2,1]), y_plus, is_formation=True, constraint_net=sociational_interactions[0])
    # BERGM_sampler21c_f.run(1000, 100, 0.01)
    # BERGM_sampler21c_f.show_traceplot()
    # BERGM_sampler21c_f.show_histogram(param_mark=[-2.5621, 0.8827])
    # print("constraint(lower bound) netStat:", model_netStat_edgeGWESP(sociational_interactions[0]))
    # print("degenerated case edge num:", 39*38/2)
    # BERGM_sampler21c_f.show_latest_exchangeSampler_netStat_traceplot()
    

    #y-
    BERGM_sampler30 = BERGM(model_netStat_edgeonly, np.array([0]), y_minus)
    BERGM_sampler30.run(10000, 30, 0.01)
    # BERGM_sampler30.MC_formation_samples = BERGM_sampler20.MC_formation_samples[2000::2]
    # BERGM_sampler30.MC_dissolution_samples = BERGM_sampler20.MC_dissolution_samples[2000::2]
    print("mean:", np.mean(BERGM_sampler30.MC_sample_trace()[0]))
    BERGM_sampler30.show_traceplot()
    BERGM_sampler30.show_histogram(param_mark=[-1.8236])
    print("degenerated case edge num:", 39*38/2)
    BERGM_sampler30.show_latest_exchangeSampler_netStat_traceplot()

    # BERGM_sampler31 = BERGM(model_netStat_edgeGWESP, np.array([-3,1]), y_minus)
    # BERGM_sampler31.run(1000, 100, 0.01)
    # BERGM_sampler31.show_traceplot()
    # BERGM_sampler31.show_histogram(param_mark=[-3.4801, 1.185])
    # BERGM_sampler31.show_latest_exchangeSampler_netStat_traceplot()

    #y- constraint
    BERGM_sampler30c_d = BERGM(model_netStat_edgeonly, np.array([0]), y_minus, is_formation=False, constraint_net=sociational_interactions[0])
    BERGM_sampler30c_d.run(10000, 30, 0.01)
    # BERGM_sampler30c_d.MC_formation_samples = BERGM_sampler20.MC_formation_samples[2000::2]
    # BERGM_sampler30c_d.MC_dissolution_samples = BERGM_sampler20.MC_dissolution_samples[2000::2]
    print("mean:", np.mean(BERGM_sampler30c_d.MC_sample_trace()[0]))
    BERGM_sampler30c_d.show_traceplot()
    BERGM_sampler30c_d.show_histogram(param_mark=[0.6274])
    print("constraint(upper bound) netStat:", model_netStat_edgeonly(sociational_interactions[0]))
    print("degenerated case edge num:", 39*38/2)
    BERGM_sampler30c_d.show_latest_exchangeSampler_netStat_traceplot()

    # BERGM_sampler31c_d = BERGM(model_netStat_edgeGWESP, np.array([0,0]), y_minus, is_formation=False, constraint_net=sociational_interactions[0])
    # BERGM_sampler31c_d.run(1000, 100, 0.01)
    # BERGM_sampler31c_d.show_traceplot()
    # BERGM_sampler31c_d.show_histogram(param_mark = [-0.1878, 0.5118])
    # print("constraint(upper bound) netStat:", model_netStat_edgeGWESP(sociational_interactions[0]))
    # print("degenerated case edge num:", 39*38/2)
    # BERGM_sampler31c_d.show_latest_exchangeSampler_netStat_traceplot()