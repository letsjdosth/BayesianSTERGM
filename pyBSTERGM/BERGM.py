import time
import csv
import math

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
        self.obs_network_netStat = self.model(self.obs_network)
        
        self.random_seed = rng_seed
        self.random_gen = np.random.default_rng(seed=rng_seed)

        self.pid = None
        if pid is not None:
            self.pid = pid
        
        self.latest_exchange_sampler = None

        self.is_formation = is_formation
        self.constraint_net = constraint_net


    def propose_param(self, last_param, cov_rate_vec):
        cov_mat = np.diag(cov_rate_vec)
        return self.random_gen.multivariate_normal(last_param, cov_mat)
    
    def get_exchange_sampler(self, exchange_iter, proposed_param, rng_seed):
        exchange_sampler = NetworkSampler(self.model, proposed_param,
            self.obs_network, is_formation=self.is_formation, constraint_net=self.constraint_net, rng_seed=rng_seed)
        exchange_sampler.run(exchange_iter)
        return exchange_sampler

    def log_prior(self, last_param, proposed_param, dist='normal'):
        # proposed - last
        if dist=='normal':
            #normal(last_param, var=100)
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

        elif dist=='unif':
            return 0
        else:
            ValueError("dist:", dist, "is not implemented. choose either 'normal' or 'unif'.")            


    def log_r(self, last_param, proposed_param, exchange_sample):
        netStat_diff = self.obs_network_netStat -  self.model(exchange_sample)
        param_diff = proposed_param - last_param
        log_r_val = np.dot(netStat_diff, param_diff) + self.log_prior(last_param, proposed_param)
        return log_r_val


    def sampler(self, exchange_iter, cov_rate_vec, rng_seed):
        last_param = self.MC_sample[-1]
        proposed_param = self.propose_param(last_param, cov_rate_vec)

        #exchange
        self.latest_exchange_sampler = self.get_exchange_sampler(exchange_iter, proposed_param, rng_seed=rng_seed)
        exchange_sample = self.latest_exchange_sampler.network_samples[-1]

        try:
            log_r_val = self.log_r(last_param, proposed_param, exchange_sample)
        except ZeroDivisionError:
            log_r_val = -math.inf

        unif_sample = self.random_gen.random()
        if np.log(unif_sample) < log_r_val:
            self.MC_sample.append(proposed_param)
        else:
            self.MC_sample.append(last_param)

    def proposal_cov_rate_setting(self, proposal_cov_rate):
        # float or list[dim=len(param)]
        if isinstance(proposal_cov_rate, float):
            dim = len(self.MC_sample[-1])
            cov_rate_vec = [proposal_cov_rate for _ in range(dim)]
        elif isinstance(proposal_cov_rate, list):
            cov_rate_vec = proposal_cov_rate
        return cov_rate_vec

    def run(self, iter, exchange_iter, proposal_cov_rate, console_output_str=""):
        start_time = time.time()
        cov_rate_vec = self.proposal_cov_rate_setting(proposal_cov_rate)

        for i in range(iter):
            rng_seed = self.random_seed + i
            self.sampler(exchange_iter, cov_rate_vec, rng_seed)
            if i%200==0:
                if self.pid is not None:
                    print("pid:",self.pid, " ", console_output_str, " iter: ", i, "/", iter, " time elapsed(second):", round(time.time()-start_time,1))
                else:
                    print(console_output_str, " iter: ", i, "/", iter, " time elapsed(second):", round(time.time()-start_time,1))
                
        if self.pid is not None:
            print("pid:",self.pid, " ", console_output_str, " iter:", iter,"/",iter, "time elapsed(second):", round(time.time()-start_time,1))
        else:
            print(console_output_str, " iter:", iter,"/",iter, "time elapsed(second):", round(time.time()-start_time,1))

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
        grid_row = len(self.initial_param)
        plt.figure(figsize=(5*grid_column, 3*grid_row))
        for i, paramSeq in enumerate(trace):
            plt.subplot(grid_row, grid_column, i+1)
            plt.plot(range(len(paramSeq)), paramSeq)

        if show:
            plt.show()

    
    def show_traceplot_eachaxis(self, axis_idx, show=False):
        axis_trace = self.MC_sample_trace()[axis_idx]
        plt.plot(range(len(axis_trace)), axis_trace)
        if show:
            plt.show()

    def show_latest_exchangeSampler_netStat_traceplot(self, show=True):
        self.latest_exchange_sampler.show_traceplot()
        
    def write_posterior_samples(self, filename: str):
        with open("pyBSTERGM/" + filename + '.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for sample in self.MC_sample:
                csv_row = sample.tolist()
                writer.writerow(csv_row)

    def write_latest_exchangeSampler_netStat(self, filename: str):
        with open("pyBSTERGM/" + filename + '.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for val_vec in self.latest_exchange_sampler.network_samples_netStats:
                csv_row = val_vec.tolist()
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

    def show_histogram_eachaxis(self, axis_idx, param_mark=None, bins=100, show=False):
        axis_trace = self.MC_sample_trace()[axis_idx]
        plt.hist(axis_trace, bins=bins, density=True)
        if param_mark is not None:
            plt.axvline(param_mark, color='red', linewidth=1.5)
        if show:
            plt.show()








# #=============================================================================================

# def propose_param_1dim(self, dim_idx, last_param, cov_rate):
#     result =[val for val in last_param]
#     new_val = self.random_gen.normal(last_param[dim_idx], cov_rate)
#     result[dim_idx] = new_val
#     return np.array(result)

# def log_r_1dim_formation(self, start_time_lag, last_formation_param, last_dissolution_param,
#         proposed_formation_param, proposed_dissolution_param,
#         exchange_formation):
#     formation_netStat_diff = self.model(self.obs_network_formation_seq[start_time_lag+1]) - self.model(exchange_formation)

#     log_r_val = np.dot(proposed_formation_param - last_formation_param, formation_netStat_diff)
#     log_r_val += self.log_prior(last_formation_param, last_dissolution_param,
#         proposed_formation_param, proposed_dissolution_param)        
#     return log_r_val

# def log_r_1dim_dissolution(self, start_time_lag, last_formation_param, last_dissolution_param,
#         proposed_formation_param, proposed_dissolution_param,
#         exchange_dissolution):
#     dissolution_netStat_diff = self.model(self.obs_network_dissolution_seq[start_time_lag+1]) - self.model(exchange_dissolution)

#     log_r_val = np.dot(proposed_dissolution_param - last_dissolution_param, dissolution_netStat_diff)
#     log_r_val += self.log_prior(last_formation_param, last_dissolution_param,
#         proposed_formation_param, proposed_dissolution_param)        
#     return log_r_val

# def sampler_1dim(self, start_time_lag, exchange_iter, rng_seed):
#     last_formation_param = self.MC_formation_samples[-1]
#     last_dissolution_param = self.MC_dissolution_samples[-1]
    
#     now_formation_param = np.array([val for val in last_formation_param])
#     now_dissolution_param = np.array([val for val in last_dissolution_param])

#     for i_idx in range(len(last_formation_param)):
#         #proposal
#         proposal_cov_rate = self.formation_cov_rate[i_idx]
#         proposed_formation_param = self.propose_param_1dim(i_idx, now_formation_param, proposal_cov_rate)
#         proposed_dissolution_param = now_dissolution_param

#         #exchange
#         self.latest_exchange_formation_sampler = self.get_exchange_sampler(start_time_lag, exchange_iter, proposed_formation_param, is_formation=True, rng_seed=rng_seed)
#         exchange_formation_sample = self.latest_exchange_formation_sampler.network_samples[-1]

#         #MCMC
#         log_r_val = self.log_r_1dim_formation(start_time_lag, now_formation_param, now_dissolution_param,
#             proposed_formation_param, proposed_dissolution_param,
#             exchange_formation_sample)

#         unif_sample = self.random_gen.random()
#         if np.log(unif_sample) < log_r_val:
#             now_formation_param = proposed_formation_param
#         else:
#             pass

#     for i_idx in range(len(last_dissolution_param)):
#         #proposal
#         proposal_cov_rate = self.dissolution_cov_rate[i_idx]
#         proposed_formation_param = now_formation_param
#         proposed_dissolution_param = self.propose_param_1dim(i_idx, now_dissolution_param, proposal_cov_rate)

#         #exchange
#         self.latest_exchange_dissolution_sampler = self.get_exchange_sampler(start_time_lag, exchange_iter, proposed_dissolution_param, is_formation=False, rng_seed=rng_seed*10)
#         exchange_dissolution_sample = self.latest_exchange_dissolution_sampler.network_samples[-1]

#         #MCMC
#         log_r_val = self.log_r_1dim_dissolution(start_time_lag, now_formation_param, now_dissolution_param,
#             proposed_formation_param, proposed_dissolution_param,
#             exchange_dissolution_sample)

#         unif_sample = self.random_gen.random()
#         if np.log(unif_sample) < log_r_val:
#             now_dissolution_param = proposed_dissolution_param
#         else:
#             pass

#     self.MC_formation_samples.append(now_formation_param)
#     self.MC_dissolution_samples.append(now_dissolution_param)


# def run_1dim(self, iter, exchange_iter=30, proposal_cov_rate=0.1):
#     # proposal_cov_rate: float or
#     #   dict structured by {"formation_cov_rate": [0,...], "dissolution_cov_rate":[0,...]}
#     start_time = time.time()
#     self.proposal_cov_rate_setting(proposal_cov_rate)
#     for i in range(iter):
#         start_time_lag = self.random_gen.integers(len(self.obs_network_seq)-1)
#         rng_seed = self.random_seed + i
#         self.sampler_1dim(start_time_lag, exchange_iter, rng_seed)
#         if i%50==0:
#             if self.pid is not None:
#                 print("pid:",self.pid, " iter: ", i, "/", iter, " time elapsed(second):", round(time.time()-start_time,1))
#             else:
#                 print("iter: ", i, "/", iter, " time elapsed(second):", round(time.time()-start_time,1))
            
#     if self.pid is not None:
#         print("pid:",self.pid," iter:", iter,"/",iter, "time elapsed(second):", round(time.time()-start_time,1))
#     else:
#         print(iter,"/",iter, "time elapsed(second):", round(time.time()-start_time,1))



if __name__ == "__main__":
    import data_samplk, data_knecht_friendship, data_tailor
    
    sociational_interactions = [
        UndirectedNetwork(np.array(data_tailor.KAPFTS1)),
        UndirectedNetwork(np.array(data_tailor.KAPFTS2))
    ]
    friendship_sequence = [
        DirectedNetwork(np.array(data_knecht_friendship.friendship_t1)),
        DirectedNetwork(np.array(data_knecht_friendship.friendship_t2)),
        DirectedNetwork(np.array(data_knecht_friendship.friendship_t3)),
        DirectedNetwork(np.array(data_knecht_friendship.friendship_t4))
    ]
    samplk_sequence = [
        DirectedNetwork(np.array(data_samplk.samplk1)),
        DirectedNetwork(np.array(data_samplk.samplk2)),
        DirectedNetwork(np.array(data_samplk.samplk3))
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
    
    tailor_y_plus, tailor_y_minus = dissociate_network(sociational_interactions[0], sociational_interactions[1], False)

    #y+
    BERGM_sampler20 = BERGM(model_netStat_edgeonly, np.array([0]), tailor_y_plus)
    BERGM_sampler20.run(10000, 30, 0.01)
    # BERGM_sampler20.MC_formation_samples = BERGM_sampler20.MC_formation_samples[2000::2]
    # BERGM_sampler20.MC_dissolution_samples = BERGM_sampler20.MC_dissolution_samples[2000::2]
    print("mean:", np.mean(BERGM_sampler20.MC_sample_trace()[0]))
    BERGM_sampler20.show_traceplot()
    BERGM_sampler20.show_histogram(param_mark=[-0.51011])
    print("degenerated case edge num:", 39*38/2)
    BERGM_sampler20.show_latest_exchangeSampler_netStat_traceplot()

    # BERGM_sampler21 = BERGM(model_netStat_edgeGWESP, np.array([-1,1]), tailor_y_plus)
    # BERGM_sampler21.run(1000, 100, 0.01)
    # BERGM_sampler21.show_traceplot()
    # BERGM_sampler21.show_histogram(param_mark=[-1.7853, 0.9149])
    # BERGM_sampler21.show_latest_exchangeSampler_netStat_traceplot()

    #y+ constraint
    BERGM_sampler20c_f = BERGM(model_netStat_edgeonly, np.array([0]), tailor_y_plus, is_formation=True, constraint_net=sociational_interactions[0])
    BERGM_sampler20c_f.run(10000, 30, 0.01)
    # BERGM_sampler20c_f.MC_formation_samples = BERGM_sampler20.MC_formation_samples[2000::2]
    # BERGM_sampler20c_f.MC_dissolution_samples = BERGM_sampler20.MC_dissolution_samples[2000::2]
    print("mean:", np.mean(BERGM_sampler20c_f.MC_sample_trace()[0]))
    BERGM_sampler20c_f.show_traceplot()
    BERGM_sampler20c_f.show_histogram(param_mark=[-1.3502])
    print("constraint(lower bound) netStat:", model_netStat_edgeonly(sociational_interactions[0]))
    print("degenerated case edge num:", 39*38/2)
    BERGM_sampler20c_f.show_latest_exchangeSampler_netStat_traceplot()


    # BERGM_sampler21c_f = BERGM(model_netStat_edgeGWESP, np.array([-2,1]), tailor_y_plus, is_formation=True, constraint_net=sociational_interactions[0])
    # BERGM_sampler21c_f.run(1000, 100, 0.01)
    # BERGM_sampler21c_f.show_traceplot()
    # BERGM_sampler21c_f.show_histogram(param_mark=[-2.5621, 0.8827])
    # print("constraint(lower bound) netStat:", model_netStat_edgeGWESP(sociational_interactions[0]))
    # print("degenerated case edge num:", 39*38/2)
    # BERGM_sampler21c_f.show_latest_exchangeSampler_netStat_traceplot()
    

    #y-
    BERGM_sampler30 = BERGM(model_netStat_edgeonly, np.array([0]), tailor_y_minus)
    BERGM_sampler30.run(10000, 30, 0.01)
    # BERGM_sampler30.MC_formation_samples = BERGM_sampler20.MC_formation_samples[2000::2]
    # BERGM_sampler30.MC_dissolution_samples = BERGM_sampler20.MC_dissolution_samples[2000::2]
    print("mean:", np.mean(BERGM_sampler30.MC_sample_trace()[0]))
    BERGM_sampler30.show_traceplot()
    BERGM_sampler30.show_histogram(param_mark=[-1.8236])
    print("degenerated case edge num:", 39*38/2)
    BERGM_sampler30.show_latest_exchangeSampler_netStat_traceplot()

    # BERGM_sampler31 = BERGM(model_netStat_edgeGWESP, np.array([-3,1]), tailor_y_minus)
    # BERGM_sampler31.run(1000, 100, 0.01)
    # BERGM_sampler31.show_traceplot()
    # BERGM_sampler31.show_histogram(param_mark=[-3.4801, 1.185])
    # BERGM_sampler31.show_latest_exchangeSampler_netStat_traceplot()

    #y- constraint
    BERGM_sampler30c_d = BERGM(model_netStat_edgeonly, np.array([0]), tailor_y_minus, is_formation=False, constraint_net=sociational_interactions[0])
    BERGM_sampler30c_d.run(10000, 30, 0.01)
    # BERGM_sampler30c_d.MC_formation_samples = BERGM_sampler20.MC_formation_samples[2000::2]
    # BERGM_sampler30c_d.MC_dissolution_samples = BERGM_sampler20.MC_dissolution_samples[2000::2]
    print("mean:", np.mean(BERGM_sampler30c_d.MC_sample_trace()[0]))
    BERGM_sampler30c_d.show_traceplot()
    BERGM_sampler30c_d.show_histogram(param_mark=[0.6274])
    print("constraint(upper bound) netStat:", model_netStat_edgeonly(sociational_interactions[0]))
    print("degenerated case edge num:", 39*38/2)
    BERGM_sampler30c_d.show_latest_exchangeSampler_netStat_traceplot()

    # BERGM_sampler31c_d = BERGM(model_netStat_edgeGWESP, np.array([0,0]), tailor_y_minus, is_formation=False, constraint_net=sociational_interactions[0])
    # BERGM_sampler31c_d.run(1000, 100, 0.01)
    # BERGM_sampler31c_d.show_traceplot()
    # BERGM_sampler31c_d.show_histogram(param_mark = [-0.1878, 0.5118])
    # print("constraint(upper bound) netStat:", model_netStat_edgeGWESP(sociational_interactions[0]))
    # print("degenerated case edge num:", 39*38/2)
    # BERGM_sampler31c_d.show_latest_exchangeSampler_netStat_traceplot()


    # friendship01_y_plus, friendship01_y_minus = dissociate_network(friendship_sequence[0], friendship_sequence[1], True)

    # BERGM_sampler40c_f = BERGM(model_netStat_edgeonly, np.array([0]), friendship01_y_plus, is_formation=True, constraint_net=friendship_sequence[0])
    # BERGM_sampler40c_f.run(10000, 30, 0.01)
    # # BERGM_sampler40c_f.MC_formation_samples = BERGM_sampler20.MC_formation_samples[2000::2]
    # # BERGM_sampler40c_f.MC_dissolution_samples = BERGM_sampler20.MC_dissolution_samples[2000::2]
    # print("mean:", np.mean(BERGM_sampler40c_f.MC_sample_trace()[0]))
    # BERGM_sampler40c_f.show_traceplot()
    # BERGM_sampler40c_f.show_histogram(param_mark=[-2.2235])
    # print("constraint(lower bound) netStat:", model_netStat_edgeonly(friendship_sequence[0]))
    # print("degenerated case edge num:", 25*24/2)
    # BERGM_sampler40c_f.show_latest_exchangeSampler_netStat_traceplot()

    # BERGM_sampler50c_d = BERGM(model_netStat_edgeonly, np.array([0]), friendship01_y_minus, is_formation=False, constraint_net=friendship_sequence[0])
    # BERGM_sampler50c_d.run(10000, 30, 0.01)
    # # BERGM_sampler50c_d.MC_formation_samples = BERGM_sampler20.MC_formation_samples[2000::2]
    # # BERGM_sampler50c_d.MC_dissolution_samples = BERGM_sampler20.MC_dissolution_samples[2000::2]
    # print("mean:", np.mean(BERGM_sampler50c_d.MC_sample_trace()[0]))
    # BERGM_sampler50c_d.show_traceplot()
    # BERGM_sampler50c_d.show_histogram(param_mark=[0.6091])
    # print("constraint(upper bound) netStat:", model_netStat_edgeonly(friendship_sequence[0]))
    # print("degenerated case edge num:", 25*24/2)
    # BERGM_sampler50c_d.show_latest_exchangeSampler_netStat_traceplot()

    
    # friendship12_y_plus, friendship12_y_minus = dissociate_network(friendship_sequence[1], friendship_sequence[2], True)

    # BERGM_sampler60c_f = BERGM(model_netStat_edgeonly, np.array([0]), friendship12_y_plus, is_formation=True, constraint_net=friendship_sequence[1])
    # BERGM_sampler60c_f.run(10000, 30, 0.01)
    # # BERGM_sampler60c_f.MC_formation_samples = BERGM_sampler20.MC_formation_samples[2000::2]
    # # BERGM_sampler60c_f.MC_dissolution_samples = BERGM_sampler20.MC_dissolution_samples[2000::2]
    # print("mean:", np.mean(BERGM_sampler60c_f.MC_sample_trace()[0]))
    # BERGM_sampler60c_f.show_traceplot()
    # BERGM_sampler60c_f.show_histogram(param_mark=[-1.9207])
    # print("constraint(lower bound) netStat:", model_netStat_edgeonly(friendship_sequence[1]))
    # print("degenerated case edge num:", 25*24/2)
    # BERGM_sampler60c_f.show_latest_exchangeSampler_netStat_traceplot()

    # BERGM_sampler70c_d = BERGM(model_netStat_edgeonly, np.array([0]), friendship12_y_minus, is_formation=False, constraint_net=friendship_sequence[1])
    # BERGM_sampler70c_d.run(10000, 30, 0.01)
    # # BERGM_sampler70c_d.MC_formation_samples = BERGM_sampler20.MC_formation_samples[2000::2]
    # # BERGM_sampler70c_d.MC_dissolution_samples = BERGM_sampler20.MC_dissolution_samples[2000::2]
    # print("mean:", np.mean(BERGM_sampler70c_d.MC_sample_trace()[0]))
    # BERGM_sampler70c_d.show_traceplot()
    # BERGM_sampler70c_d.show_histogram(param_mark=[0.6376])
    # print("constraint(upper bound) netStat:", model_netStat_edgeonly(friendship_sequence[1]))
    # print("degenerated case edge num:", 25*24/2)
    # BERGM_sampler70c_d.show_latest_exchangeSampler_netStat_traceplot()


    
    # friendship23_y_plus, friendship23_y_minus = dissociate_network(friendship_sequence[2], friendship_sequence[3], True)

    # BERGM_sampler60c_f = BERGM(model_netStat_edgeonly, np.array([0]), friendship23_y_plus, is_formation=True, constraint_net=friendship_sequence[2])
    # BERGM_sampler60c_f.run(10000, 30, 0.01)
    # # BERGM_sampler60c_f.MC_formation_samples = BERGM_sampler20.MC_formation_samples[2000::2]
    # # BERGM_sampler60c_f.MC_dissolution_samples = BERGM_sampler20.MC_dissolution_samples[2000::2]
    # print("mean:", np.mean(BERGM_sampler60c_f.MC_sample_trace()[0]))
    # BERGM_sampler60c_f.show_traceplot()
    # BERGM_sampler60c_f.show_histogram(param_mark=[-2.2632])
    # print("constraint(lower bound) netStat:", model_netStat_edgeonly(friendship_sequence[2]))
    # print("degenerated case edge num:", 25*24/2)
    # BERGM_sampler60c_f.show_latest_exchangeSampler_netStat_traceplot()

    # BERGM_sampler70c_d = BERGM(model_netStat_edgeonly, np.array([0]), friendship23_y_minus, is_formation=False, constraint_net=friendship_sequence[2])
    # BERGM_sampler70c_d.run(10000, 30, 0.01)
    # # BERGM_sampler70c_d.MC_formation_samples = BERGM_sampler20.MC_formation_samples[2000::2]
    # # BERGM_sampler70c_d.MC_dissolution_samples = BERGM_sampler20.MC_dissolution_samples[2000::2]
    # print("mean:", np.mean(BERGM_sampler70c_d.MC_sample_trace()[0]))
    # BERGM_sampler70c_d.show_traceplot()
    # BERGM_sampler70c_d.show_histogram(param_mark=[0.2570])
    # print("constraint(upper bound) netStat:", model_netStat_edgeonly(friendship_sequence[2]))
    # print("degenerated case edge num:", 25*24/2)
    # BERGM_sampler70c_d.show_latest_exchangeSampler_netStat_traceplot()

    
    # samplk01_y_plus, samplk01_y_minus = dissociate_network(samplk_sequence[0], samplk_sequence[1], True)

    # BERGM_sampler80c_f = BERGM(model_netStat_edgeonly, np.array([0]), samplk01_y_plus, is_formation=True, constraint_net=samplk_sequence[0])
    # BERGM_sampler80c_f.run(10000, 30, 0.01)
    # # BERGM_sampler80c_f.MC_formation_samples = BERGM_sampler20.MC_formation_samples[2000::2]
    # # BERGM_sampler80c_f.MC_dissolution_samples = BERGM_sampler20.MC_dissolution_samples[2000::2]
    # print("mean:", np.mean(BERGM_sampler80c_f.MC_sample_trace()[0]))
    # BERGM_sampler80c_f.show_traceplot()
    # BERGM_sampler80c_f.show_histogram(param_mark=[-2.3427])
    # print("constraint(lower bound) netStat:", model_netStat_edgeonly(samplk_sequence[0]))
    # print("degenerated case edge num:", 18*17/2)
    # BERGM_sampler80c_f.show_latest_exchangeSampler_netStat_traceplot()

    # BERGM_sampler90c_d = BERGM(model_netStat_edgeonly, np.array([0]), samplk01_y_minus, is_formation=False, constraint_net=samplk_sequence[0])
    # BERGM_sampler90c_d.run(10000, 30, 0.01)
    # # BERGM_sampler90c_d.MC_formation_samples = BERGM_sampler20.MC_formation_samples[2000::2]
    # # BERGM_sampler90c_d.MC_dissolution_samples = BERGM_sampler20.MC_dissolution_samples[2000::2]
    # print("mean:", np.mean(BERGM_sampler90c_d.MC_sample_trace()[0]))
    # BERGM_sampler90c_d.show_traceplot()
    # BERGM_sampler90c_d.show_histogram(param_mark=[0.5596])
    # print("constraint(upper bound) netStat:", model_netStat_edgeonly(samplk_sequence[0]))
    # print("degenerated case edge num:", 18*17/2)
    # BERGM_sampler90c_d.show_latest_exchangeSampler_netStat_traceplot()

    
    # samplk12_y_plus, samplk12_y_minus = dissociate_network(samplk_sequence[1], samplk_sequence[2], True)

    # BERGM_sampler80c_f = BERGM(model_netStat_edgeonly, np.array([0]), samplk12_y_plus, is_formation=True, constraint_net=samplk_sequence[1])
    # BERGM_sampler80c_f.run(10000, 30, 0.01)
    # # BERGM_sampler80c_f.MC_formation_samples = BERGM_sampler20.MC_formation_samples[2000::2]
    # # BERGM_sampler80c_f.MC_dissolution_samples = BERGM_sampler20.MC_dissolution_samples[2000::2]
    # print("mean:", np.mean(BERGM_sampler80c_f.MC_sample_trace()[0]))
    # BERGM_sampler80c_f.show_traceplot()
    # BERGM_sampler80c_f.show_histogram(param_mark=[-2.6784])
    # print("constraint(lower bound) netStat:", model_netStat_edgeonly(samplk_sequence[1]))
    # print("degenerated case edge num:", 18*17/2)
    # BERGM_sampler80c_f.show_latest_exchangeSampler_netStat_traceplot()

    # BERGM_sampler90c_d = BERGM(model_netStat_edgeonly, np.array([0]), samplk12_y_minus, is_formation=False, constraint_net=samplk_sequence[1])
    # BERGM_sampler90c_d.run(10000, 30, 0.01)
    # # BERGM_sampler90c_d.MC_formation_samples = BERGM_sampler20.MC_formation_samples[2000::2]
    # # BERGM_sampler90c_d.MC_dissolution_samples = BERGM_sampler20.MC_dissolution_samples[2000::2]
    # print("mean:", np.mean(BERGM_sampler90c_d.MC_sample_trace()[0]))
    # BERGM_sampler90c_d.show_traceplot()
    # BERGM_sampler90c_d.show_histogram(param_mark=[0.8557])
    # print("constraint(upper bound) netStat:", model_netStat_edgeonly(samplk_sequence[1]))
    # print("degenerated case edge num:", 18*17/2)
    # BERGM_sampler90c_d.show_latest_exchangeSampler_netStat_traceplot()