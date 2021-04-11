import time
import csv
import math

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

from network import UndirectedNetwork, DirectedNetwork
from network_sampler import NetworkSampler, NetworkSampler_integrated

class BSTERGM:
    def __init__(self, model_fn, initial_formation_param, initial_dissolution_param, obs_network_seq, rng_seed=2021, pid=None):
        #variables
        self.obs_network_seq = []
        self.initial_formation_param = np.array(0)
        self.initial_dissolution_param = np.array(0)
        self.MC_formation_samples = []
        self.MC_dissolution_samples = []
        self.node_num = 0
        self.isDirected = False

        self.random_seed = 2021
        self.random_gen = 0
        self.obs_network_formation_seq = []
        self.obs_network_dissolution_seq = []

        self.latest_exchange_formation_sampler = None
        self.latest_exchange_dissolution_sampler = None
        self.latest_exchange_integrated_sampler = None
        self.pid = None

        
        #initialize
        self.obs_network_seq = obs_network_seq
        if isinstance(obs_network_seq[0], DirectedNetwork):
            self.isDirected = True
        else:
            self.isDirected = False
        
        self.initial_formation_param = initial_formation_param
        self.initial_dissolution_param = initial_dissolution_param
        self.MC_formation_samples.append(initial_formation_param)
        self.MC_dissolution_samples.append(initial_dissolution_param)
        self.obs_network_formation_seq.append(initial_formation_param)
        self.obs_network_dissolution_seq.append(initial_dissolution_param)
        self.dissociate_obsSeq()
        self.model = model_fn
        self.node_num = obs_network_seq[0].node_num
        
        self.random_seed = rng_seed
        self.random_gen = np.random.default_rng(seed=rng_seed)
        if pid is not None:
            self.pid = pid


    
    def __str__(self):
        string_val = "<BSTERGM.BSTERGM object>\n" + self.MC_formation_samples.__str__() +"\n" + self.MC_dissolution_samples.__str__()
        return string_val

    def dissociate_network(self, last_network, now_network):
        y_last_structure = last_network.structure
        y_now_structure = now_network.structure
        y_plus = y_last_structure.copy()
        y_minus = y_last_structure.copy()
        for row in range(self.node_num):
            for col in range(self.node_num):
                if y_now_structure[row,col]==1:
                    y_plus[row,col]=1
                if y_now_structure[row,col]==0:
                    y_minus[row,col]=0
        result = 0
        if self.isDirected:
            result = (DirectedNetwork(y_plus), DirectedNetwork(y_minus))
        else:
            result = (UndirectedNetwork(y_plus), UndirectedNetwork(y_minus))
        return result

    def dissociate_obsSeq(self):
        for i in range(1, len(self.obs_network_seq)):
            last_net = self.obs_network_seq[i-1]
            now_net = self.obs_network_seq[i]
            y_plus, y_minus = self.dissociate_network(last_net, now_net)
            self.obs_network_formation_seq.append(y_plus)
            self.obs_network_dissolution_seq.append(y_minus)

    def propose_param(self, last_param, cov_rate):
        cov_mat = np.diag(cov_rate)
        return self.random_gen.multivariate_normal(last_param, cov_mat)

    def get_exchange_sampler(self, start_time_lag, exchange_iter, proposed_param, is_formation, rng_seed):
        exchange_sampler = NetworkSampler(self.model, proposed_param,
            self.obs_network_seq[start_time_lag], is_formation=is_formation, rng_seed=rng_seed)
        exchange_sampler.run(exchange_iter)
        return exchange_sampler

    def get_integrated_exchange_sampler(self, start_time_lag, exchange_iter, proposed_formation_param, proposed_dissolution_param, rng_seed):
        exchange_sampler = NetworkSampler_integrated(self.model, proposed_formation_param, proposed_dissolution_param,
            self.obs_network_seq[start_time_lag], rng_seed=rng_seed)
        exchange_sampler.run(exchange_iter)
        return exchange_sampler

    def log_prior(self, last_formation_param, last_dissolution_param,
            proposed_formation_param, proposed_dissolution_param):
        #normal(0,100) prior

        dim = len(last_formation_param)
        zero_mean = [0 for _ in range(dim)]
        cov = np.identity(dim) * 100
        proposed_formation_prior = multivariate_normal.pdf(proposed_formation_param, zero_mean, cov)
        proposed_dissolution_prior = multivariate_normal.pdf(proposed_dissolution_param, zero_mean, cov)
        if proposed_formation_prior == 0 or proposed_dissolution_prior == 0:
            raise ZeroDivisionError("divide by zero encountered in log in calculating log_prior")

        last_formation_prior = multivariate_normal.pdf(last_formation_param, zero_mean, cov)
        last_dissolution_prior = multivariate_normal.pdf(last_dissolution_param, zero_mean, cov)
        log_prior_val = 0
        log_prior_val += np.log(proposed_formation_prior) #underflow warning (it is why there are above 'if' sentences)
        log_prior_val += np.log(proposed_dissolution_prior) #underflow warning
        log_prior_val -= np.log(last_formation_prior)
        log_prior_val -= np.log(last_dissolution_prior)
        return log_prior_val

    def log_r(self, start_time_lag, last_formation_param, last_dissolution_param,
            proposed_formation_param, proposed_dissolution_param,
            exchange_formation, exchange_dissolution):
        formation_netStat_diff = self.model(self.obs_network_formation_seq[start_time_lag+1]) - self.model(exchange_formation)
        dissolution_netStat_diff = self.model(self.obs_network_dissolution_seq[start_time_lag+1]) - self.model(exchange_dissolution)

        log_r_val = np.dot(proposed_formation_param - last_formation_param, formation_netStat_diff)
        log_r_val += np.dot(proposed_dissolution_param - last_dissolution_param, dissolution_netStat_diff)
        log_r_val += self.log_prior(last_formation_param, last_dissolution_param,
            proposed_formation_param, proposed_dissolution_param)        
        return log_r_val

    def sampler(self, start_time_lag, exchange_iter, rng_seed):
        last_formation_param = self.MC_formation_samples[-1]
        last_dissolution_param = self.MC_dissolution_samples[-1]
        
        #proposal
        proposed_formation_param = 0
        proposed_dissolution_param = 0
        
        proposed_dissolution_param = self.propose_param(last_dissolution_param, self.dissolution_cov_rate)
        proposed_formation_param = self.propose_param(last_formation_param, self.formation_cov_rate)
        
        #exchange
        # separately generated case
        self.latest_exchange_formation_sampler = self.get_exchange_sampler(start_time_lag, exchange_iter, proposed_formation_param, is_formation=True, rng_seed=rng_seed)
        exchange_formation_sample = self.latest_exchange_formation_sampler.network_samples[-1]
        self.latest_exchange_dissolution_sampler = self.get_exchange_sampler(start_time_lag, exchange_iter, proposed_dissolution_param, is_formation=False, rng_seed=rng_seed*10)
        exchange_dissolution_sample = self.latest_exchange_dissolution_sampler.network_samples[-1]

        #intergratedly generated case
        # self.latest_exchange_integrated_sampler = self.get_integrated_exchange_sampler(start_time_lag, exchange_iter, 
        #     proposed_formation_param, proposed_dissolution_param, rng_seed=rng_seed)
        # exchange_integrated_sample = self.latest_exchange_integrated_sampler.network_samples[-1]
        # exchange_formation_sample, exchange_dissolution_sample = self.dissociate_network(self.obs_network_seq[start_time_lag], exchange_integrated_sample)

        #MCMC
        try:
            log_r_val = self.log_r(start_time_lag, last_formation_param, last_dissolution_param,
                proposed_formation_param, proposed_dissolution_param,
                exchange_formation_sample, exchange_dissolution_sample)
        except ZeroDivisionError:
            log_r_val = -math.inf #manual reject

        unif_sample = self.random_gen.random()
        if np.log(unif_sample) < log_r_val:
            self.MC_formation_samples.append(proposed_formation_param)
            self.MC_dissolution_samples.append(proposed_dissolution_param)
        else:
            self.MC_formation_samples.append(last_formation_param)
            self.MC_dissolution_samples.append(last_dissolution_param)

    def proposal_cov_rate_setting(self, proposal_cov_rate):
        # float or
        # dict structured by {"formation_cov_rate": [0,...], "dissolution_cov_rate":[0,...]}
        if isinstance(proposal_cov_rate, float):
            formation_dim = len(self.MC_formation_samples[-1])
            dissolution_dim = len(self.MC_dissolution_samples[-1])
            self.formation_cov_rate = [proposal_cov_rate for _ in range(formation_dim)]
            self.dissolution_cov_rate = [proposal_cov_rate for _ in range(dissolution_dim)]
        elif isinstance(proposal_cov_rate, dict):
            self.formation_cov_rate = proposal_cov_rate["formation_cov_rate"]
            self.dissolution_cov_rate = proposal_cov_rate["disolution_cov_rate"]

    def run(self, iter, exchange_iter=30, proposal_cov_rate=0.01):
        # proposal_cov_rate: float or
        #   dict structured by {"formation_cov_rate": [0,...], "dissolution_cov_rate":[0,...]}
        start_time = time.time()
        self.proposal_cov_rate_setting(proposal_cov_rate)
        for i in range(iter):
            start_time_lag = self.random_gen.integers(len(self.obs_network_seq)-1)
            rng_seed = self.random_seed + i
            self.sampler(start_time_lag, exchange_iter, rng_seed)
            if i%200==0:
                if self.pid is not None:
                    print("pid:",self.pid, " iter: ", i, "/", iter, " time elapsed(second):", round(time.time()-start_time,1))
                else:
                    print("iter: ", i, "/", iter, " time elapsed(second):", round(time.time()-start_time,1))
                
        if self.pid is not None:
            print("pid:",self.pid," iter:", iter,"/",iter, "time elapsed(second):", round(time.time()-start_time,1))
        else:
            print(iter,"/",iter, "time elapsed(second):", round(time.time()-start_time,1))

    #=============================================================================================

    def propose_param_1dim(self, dim_idx, last_param, cov_rate):
        result =[val for val in last_param]
        new_val = self.random_gen.normal(last_param[dim_idx], cov_rate)
        result[dim_idx] = new_val
        return np.array(result)
    
    def log_r_1dim_formation(self, start_time_lag, last_formation_param, last_dissolution_param,
            proposed_formation_param, proposed_dissolution_param,
            exchange_formation):
        formation_netStat_diff = self.model(self.obs_network_formation_seq[start_time_lag+1]) - self.model(exchange_formation)

        log_r_val = np.dot(proposed_formation_param - last_formation_param, formation_netStat_diff)
        log_r_val += self.log_prior(last_formation_param, last_dissolution_param,
            proposed_formation_param, proposed_dissolution_param)        
        return log_r_val

    def log_r_1dim_dissolution(self, start_time_lag, last_formation_param, last_dissolution_param,
            proposed_formation_param, proposed_dissolution_param,
            exchange_dissolution):
        dissolution_netStat_diff = self.model(self.obs_network_dissolution_seq[start_time_lag+1]) - self.model(exchange_dissolution)

        log_r_val = np.dot(proposed_dissolution_param - last_dissolution_param, dissolution_netStat_diff)
        log_r_val += self.log_prior(last_formation_param, last_dissolution_param,
            proposed_formation_param, proposed_dissolution_param)        
        return log_r_val

    def sampler_1dim(self, start_time_lag, exchange_iter, rng_seed):
        last_formation_param = self.MC_formation_samples[-1]
        last_dissolution_param = self.MC_dissolution_samples[-1]
        
        now_formation_param = np.array([val for val in last_formation_param])
        now_dissolution_param = np.array([val for val in last_dissolution_param])

        for i_idx in range(len(last_formation_param)):
            #proposal
            proposal_cov_rate = self.formation_cov_rate[i_idx]
            proposed_formation_param = self.propose_param_1dim(i_idx, now_formation_param, proposal_cov_rate)
            proposed_dissolution_param = now_dissolution_param

            #exchange
            self.latest_exchange_formation_sampler = self.get_exchange_sampler(start_time_lag, exchange_iter, proposed_formation_param, is_formation=True, rng_seed=rng_seed)
            exchange_formation_sample = self.latest_exchange_formation_sampler.network_samples[-1]

            #MCMC
            log_r_val = self.log_r_1dim_formation(start_time_lag, now_formation_param, now_dissolution_param,
                proposed_formation_param, proposed_dissolution_param,
                exchange_formation_sample)

            unif_sample = self.random_gen.random()
            if np.log(unif_sample) < log_r_val:
                now_formation_param = proposed_formation_param
            else:
                pass

        for i_idx in range(len(last_dissolution_param)):
            #proposal
            proposal_cov_rate = self.dissolution_cov_rate[i_idx]
            proposed_formation_param = now_formation_param
            proposed_dissolution_param = self.propose_param_1dim(i_idx, now_dissolution_param, proposal_cov_rate)

            #exchange
            self.latest_exchange_dissolution_sampler = self.get_exchange_sampler(start_time_lag, exchange_iter, proposed_dissolution_param, is_formation=False, rng_seed=rng_seed*10)
            exchange_dissolution_sample = self.latest_exchange_dissolution_sampler.network_samples[-1]

            #MCMC
            log_r_val = self.log_r_1dim_dissolution(start_time_lag, now_formation_param, now_dissolution_param,
                proposed_formation_param, proposed_dissolution_param,
                exchange_dissolution_sample)

            unif_sample = self.random_gen.random()
            if np.log(unif_sample) < log_r_val:
                now_dissolution_param = proposed_dissolution_param
            else:
                pass

        self.MC_formation_samples.append(now_formation_param)
        self.MC_dissolution_samples.append(now_dissolution_param)


    def run_1dim(self, iter, exchange_iter=30, proposal_cov_rate=0.1):
        # proposal_cov_rate: float or
        #   dict structured by {"formation_cov_rate": [0,...], "dissolution_cov_rate":[0,...]}
        start_time = time.time()
        self.proposal_cov_rate_setting(proposal_cov_rate)
        for i in range(iter):
            start_time_lag = self.random_gen.integers(len(self.obs_network_seq)-1)
            rng_seed = self.random_seed + i
            self.sampler_1dim(start_time_lag, exchange_iter, rng_seed)
            if i%50==0:
                if self.pid is not None:
                    print("pid:",self.pid, " iter: ", i, "/", iter, " time elapsed(second):", round(time.time()-start_time,1))
                else:
                    print("iter: ", i, "/", iter, " time elapsed(second):", round(time.time()-start_time,1))
                
        if self.pid is not None:
            print("pid:",self.pid," iter:", iter,"/",iter, "time elapsed(second):", round(time.time()-start_time,1))
        else:
            print(iter,"/",iter, "time elapsed(second):", round(time.time()-start_time,1))


    #=============================================================================================

    def sampler_2dim(self, start_time_lag, exchange_iter, rng_seed):
        last_formation_param = self.MC_formation_samples[-1]
        last_dissolution_param = self.MC_dissolution_samples[-1]
        if len(last_formation_param) != len(last_dissolution_param):
            raise ValueError('dimension mismatch: formation and dissolution')

        now_formation_param = np.array([val for val in last_formation_param])
        now_dissolution_param = np.array([val for val in last_dissolution_param])

        for i_idx in range(len(last_formation_param)):
            #proposal
            proposal_formation_cov_rate = self.formation_cov_rate[i_idx]
            proposal_dissolution_cov_rate = self.dissolution_cov_rate[i_idx]
            proposed_formation_param = self.propose_param_1dim(i_idx, now_formation_param, proposal_formation_cov_rate)
            proposed_dissolution_param = self.propose_param_1dim(i_idx, now_dissolution_param, proposal_dissolution_cov_rate)

            #exchange
            #separately generated case
            self.latest_exchange_formation_sampler = self.get_exchange_sampler(start_time_lag, exchange_iter, proposed_formation_param, is_formation=True, rng_seed=rng_seed)
            self.latest_exchange_dissolution_sampler = self.get_exchange_sampler(start_time_lag, exchange_iter, proposed_dissolution_param, is_formation=False, rng_seed=rng_seed*10)
            exchange_formation_sample = self.latest_exchange_formation_sampler.network_samples[-1]
            exchange_dissolution_sample = self.latest_exchange_dissolution_sampler.network_samples[-1]

            #intergratedly generated case
            # self.latest_exchange_integrated_sampler = self.get_integrated_exchange_sampler(start_time_lag, exchange_iter, 
            #     proposed_formation_param, proposed_dissolution_param, rng_seed=rng_seed)
            # exchange_integrated_sample = self.latest_exchange_integrated_sampler.network_samples[-1]
            # exchange_formation_sample, exchange_dissolution_sample = self.dissociate_network(self.obs_network_seq[start_time_lag], exchange_integrated_sample)

            try:
                log_r_val = self.log_r(start_time_lag, now_formation_param, now_dissolution_param,
                    proposed_formation_param, proposed_dissolution_param,
                    exchange_formation_sample, exchange_dissolution_sample)
            except ZeroDivisionError:
                log_r_val = -math.inf #manual reject

            unif_sample = self.random_gen.random()
            if np.log(unif_sample) < log_r_val:
                now_formation_param = proposed_formation_param
                now_dissolution_param = proposed_dissolution_param
            else:
                pass
        
        self.MC_formation_samples.append(now_formation_param)
        self.MC_dissolution_samples.append(now_dissolution_param)

    def run_2dim(self, iter, exchange_iter=30, proposal_cov_rate=0.1):
        # proposal_cov_rate: float or
        #   dict structured by {"formation_cov_rate": [0,...], "dissolution_cov_rate":[0,...]}
        start_time = time.time()
        self.proposal_cov_rate_setting(proposal_cov_rate)
        for i in range(iter):
            start_time_lag = self.random_gen.integers(len(self.obs_network_seq)-1)
            rng_seed = self.random_seed + i
            self.sampler_2dim(start_time_lag, exchange_iter, rng_seed)
            if i%50==0:
                if self.pid is not None:
                    print("pid:",self.pid, " iter: ", i, "/", iter, " time elapsed(second):", round(time.time()-start_time,1))
                else:
                    print("iter: ", i, "/", iter, " time elapsed(second):", round(time.time()-start_time,1))
                
        if self.pid is not None:
            print("pid:",self.pid," iter:", iter,"/",iter, "time elapsed(second):", round(time.time()-start_time,1))
        else:
            print(iter,"/",iter, "time elapsed(second):", round(time.time()-start_time,1))


    #=============================================================================================

    def sampler_by_group(self, start_time_lag, exchange_iter, rng_seed):
        last_formation_param = self.MC_formation_samples[-1]
        last_dissolution_param = self.MC_dissolution_samples[-1]
        
        now_formation_param = np.array([val for val in last_formation_param])
        now_dissolution_param = np.array([val for val in last_dissolution_param])

        #step1. formation group proposal
        proposed_formation_param = 0
        proposed_dissolution_param = 0
        
        proposed_formation_param = self.propose_param(last_formation_param, self.formation_cov_rate)
        proposed_dissolution_param = now_dissolution_param
        
        #exchange
        # separately generated case
        self.latest_exchange_formation_sampler = self.get_exchange_sampler(start_time_lag, exchange_iter, proposed_formation_param, is_formation=True, rng_seed=rng_seed)
        exchange_formation_sample = self.latest_exchange_formation_sampler.network_samples[-1]
        self.latest_exchange_dissolution_sampler = self.get_exchange_sampler(start_time_lag, exchange_iter, proposed_dissolution_param, is_formation=False, rng_seed=rng_seed*10)
        exchange_dissolution_sample = self.latest_exchange_dissolution_sampler.network_samples[-1]

        #intergratedly generated case
        # self.latest_exchange_integrated_sampler = self.get_integrated_exchange_sampler(start_time_lag, exchange_iter, 
        #     proposed_formation_param, proposed_dissolution_param, rng_seed=rng_seed)
        # exchange_integrated_sample = self.latest_exchange_integrated_sampler.network_samples[-1]
        # exchange_formation_sample, exchange_dissolution_sample = self.dissociate_network(self.obs_network_seq[start_time_lag], exchange_integrated_sample)

        #MCMC
        try:
            log_r_val = self.log_r(start_time_lag, last_formation_param, last_dissolution_param,
                proposed_formation_param, proposed_dissolution_param,
                exchange_formation_sample, exchange_dissolution_sample)
        except ZeroDivisionError:
            log_r_val = -math.inf #manual reject

        unif_sample = self.random_gen.random()
        if np.log(unif_sample) < log_r_val:
                now_formation_param = proposed_formation_param
        else:
            pass

        #step2. dissolution proposal
        proposed_formation_param = now_formation_param
        proposed_dissolution_param = self.propose_param(last_dissolution_param, self.dissolution_cov_rate)
        
        #exchange
        # separately generated case
        self.latest_exchange_formation_sampler = self.get_exchange_sampler(start_time_lag, exchange_iter, proposed_formation_param, is_formation=True, rng_seed=rng_seed)
        exchange_formation_sample = self.latest_exchange_formation_sampler.network_samples[-1]
        self.latest_exchange_dissolution_sampler = self.get_exchange_sampler(start_time_lag, exchange_iter, proposed_dissolution_param, is_formation=False, rng_seed=rng_seed*10)
        exchange_dissolution_sample = self.latest_exchange_dissolution_sampler.network_samples[-1]

        #intergratedly generated case
        # self.latest_exchange_integrated_sampler = self.get_integrated_exchange_sampler(start_time_lag, exchange_iter, 
        #     proposed_formation_param, proposed_dissolution_param, rng_seed=rng_seed)
        # exchange_integrated_sample = self.latest_exchange_integrated_sampler.network_samples[-1]
        # exchange_formation_sample, exchange_dissolution_sample = self.dissociate_network(self.obs_network_seq[start_time_lag], exchange_integrated_sample)

        #MCMC
        try:
            log_r_val = self.log_r(start_time_lag, last_formation_param, last_dissolution_param,
                proposed_formation_param, proposed_dissolution_param,
                exchange_formation_sample, exchange_dissolution_sample)
        except ZeroDivisionError:
            log_r_val = -math.inf #manual reject

        unif_sample = self.random_gen.random()
        if np.log(unif_sample) < log_r_val:
                now_formation_param = proposed_formation_param
                now_dissolution_param = proposed_dissolution_param
        else:
            pass

        #step 3
        self.MC_formation_samples.append(now_formation_param)
        self.MC_dissolution_samples.append(now_dissolution_param)
    


    def run_by_group(self, iter, exchange_iter=30, proposal_cov_rate=0.1):
        start_time = time.time()
        self.proposal_cov_rate_setting(proposal_cov_rate)
        for i in range(iter):
            start_time_lag = self.random_gen.integers(len(self.obs_network_seq)-1)
            rng_seed = self.random_seed + i
            self.sampler_by_group(start_time_lag, exchange_iter, rng_seed)
            if i%50==0:
                if self.pid is not None:
                    print("pid:",self.pid, " iter: ", i, "/", iter, " time elapsed(second):", round(time.time()-start_time,1))
                else:
                    print("iter: ", i, "/", iter, " time elapsed(second):", round(time.time()-start_time,1))
                
        if self.pid is not None:
            print("pid:",self.pid," iter:", iter,"/",iter, "time elapsed(second):", round(time.time()-start_time,1))
        else:
            print(iter,"/",iter, "time elapsed(second):", round(time.time()-start_time,1))

    

    #=============================================================================================



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

    def show_latest_exchangeSampler_netStat_traceplot(self, show=True):
        if self.latest_exchange_integrated_sampler is None:
            self.latest_exchange_formation_sampler.show_traceplot()
            self.latest_exchange_dissolution_sampler.show_traceplot()
        else:
            self.latest_exchange_integrated_sampler.show_traceplot()
        
    
    def write_posterior_samples(self, filename: str):
        # print(self.MC_dissolution_samples)
        with open("pyBSTERGM/" + filename + '.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for sample_formation, sample_dissolution in zip(self.MC_formation_samples, self.MC_dissolution_samples):
                csv_row = sample_formation.tolist() + sample_dissolution.tolist()
                writer.writerow(csv_row)

    def write_latest_exchangeSampler_netStat(self, filename: str):
        if self.latest_exchange_integrated_sampler is None:
            formation_netStat = self.latest_exchange_formation_sampler.netStat_trace()
            dissolution_netStat = self.latest_exchange_dissolution_sampler.netStat_trace()
            netStat_list = (np.array(formation_netStat + dissolution_netStat).T).tolist()
        else:
            intergrated_netStat = self.latest_exchange_integrated_sampler.netStat_trace()
            netStat_list = (np.array(intergrated_netStat).T).tolist()

        with open("pyBSTERGM/" + filename + '.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for csv_row in netStat_list:
                writer.writerow(csv_row)


if __name__=="__main__":
    
    def model_netStat(network):
        model = []
        #define model
        model.append(network.statCal_edgeNum())
        model.append(network.statCal_geoWeightedESP(0.25))
        return np.array(model)


    test_structure1 = np.array(
    [
        [0,1,1,0,0,0,0,0,0,0],
        [1,0,1,1,0,0,0,0,0,0],
        [1,1,0,1,0,0,0,0,0,0],
        [0,1,1,0,1,0,0,0,0,0],
        [0,0,0,1,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0]
    ]
    )
    test_structure2 = np.array(
    [
        [0,1,1,0,0,0,0,0,0,0],
        [1,0,1,1,0,0,0,0,0,0],
        [1,1,0,1,0,0,0,0,0,0],
        [0,1,1,0,1,1,0,0,0,0],
        [0,0,0,1,0,0,0,0,0,0],
        [0,0,0,1,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,1,1,0],
        [0,0,0,0,0,0,1,0,0,1],
        [0,0,0,0,0,0,1,0,0,1],
        [0,0,0,0,0,0,0,1,1,0]
    ]
    )
    test_structure3 = np.array(
    [
        [0,0,1,0,0,0,0,0,1,1],
        [0,0,1,1,0,0,0,0,1,0],
        [1,1,0,0,0,0,0,1,0,0],
        [0,1,0,0,1,1,0,0,0,0],
        [0,0,0,1,0,0,0,0,0,0],
        [0,0,0,1,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,1,1,0],
        [0,0,1,0,0,0,1,0,0,1],
        [1,1,0,0,0,0,1,0,0,1],
        [1,0,0,0,0,0,0,1,1,0]
    ]
    )

    test_initnet1 = UndirectedNetwork(test_structure1)
    test_initnet2 = UndirectedNetwork(test_structure2)
    test_initnet3 = UndirectedNetwork(test_structure3)

    # print(model_netStat(test_initnet1))
    # print(model_netStat(test_initnet2))
    # print(model_netStat(test_initnet3))


    test_obs_seq = [test_initnet1, test_initnet2, test_initnet3]
    
    initial_formation_param = np.array([-1.5, -0.3])
    initial_dissolution_param = np.array([2.5, -0.4])
    test_BSTERGM_sampler = BSTERGM(model_netStat, initial_formation_param, initial_dissolution_param, test_obs_seq, 2021)

    test_BSTERGM_sampler.run_by_group(100, exchange_iter=50)
    print(test_BSTERGM_sampler.MC_formation_samples)
    print(test_BSTERGM_sampler.MC_dissolution_samples)
    test_BSTERGM_sampler.show_traceplot()
    test_BSTERGM_sampler.show_latest_exchangeSampler_netStat_traceplot()

    # test_BSTERGM_sampler.write_posterior_samples("test")
    # test_BSTERGM_sampler.write_latest_exchangeSampler_netStat("test_netStat")



    # test_BSTERGM_sampler.run_1dim(1500, exchange_iter=100)
    # test_BSTERGM_sampler.show_traceplot()
    # test_BSTERGM_sampler.show_latest_exchangeSampler_netStat_traceplot()

    # print(test_BSTERGM_sampler.propose_param_1dim(0, [0,0], 1))
    # print(test_BSTERGM_sampler.propose_param_1dim(1, [0,0], 1))
    # print(test_BSTERGM_sampler.propose_param_1dim(0, [0,0], 1))


    # test_BSTERGM_sampler.run_2dim(2000, exchange_iter=100)
    # test_BSTERGM_sampler.show_traceplot()
    # test_BSTERGM_sampler.show_latest_exchangeSampler_netStat_traceplot()
    # freq edgeonly -1.962, 2.0149
    # freq edge+gwesp(0.25) -1.6782 -0.2836 // 2.7845, -0.4699