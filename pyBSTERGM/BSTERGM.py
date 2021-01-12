import time


import numpy as np
import matplotlib.pyplot as plt

from network import UndirectedNetwork
from network_sampler import NetworkSampler

def model_netStat(network : UndirectedNetwork):
    model = []
    #define model
    model.append(network.statCal_edgeNum())
    model.append(network.statCal_geoWeightedESP(0.5))
    return np.array(model)


class BSTERGM:
    obs_network_seq = []
    initial_formation_param = np.array(0)
    initial_dissolution_param = np.array(0)
    MC_formation_samples = []
    MC_dissolution_samples = []
    node_num = 0
    
    random_seed = 2021
    random_gen = 0
    obs_network_formation_seq = []
    obs_network_dissolution_seq = []

    def __init__(self, model_fn, initial_formation_param, initial_dissolution_param, obs_network_seq, rng_seed=2021):
        self.obs_network_seq = obs_network_seq
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
        return (UndirectedNetwork(y_plus), UndirectedNetwork(y_minus))

    def dissociate_obsSeq(self):
        for i in range(1, len(self.obs_network_seq)):
            last_net = self.obs_network_seq[i-1]
            now_net = self.obs_network_seq[i]
            y_plus, y_minus = self.dissociate_network(last_net, now_net)
            self.obs_network_formation_seq.append(y_plus)
            self.obs_network_dissolution_seq.append(y_minus)

    def propose_param(self, last_param, cov_rate):
        cov_mat = np.identity(len(last_param))
        return self.random_gen.multivariate_normal(last_param, cov_mat)

    def get_exchange_sample(self, start_time_lag, exchange_iter, proposed_formation_param, proposed_dissolution_param, rng_seed):
        exchange_sampler = NetworkSampler(self.model, 
            proposed_formation_param, proposed_dissolution_param,
            self.obs_network_seq[start_time_lag], rng_seed)
        exchange_sampler.run(exchange_iter)
        return exchange_sampler.network_samples[-1]

    def log_prior(self, last_formation_param, last_dissolution_param,
            proposed_formation_param, proposed_dissolution_param):
        return 0

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

    def sampler(self, start_time_lag, dissolution_propose:bool, exchange_iter, rng_seed, proposal_cov_rate):
        last_formation_param = self.MC_formation_samples[-1]
        last_dissolution_param = self.MC_dissolution_samples[-1]
        
        #proposal
        proposed_formation_param = 0
        proposed_dissolution_param = 0
        if dissolution_propose:
            proposed_formation_param = last_formation_param
            proposed_dissolution_param = self.propose_param(last_dissolution_param, proposal_cov_rate)
        else:
            proposed_formation_param = self.propose_param(last_formation_param, proposal_cov_rate)
            proposed_dissolution_param = last_dissolution_param
        
        #exchange
        exchange_network = self.get_exchange_sample(start_time_lag, exchange_iter, proposed_formation_param, proposed_dissolution_param, rng_seed)
        exchange_formation, exchange_dissolution = self.dissociate_network(self.obs_network_seq[start_time_lag], exchange_network)

        #MCMC
        log_r_val = self.log_r(start_time_lag, last_formation_param, last_dissolution_param,
            proposed_formation_param, proposed_dissolution_param,
            exchange_formation, exchange_dissolution)

        unif_sample = self.random_gen.random()
        if np.log(unif_sample) < log_r_val:
            self.MC_formation_samples.append(proposed_formation_param)
            self.MC_dissolution_samples.append(proposed_dissolution_param)
        else:
            self.MC_formation_samples.append(last_formation_param)
            self.MC_dissolution_samples.append(last_dissolution_param)

    def run(self, iter, exchange_iter=30, proposal_cov_rate=0.01):
        start_time = time.time()
        for i in range(iter):
            start_time_lag = self.random_gen.integers(len(self.obs_network_seq)-1)
            dissolution_propose = i % 2
            rng_seed = self.random_seed + i
            self.sampler(start_time_lag, dissolution_propose, exchange_iter, rng_seed, proposal_cov_rate)
            if i%200==0:
                print("iter: ", i, "/", iter, " time elapsed(second):", round(time.time()-start_time,1))
        print(iter,"/",iter, "time elapsed(second):", round(time.time()-start_time,1))

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



if __name__=="__main__":
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

    test_obs_seq = [test_initnet1, test_initnet2, test_initnet3]
    
    initial_formation_param = np.array([0.1, 0.1])
    initial_dissolution_param = np.array([0.1, 0.1])
    test_BSTERGM_sampler = BSTERGM(model_netStat, initial_formation_param, initial_dissolution_param, test_obs_seq, 2021)
    test_BSTERGM_sampler.run(30000, exchange_iter=30)
    # print(test_BSTERGM_sampler.MC_formation_samples)
    # print(test_BSTERGM_sampler.MC_dissolution_samples)
    test_BSTERGM_sampler.show_traceplot()