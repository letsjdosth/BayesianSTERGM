import time
import csv
import math

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.linalg import block_diag

from network import UndirectedNetwork, DirectedNetwork
from network_sampler import NetworkSampler
from BERGM import BERGM

class BSTERGM:
    def __init__(self, model_fn, initial_formation_param, initial_dissolution_param, obs_network_seq, rng_seed=2021, pid=None):
        # observed networks
        self.obs_network_seq = obs_network_seq
        self.node_num = obs_network_seq[0].node_num
        self.isDirected = False
        if isinstance(obs_network_seq[0], DirectedNetwork):
            self.isDirected = True
        else:
            self.isDirected = False
        
        self.obs_network_formation_seq = [0] # dummy for 0-th index. 0->1 : use [1]
        self.obs_network_dissolution_seq = [0] # dummy for 0-th index
        self.dissociate_obsSeq()

        # parameter containers
        self.initial_formation_param = initial_formation_param #np array
        self.initial_dissolution_param = initial_dissolution_param  #np array
        # self.MC_formation_samples = [initial_formation_param]
        # self.MC_dissolution_samples = [initial_dissolution_param]
        
        # exchange sampler references
        # self.latest_exchange_formation_sampler = None
        # self.latest_exchange_dissolution_sampler = None

        # other settings        
        self.model = model_fn
        
        self.random_seed = rng_seed
        self.random_gen = np.random.default_rng(seed=rng_seed)
        
        self.pid = None
        if pid is not None:
            self.pid = pid

    def __str__(self):
        string_val = "<BSTERGM.BSTERGM object>\n"
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

    # def log_r(self, start_time_lag, last_formation_param, last_dissolution_param,
    #         proposed_formation_param, proposed_dissolution_param,
    #         exchange_formation, exchange_dissolution):
    #     formation_netStat_diff = self.model(self.obs_network_formation_seq[start_time_lag+1]) - self.model(exchange_formation)
    #     dissolution_netStat_diff = self.model(self.obs_network_dissolution_seq[start_time_lag+1]) - self.model(exchange_dissolution)

    #     log_r_val = np.dot(proposed_formation_param - last_formation_param, formation_netStat_diff)
    #     log_r_val += np.dot(proposed_dissolution_param - last_dissolution_param, dissolution_netStat_diff)
    #     log_r_val += self.log_prior(last_formation_param, last_dissolution_param,
    #         proposed_formation_param, proposed_dissolution_param)        
    #     return log_r_val

    def proposal_cov_rate_setting(self, proposal_cov_rate):
        # float or
        # dict structured by {"formation_cov_rate": [0,...], "dissolution_cov_rate":[0,...]}
        if isinstance(proposal_cov_rate, float):
            formation_dim = len(self.initial_formation_param)
            dissolution_dim = len(self.initial_dissolution_param)
            formation_cov_rate = [proposal_cov_rate for _ in range(formation_dim)]
            dissolution_cov_rate = [proposal_cov_rate for _ in range(dissolution_dim)]
        elif isinstance(proposal_cov_rate, dict):
            formation_cov_rate = proposal_cov_rate["formation_cov_rate"]
            dissolution_cov_rate = proposal_cov_rate["disolution_cov_rate"]
        return formation_cov_rate, dissolution_cov_rate

    
    def make_block_diag_net(self, network_list):
        num_block = len(network_list)
        block_mat = network_list[0].structure
        for i in range(1, num_block):
            block_mat = block_diag(block_mat, network_list[i].structure)
        if self.isDirected:
            return DirectedNetwork(block_mat)
        else:
            return UndirectedNetwork(block_mat)


    def set_bergm_objects(self, time_lag):
        self.formation_BERGM = BERGM(self.model, 
            self.initial_formation_param, self.obs_network_formation_seq[time_lag+1],
            rng_seed=self.random_seed+1,
            is_formation = True, constraint_net=self.obs_network_seq[time_lag], 
            pid=self.pid)
        self.dissolution_BERGM = BERGM(self.model, 
            self.initial_dissolution_param, self.obs_network_dissolution_seq[time_lag+1],
            rng_seed=self.random_seed+2,
            is_formation = False, constraint_net=self.obs_network_seq[time_lag], 
            pid=self.pid)


    def set_bergm_objects_jointly(self):
        joint_constraint = self.make_block_diag_net(self.obs_network_seq[:-1])
        joint_obs_formation = self.make_block_diag_net(self.obs_network_formation_seq[1:])
        joint_obs_dissolution = self.make_block_diag_net(self.obs_network_dissolution_seq[1:])
        # print(joint_constraint.structure.shape, joint_obs_formation.structure.shape, joint_obs_dissolution.structure.shape)
        
        num_block = len(self.obs_network_formation_seq[1:])
        # print("num_block:", num_block)

        self.formation_BERGM = BERGM(self.model, 
            self.initial_formation_param, joint_obs_formation,
            rng_seed=self.random_seed+1,
            is_formation = True, constraint_net=joint_constraint, 
            num_joint_blocks= num_block,
            pid=self.pid)
        self.dissolution_BERGM = BERGM(self.model, 
            self.initial_dissolution_param, joint_obs_dissolution,
            rng_seed=self.random_seed+2,
            is_formation = False, constraint_net=joint_constraint, 
            num_joint_blocks= num_block,
            pid=self.pid)


    def run(self, iter, exchange_iter, time_lag, proposal_cov_rate=0.01, console_string=''):
        '''
        time_lag: int or 'joint'
            For example, if 0, then BSTERGM estimates coefficients using network[0] and network[1]
            If 'joint', then BSTERGM fits coefficient using extended block-structured networks
        proposal_cov_rate: float or dict 
            For example, dict should be structured by {"formation_cov_rate": [0,...], "dissolution_cov_rate":[0,...]}
            If input is float, called x, then the covariance would be set to {"formation_cov_rate": [x,...,x], "dissolution_cov_rate":[x,...,x]}
        '''

        if isinstance(time_lag, int):
            self.set_bergm_objects(time_lag)
        elif time_lag == 'joint':
            self.set_bergm_objects_jointly()
        else:
            TypeError('choose time_lag. (int or str:"joint") For example, if 0, then BSTERGM estimates coefficients using network[0] and network[1]')
        
        formation_cov_rate, dissolution_cov_rate = self.proposal_cov_rate_setting(proposal_cov_rate)

        start_time = time.time()
        self.formation_BERGM.run(iter, exchange_iter, formation_cov_rate, console_output_str=console_string+' formation')
        self.dissolution_BERGM.run(iter, exchange_iter, dissolution_cov_rate, console_output_str=console_string+' dissolution')
        print("BSTERGM complete: time elapsed(second): ", round(time.time()-start_time,1))

    def get_bergm_objects_with_setting(self, time_lag):
        '''
        time_lag: int or 'joint'
            For example, if 0, then BSTERGM estimates coefficients using network[0] and network[1]
            If 'joint', then BSTERGM fits coefficient using extended block-structured networks
        '''
        if isinstance(time_lag, int):
            self.set_bergm_objects(time_lag)
        elif time_lag == 'joint':
            self.set_bergm_objects_jointly()
        else:
            TypeError('choose time_lag. (int) For example, if 0, then BSTERGM estimates coefficients using network[0] and network[1]')
        return self.formation_BERGM, self.dissolution_BERGM

    #=============================================================================================

    def show_traceplot(self, show=True):
        grid_column = max(len(self.initial_formation_param), len(self.initial_dissolution_param))
        grid_row = 2
        plt.figure(figsize=(5*grid_column, 3*grid_row))
        for i in range(len(self.initial_formation_param)):
            plt.subplot(grid_row, grid_column, i+1)
            self.formation_BERGM.show_traceplot_eachaxis(i, False)
            plt.ylabel('formation'+str(i))
        for i in range(len(self.initial_dissolution_param)):
            plt.subplot(grid_row, grid_column, len(self.initial_formation_param)+i+1)
            self.dissolution_BERGM.show_traceplot_eachaxis(i, False)
            plt.ylabel('dissolution'+str(i))
        if show:
            plt.show()

    def show_latest_exchangeSampler_netStat_traceplot(self, show=True):
        self.formation_BERGM.show_latest_exchangeSampler_netStat_traceplot(show)
        self.dissolution_BERGM.show_latest_exchangeSampler_netStat_traceplot(show)

    def show_histogram(self, show=True, formation_param_mark_vec=None, dissolution_param_mark_vec=None):
        grid_column = max(len(self.initial_formation_param), len(self.initial_dissolution_param))
        grid_row = 2
        plt.figure(figsize=(5*grid_column, 3*grid_row))
        for i in range(len(self.initial_formation_param)):
            plt.subplot(grid_row, grid_column, i+1)
            if formation_param_mark_vec is None:
                self.formation_BERGM.show_histogram_eachaxis(i, show=False)
            else:
                self.formation_BERGM.show_histogram_eachaxis(i, param_mark=formation_param_mark_vec[i])
            plt.ylabel('formation'+str(i))
        for i in range(len(self.initial_dissolution_param)):
            plt.subplot(grid_row, grid_column, len(self.initial_formation_param)+i+1)
            if formation_param_mark_vec is None:
                self.dissolution_BERGM.show_histogram_eachaxis(i, show=False)
            else:
                self.dissolution_BERGM.show_histogram_eachaxis(i, param_mark=dissolution_param_mark_vec[i])
            plt.ylabel('dissolution'+str(i))
        if show:
            plt.show()

    def write_posterior_samples(self, filename: str):
        MC_formation_samples = self.formation_BERGM.MC_sample
        MC_dissolution_samples = self.dissolution_BERGM.MC_sample
        with open("pyBSTERGM/" + filename + '.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for sample_formation, sample_dissolution in zip(MC_formation_samples, MC_dissolution_samples):
                csv_row = sample_formation.tolist() + sample_dissolution.tolist()
                writer.writerow(csv_row)

    def write_latest_exchangeSampler_netStat(self, filename: str):
        formation_netStat = self.formation_BERGM.latest_exchange_sampler.network_samples_netStats
        dissolution_netStat = self.dissolution_BERGM.latest_exchange_sampler.network_samples_netStats

        with open("pyBSTERGM/" + filename + '.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for vec_f, vec_d in zip(formation_netStat, dissolution_netStat):
                csv_row = vec_f.tolist() + vec_d.tolist()
                writer.writerow(csv_row)


if __name__=="__main__":
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

    from model_settings import model_netStat_samplk_vignettesEx
    
    # BSTERGM_sampler0 = BSTERGM(model_netStat_edgeonly, np.array([0]), np.array([0]), samplk_sequence, pid='000')
    # BSTERGM_sampler0.run(5000, 50, time_lag='joint')
    # # BSTERGM_sampler0.write_posterior_samples("edgeonly_samplk_jointlag")
    # # BSTERGM_sampler0.write_latest_exchangeSampler_netStat("edgeonly_samplk_jointlag_netStat")
    # BSTERGM_sampler0.show_traceplot()
    # BSTERGM_sampler0.show_histogram(formation_param_mark_vec=[-2.4980], dissolution_param_mark_vec=[0.7066])
    # BSTERGM_sampler0.show_latest_exchangeSampler_netStat_traceplot()
    

    # BSTERGM_sampler1 = BSTERGM(model_netStat_edgeonly, np.array([0]), np.array([0]), samplk_sequence, pid='000')
    # BSTERGM_sampler1.run(5000, exchange_iter=50, time_lag=1)
    # # BSTERGM_sampler1.write_posterior_samples("edgeonly_samplk_time12")
    # # BSTERGM_sampler1.write_latest_exchangeSampler_netStat("edgeonly_samplk_time12_netStat")
    # BSTERGM_sampler1.show_traceplot()
    # BSTERGM_sampler1.show_histogram(formation_param_mark_vec=[-2.6784], dissolution_param_mark_vec=[0.8557])
    # BSTERGM_sampler1.show_latest_exchangeSampler_netStat_traceplot()

    # BSTERGM_sampler2 = BSTERGM(model_netStat_edgeonly, np.array([0]), np.array([0]), samplk_sequence, pid='000')
    # BSTERGM_sampler2.run(5000, exchange_iter=50, time_lag=0)
    # # BSTERGM_sampler2.write_posterior_samples("edgeonly_samplk_time01")
    # # BSTERGM_sampler2.write_latest_exchangeSampler_netStat("edgeonly_samplk_time01_netStat")
    # BSTERGM_sampler2.show_traceplot()
    # BSTERGM_sampler2.show_histogram(formation_param_mark_vec=[-2.3427], dissolution_param_mark_vec=[0.5596])
    # BSTERGM_sampler2.show_latest_exchangeSampler_netStat_traceplot()


    #get_bergm test
    BSTERGM_sampler4 = BSTERGM(model_netStat_edgeonly, np.array([0]), np.array([0]), sociational_interactions, pid='000')
    BERGM_sampler4_f, BERGM_sampler4_d = BSTERGM_sampler4.get_bergm_objects_with_setting(time_lag=0)
    BERGM_sampler4_f.run(5000, 50, proposal_cov_rate=0.01, console_output_str='formation')
    # BERGM_sampler4_f.write_posterior_samples("BERGM_sampler4_f")
    # BERGM_sampler4_f.write_latest_exchangeSampler_netStat("BERGM_sampler4_f_netStat")
    BERGM_sampler4_d.run(5000, 50, proposal_cov_rate=0.01, console_output_str='dissolution')
    # BERGM_sampler4_d.write_posterior_samples("BERGM_sampler4_d")
    # BERGM_sampler4_d.write_latest_exchangeSampler_netStat("BERGM_sampler4_d_netStat")
    
    BERGM_sampler4_f.show_traceplot()
    BERGM_sampler4_d.show_traceplot()
    # BERGM_sampler4_f.show_histogram(param_mark=[-1.3502])
    # BERGM_sampler4_d.show_histogram(param_mark=[0.6274])
    # BERGM_sampler4_f.show_latest_exchangeSampler_netStat_traceplot()
    # BERGM_sampler4_d.show_latest_exchangeSampler_netStat_traceplot()

    BSTERGM_sampler4.show_traceplot()