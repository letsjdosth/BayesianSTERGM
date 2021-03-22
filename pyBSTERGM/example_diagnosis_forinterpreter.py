import csv

import numpy as np

from BSTERGM_diagnosis import BSTERGM_posterior_work, BSTERGM_latest_exchangeSampler_work

model_select = 'samplk_edgeonly'
prior_select = 'norm' #'norm', 'unif

#select string list (norm, unif)
# samplk_vig
# friendship_KHex
# friendship_2hom_noprisch (#<-now running)
# tailorshop_edgeDegrESP
# tailorshop_edgeDegrESPDSP

#(unif only)
# friendship_simplified #<- no freq estimate (X)
# friendship_2hom_chains #<- no freq estimate (X)
# tailorshop_edgeESP
# tailorshop_edgeESPDSP


#(norm only)
# samplk_edgeonly
# friendship_edgeonly
# friendship_edge1hom
# friendship_edge1homModelstat
# friendship_edge2hom
# tailorshop_edgeonly
# tailorshop_edgeGWdgre
# tailorshop_edgeGWESP
# tailorshop_edgeGWDSP


chain_idx = 1
#0~5 (some models permit ~8)



if model_select == 'samplk_edgeonly':
    reader_inst = BSTERGM_posterior_work()
    reader_inst.read_from_csv("example_results_"+ prior_select +"Prior/samplk_"+ prior_select +"Prior_edgeonly_"+str(chain_idx)+"chain", 1, 1)
    
    reader_inst.MC_formation_samples = reader_inst.MC_formation_samples[10000::50]
    reader_inst.MC_dissolution_samples = reader_inst.MC_dissolution_samples[10000::50]

    print(np.mean(reader_inst.MC_sample_trace()[0][0]))
    # STERGM formation: -2.4980
    
    print(np.mean(reader_inst.MC_sample_trace()[1][0]))
    # STERGM dissolution: 0.7066

    reader_inst.show_traceplot()
    reader_inst.show_histogram(formation_mark=[-2.4980],
        dissolution_mark=[0.7066])
    reader_inst.show_acfplot()


elif model_select == 'samplk_vig': # model_netStat_samplk_vignettesEx
    reader_inst = BSTERGM_posterior_work()
    reader_inst.read_from_csv("example_results_"+ prior_select +"Prior/samplk_"+ prior_select +"Prior_vignetModel_"+str(chain_idx)+"chain", 4, 4)
    
    reader_inst.MC_formation_samples = reader_inst.MC_formation_samples[2000::20]
    reader_inst.MC_dissolution_samples = reader_inst.MC_dissolution_samples[2000::20]

    print(np.mean(reader_inst.MC_sample_trace()[0][0]), np.mean(reader_inst.MC_sample_trace()[0][1]),
        np.mean(reader_inst.MC_sample_trace()[0][2]), np.mean(reader_inst.MC_sample_trace()[0][3]))
    # STERGM formation: -3.5586, 2.2624, -0.4994, 0.2945
    
    print(np.mean(reader_inst.MC_sample_trace()[1][0]), np.mean(reader_inst.MC_sample_trace()[1][1]),
        np.mean(reader_inst.MC_sample_trace()[1][2]), np.mean(reader_inst.MC_sample_trace()[1][3]))
    # STERGM dissolution: -0.1164, 1.5791, -1.6957, 0.6847

    reader_inst.show_traceplot()
    reader_inst.show_histogram(formation_mark=[-3.5586, 2.2624, -0.4994, 0.2945], dissolution_mark=[-0.1164, 1.5791, -1.6957, 0.6847]) #mark: frequentist model
    reader_inst.show_acfplot()



elif model_select == 'friendship_KHex': #FULL MODEL OF STERGM paper
    reader_inst = BSTERGM_posterior_work()
    reader_inst.read_from_csv("example_results_"+ prior_select +"Prior/friendship_"+ prior_select +"Prior_KHExModel_"+str(chain_idx)+"chain", 8, 8)
    
    reader_inst.MC_formation_samples = reader_inst.MC_formation_samples[5000::20]
    reader_inst.MC_dissolution_samples = reader_inst.MC_dissolution_samples[5000::20]

    print(np.mean(reader_inst.MC_sample_trace()[0][0]), np.mean(reader_inst.MC_sample_trace()[0][1]),
        np.mean(reader_inst.MC_sample_trace()[0][2]), np.mean(reader_inst.MC_sample_trace()[0][3]),
        np.mean(reader_inst.MC_sample_trace()[0][4]), np.mean(reader_inst.MC_sample_trace()[0][5]),
        np.mean(reader_inst.MC_sample_trace()[0][6]), np.mean(reader_inst.MC_sample_trace()[0][7]))
    # STERGM formation: -3.336, 0.480, 0.973, -0.358, 0.650, 1.384, 0.886, -0.389
    
    print(np.mean(reader_inst.MC_sample_trace()[1][0]), np.mean(reader_inst.MC_sample_trace()[1][1]),
        np.mean(reader_inst.MC_sample_trace()[1][2]), np.mean(reader_inst.MC_sample_trace()[1][3]),
        np.mean(reader_inst.MC_sample_trace()[1][4]), np.mean(reader_inst.MC_sample_trace()[1][5]),
        np.mean(reader_inst.MC_sample_trace()[1][6]), np.mean(reader_inst.MC_sample_trace()[1][7]))
    # STERGM dissolution: -1.132, 0.122, 1.168, -0.577, 0.451, 2.682, 1.121, -1.016

    reader_inst.show_traceplot()
    reader_inst.show_histogram(formation_mark=[-3.336, 0.480, 0.973, -0.358, 0.650, 1.384, 0.886, -0.389], 
        dissolution_mark=[-1.132, 0.122, 1.168, -0.577, 0.451, 2.682, 1.121, -1.016]) #mark: frequentist model
    reader_inst.show_acfplot()

# elif model_select == 'friendship_simplified': #edges +hetero + primary school + mutual + transitiveties + cyclicties
#     reader_inst = BSTERGM_posterior_work()
#     reader_inst.read_from_csv("friendship_simplied_model/friendship_sequence_simplified_"+str(chain_idx)+"chain", 6, 6)
    
#     reader_inst.MC_formation_samples = reader_inst.MC_formation_samples[5000::20]
#     reader_inst.MC_dissolution_samples = reader_inst.MC_dissolution_samples[5000::20]

#     print(np.mean(reader_inst.MC_sample_trace()[0][0]), np.mean(reader_inst.MC_sample_trace()[0][1]),
#         np.mean(reader_inst.MC_sample_trace()[0][2]), np.mean(reader_inst.MC_sample_trace()[0][3]),
#         np.mean(reader_inst.MC_sample_trace()[0][4]), np.mean(reader_inst.MC_sample_trace()[0][5]))
#     # STERGM formation: ??
    
#     print(np.mean(reader_inst.MC_sample_trace()[1][0]), np.mean(reader_inst.MC_sample_trace()[1][1]),
#         np.mean(reader_inst.MC_sample_trace()[1][2]), np.mean(reader_inst.MC_sample_trace()[1][3]),
#         np.mean(reader_inst.MC_sample_trace()[1][4]), np.mean(reader_inst.MC_sample_trace()[1][5]))
#     # STERGM dissolution: ??

#     reader_inst.show_traceplot()
#     reader_inst.show_histogram()
#     reader_inst.show_acfplot()

# elif model_select == 'friendship_2hom_chains': #edges+ homo(girl) + homo(boys) + primary school + mutual + transitiveties + cyclicties
#     reader_inst = BSTERGM_posterior_work()
#     reader_inst.read_from_csv("friendship_sequence_2hom_chains/friendship_sequence_2hom_"+str(chain_idx)+"chain", 7, 7)
    
#     reader_inst.MC_formation_samples = reader_inst.MC_formation_samples[10000::50]
#     reader_inst.MC_dissolution_samples = reader_inst.MC_dissolution_samples[10000::50]

#     print(np.mean(reader_inst.MC_sample_trace()[0][0]), np.mean(reader_inst.MC_sample_trace()[0][1]),
#         np.mean(reader_inst.MC_sample_trace()[0][2]), np.mean(reader_inst.MC_sample_trace()[0][3]),
#         np.mean(reader_inst.MC_sample_trace()[0][4]), np.mean(reader_inst.MC_sample_trace()[0][5]),
#         np.mean(reader_inst.MC_sample_trace()[0][6]))
#     # STERGM formation: ??
    
#     print(np.mean(reader_inst.MC_sample_trace()[1][0]), np.mean(reader_inst.MC_sample_trace()[1][1]),
#         np.mean(reader_inst.MC_sample_trace()[1][2]), np.mean(reader_inst.MC_sample_trace()[1][3]),
#         np.mean(reader_inst.MC_sample_trace()[1][4]), np.mean(reader_inst.MC_sample_trace()[1][5]),
#         np.mean(reader_inst.MC_sample_trace()[1][6]))
#     # STERGM dissolution: ??

#     reader_inst.show_traceplot()
#     reader_inst.show_histogram()
#     reader_inst.show_acfplot()

elif model_select == 'friendship_2hom_noprisch': # correct R code?
    reader_inst = BSTERGM_posterior_work()
    reader_inst.read_from_csv("example_results_"+ prior_select +"Prior/friendship_"+ prior_select +"Prior_2homNoprischModel_"+str(chain_idx)+"chain", 6, 6)
    
    reader_inst.MC_formation_samples = reader_inst.MC_formation_samples[10000::50]
    reader_inst.MC_dissolution_samples = reader_inst.MC_dissolution_samples[10000::50]

    print(np.mean(reader_inst.MC_sample_trace()[0][0]), np.mean(reader_inst.MC_sample_trace()[0][1]),
        np.mean(reader_inst.MC_sample_trace()[0][2]), np.mean(reader_inst.MC_sample_trace()[0][3]),
        np.mean(reader_inst.MC_sample_trace()[0][4]), np.mean(reader_inst.MC_sample_trace()[0][5]))
    # STERGM formation: -3.4113, 0.6497, 0.9130, 1.6660, 0.7663, -0.3590
    
    print(np.mean(reader_inst.MC_sample_trace()[1][0]), np.mean(reader_inst.MC_sample_trace()[1][1]),
        np.mean(reader_inst.MC_sample_trace()[1][2]), np.mean(reader_inst.MC_sample_trace()[1][3]),
        np.mean(reader_inst.MC_sample_trace()[1][4]), np.mean(reader_inst.MC_sample_trace()[1][5]))
    # STERGM dissolution: -1.1243, 0.1402, 1.1940, 2.4484, 1.1083, -0.9749

    reader_inst.show_traceplot()
    reader_inst.show_histogram(formation_mark=[-3.4113, 0.6497, 0.9130, 1.6660, 0.7663, -0.3590],
        dissolution_mark=[-1.1243, 0.1402, 1.1940, 2.4484, 1.1083, -0.9749])
    reader_inst.show_acfplot()



elif model_select == 'friendship_edgeonly':
    reader_inst = BSTERGM_posterior_work()
    reader_inst.read_from_csv("example_results_"+ prior_select +"Prior/friendship_"+ prior_select +"Prior_edgeonly_"+str(chain_idx)+"chain", 1, 1)
    
    reader_inst.MC_formation_samples = reader_inst.MC_formation_samples[10000::50]
    reader_inst.MC_dissolution_samples = reader_inst.MC_dissolution_samples[10000::50]

    print(np.mean(reader_inst.MC_sample_trace()[0][0]))
    # STERGM formation: -2.12535
    
    print(np.mean(reader_inst.MC_sample_trace()[1][0]))
    # STERGM dissolution: 0.4720

    reader_inst.show_traceplot()
    reader_inst.show_histogram(formation_mark=[-2.12535],
        dissolution_mark=[0.4720])
    reader_inst.show_acfplot()


elif model_select == 'friendship_edge1hom':
    reader_inst = BSTERGM_posterior_work()
    reader_inst.read_from_csv("example_results_"+ prior_select +"Prior/friendship_"+ prior_select +"Prior_edge1hom_"+str(chain_idx)+"chain", 2, 2)
    
    reader_inst.MC_formation_samples = reader_inst.MC_formation_samples[10000::50]
    reader_inst.MC_dissolution_samples = reader_inst.MC_dissolution_samples[10000::50]

    print(np.mean(reader_inst.MC_sample_trace()[0][0]), np.mean(reader_inst.MC_sample_trace()[0][1]))
    # STERGM formation: -2.3967, 0.6393
    
    print(np.mean(reader_inst.MC_sample_trace()[1][0]), np.mean(reader_inst.MC_sample_trace()[1][1]))
    # STERGM dissolution: 0.6725, -0.3529

    reader_inst.show_traceplot()
    reader_inst.show_histogram(formation_mark=[-2.3967, 0.6393],
        dissolution_mark=[0.6725, -0.3529])
    reader_inst.show_acfplot()


elif model_select == 'friendship_edge1homModelstat':
    reader_inst = BSTERGM_posterior_work()
    reader_inst.read_from_csv("example_results_"+ prior_select +"Prior/friendship_"+ prior_select +"Prior_edge1homModelstat_"+str(chain_idx)+"chain", 5, 5)
    
    reader_inst.MC_formation_samples = reader_inst.MC_formation_samples[10000::50]
    reader_inst.MC_dissolution_samples = reader_inst.MC_dissolution_samples[10000::50]

    print(np.mean(reader_inst.MC_sample_trace()[0][0]), np.mean(reader_inst.MC_sample_trace()[0][1]),
            np.mean(reader_inst.MC_sample_trace()[0][2]), np.mean(reader_inst.MC_sample_trace()[0][3]),
            np.mean(reader_inst.MC_sample_trace()[0][4]))
    # STERGM formation: -3.3356, 0.3729, 1.7863, 0.8548, -0.3395
    
    print(np.mean(reader_inst.MC_sample_trace()[1][0]), np.mean(reader_inst.MC_sample_trace()[1][1]),
        np.mean(reader_inst.MC_sample_trace()[1][2]), np.mean(reader_inst.MC_sample_trace()[1][3]),
        np.mean(reader_inst.MC_sample_trace()[1][4]))
    # STERGM dissolution: -0.5614, -0.3988, 2.4832, 1.0823, -0.9708

    reader_inst.show_traceplot()
    reader_inst.show_histogram(formation_mark=[-3.3356, 0.3729, 1.7863, 0.8548, -0.3395],
        dissolution_mark=[-0.5614, -0.3988, 2.4832, 1.0823, -0.9708])
    reader_inst.show_acfplot()


elif model_select == 'friendship_edge2hom':
    reader_inst = BSTERGM_posterior_work()
    reader_inst.read_from_csv("example_results_"+ prior_select +"Prior/friendship_"+ prior_select +"Prior_edge2hom_"+str(chain_idx)+"chain", 3, 3)

    reader_inst.MC_formation_samples = reader_inst.MC_formation_samples[10000::50]
    reader_inst.MC_dissolution_samples = reader_inst.MC_dissolution_samples[10000::50]

    print(np.mean(reader_inst.MC_sample_trace()[0][0]), np.mean(reader_inst.MC_sample_trace()[0][1]),
            np.mean(reader_inst.MC_sample_trace()[0][2]))
    # STERGM formation: -2.7502, 0.9928, 1.4880
    
    print(np.mean(reader_inst.MC_sample_trace()[1][0]), np.mean(reader_inst.MC_sample_trace()[1][1]),
        np.mean(reader_inst.MC_sample_trace()[1][2]))
    # STERGM dissolution: -0.03077, 0.35041, 1.41707

    reader_inst.show_traceplot()
    reader_inst.show_histogram(formation_mark=[-2.7502, 0.9928, 1.4880],
        dissolution_mark=[-0.03077, 0.35041, 1.41707])
    reader_inst.show_acfplot()


elif model_select == 'tailorshop_edgeDegrESP': #edges + gwdegree(0.25) + gwesp(0.25)
    reader_inst = BSTERGM_posterior_work()
    reader_inst.read_from_csv("example_results_"+ prior_select +"Prior/tailorshop_social_"+ prior_select +"Prior_edgeGWDgr025GWESP025_"+str(chain_idx)+"chain", 3, 3)
    
    reader_inst.MC_formation_samples = reader_inst.MC_formation_samples[5000::20]
    reader_inst.MC_dissolution_samples = reader_inst.MC_dissolution_samples[5000::20]

    print(np.mean(reader_inst.MC_sample_trace()[0][0]), np.mean(reader_inst.MC_sample_trace()[0][1]),
        np.mean(reader_inst.MC_sample_trace()[0][2]))
    # STERGM formation: 2.5761, -18.7338, 0.8973
    
    print(np.mean(reader_inst.MC_sample_trace()[1][0]), np.mean(reader_inst.MC_sample_trace()[1][1]),
        np.mean(reader_inst.MC_sample_trace()[1][2]))
    # STERGM dissolution: -0.190885, 0.000973, 0.513791

    reader_inst.show_traceplot()
    reader_inst.show_histogram(formation_mark=[2.5761, -18.7338, 0.8973], dissolution_mark=[-0.190885, 0.000973, 0.513791])
    reader_inst.show_acfplot()

elif model_select == 'tailorshop_edgeDegrESPDSP': #edges + gwdegree(0.25) + gwesp(0.25) + gwdsp(0.25)
    reader_inst = BSTERGM_posterior_work()
    reader_inst.read_from_csv("example_results_"+ prior_select +"Prior/tailorshop_social_"+ prior_select +"Prior_edgeGWDgr025GWESP025GWDSP025_"+str(chain_idx)+"chain", 4, 4)

    
    reader_inst.MC_formation_samples = reader_inst.MC_formation_samples[10000::50]
    reader_inst.MC_dissolution_samples = reader_inst.MC_dissolution_samples[10000::50]

    print(np.mean(reader_inst.MC_sample_trace()[0][0]), np.mean(reader_inst.MC_sample_trace()[0][1]),
        np.mean(reader_inst.MC_sample_trace()[0][2]), np.mean(reader_inst.MC_sample_trace()[0][3]))
    # STERGM formation: -2.35, -12640(R return: -1.264e+04), 0.821, 0.0799
    
    print(np.mean(reader_inst.MC_sample_trace()[1][0]), np.mean(reader_inst.MC_sample_trace()[1][1]),
        np.mean(reader_inst.MC_sample_trace()[1][2]), np.mean(reader_inst.MC_sample_trace()[1][3]))
    # STERGM dissolution: 0.17852, 0.26155, 0.53004, -0.07981

    reader_inst.show_traceplot()
    reader_inst.show_histogram(formation_mark=[-2.35, -12640, 0.821, 0.0799], dissolution_mark=[0.17852, 0.26155, 0.53004, -0.07981])
    reader_inst.show_acfplot()

elif model_select == 'tailorshop_edgeESP': #edges + gwesp(0.3)
    reader_inst = BSTERGM_posterior_work()
    reader_inst.read_from_csv("tailorSoc_results/tailorshop_social_edgeGWESP03_"+str(chain_idx)+"chain", 2, 2)
    
    reader_inst.MC_formation_samples = reader_inst.MC_formation_samples[10000::50]
    reader_inst.MC_dissolution_samples = reader_inst.MC_dissolution_samples[10000::50]

    print(np.mean(reader_inst.MC_sample_trace()[0][0]), np.mean(reader_inst.MC_sample_trace()[0][1]))
    # STERGM formation: -2.8992, 1.0662
    
    print(np.mean(reader_inst.MC_sample_trace()[1][0]), np.mean(reader_inst.MC_sample_trace()[1][1]))
    # STERGM dissolution: -0.2389, 0.5237

    reader_inst.show_traceplot()
    reader_inst.show_histogram(formation_mark=[-2.8992, 1.0662], dissolution_mark=[-0.2389, 0.5237])
    reader_inst.show_acfplot()

elif model_select == 'tailorshop_edgeESPDSP': #edges + gwesp(0.25) + gwesp(0.25)
    reader_inst = BSTERGM_posterior_work()
    reader_inst.read_from_csv("tailorSoc_results/tailorshop_social_edgeGWESP025GWDSP025_"+str(chain_idx)+"chain", 3, 3)
    
    reader_inst.MC_formation_samples = reader_inst.MC_formation_samples[10000::50]
    reader_inst.MC_dissolution_samples = reader_inst.MC_dissolution_samples[10000::50]

    print(np.mean(reader_inst.MC_sample_trace()[0][0]), np.mean(reader_inst.MC_sample_trace()[0][1]),
    np.mean(reader_inst.MC_sample_trace()[0][2]))
    # STERGM formation: -2.99616, 1.34537, -0.23871
    
    print(np.mean(reader_inst.MC_sample_trace()[1][0]), np.mean(reader_inst.MC_sample_trace()[1][1]),
    np.mean(reader_inst.MC_sample_trace()[1][2]))
    # STERGM dissolution: 0.22094, 0.49755, -0.07524

    reader_inst.show_traceplot()
    reader_inst.show_histogram(formation_mark=[-2.99616, 1.34537, -0.23871], dissolution_mark=[0.22094, 0.49755, -0.07524])
    reader_inst.show_acfplot()


elif model_select == 'tailorshop_edgeGWdgre': #edges + gwDgre(0.25)
    reader_inst = BSTERGM_posterior_work()
    reader_inst.read_from_csv("example_results_"+ prior_select +"Prior/tailorshop_social_"+ prior_select +"Prior_edgeGWDgre_"+str(chain_idx)+"chain", 2, 2)
    
    reader_inst.MC_formation_samples = reader_inst.MC_formation_samples[10000::50]
    reader_inst.MC_dissolution_samples = reader_inst.MC_dissolution_samples[10000::50]

    print(np.mean(reader_inst.MC_sample_trace()[0][0]), np.mean(reader_inst.MC_sample_trace()[0][1]))
    # STERGM formation: not provided.
    
    print(np.mean(reader_inst.MC_sample_trace()[1][0]), np.mean(reader_inst.MC_sample_trace()[1][1]))
    # STERGM dissolution: not provided.

    reader_inst.show_traceplot()
    reader_inst.show_histogram()
    reader_inst.show_acfplot()


elif model_select == 'tailorshop_edgeGWDSP': #edges + gwDSP(0.25)
    reader_inst = BSTERGM_posterior_work()
    reader_inst.read_from_csv("example_results_"+ prior_select +"Prior/tailorshop_social_"+ prior_select +"Prior_edgeGWDSP_"+str(chain_idx)+"chain", 2, 2)
    
    reader_inst.MC_formation_samples = reader_inst.MC_formation_samples[10000::50]
    reader_inst.MC_dissolution_samples = reader_inst.MC_dissolution_samples[10000::50]

    print(np.mean(reader_inst.MC_sample_trace()[0][0]), np.mean(reader_inst.MC_sample_trace()[0][1]))
    # STERGM formation: -1.20221, -0.15973
    
    print(np.mean(reader_inst.MC_sample_trace()[1][0]), np.mean(reader_inst.MC_sample_trace()[1][1]))
    # STERGM dissolution: 1.08853, -0.08773

    reader_inst.show_traceplot()
    reader_inst.show_histogram(formation_mark=[-1.20221, -0.15973], dissolution_mark=[1.08853, -0.08773])
    reader_inst.show_acfplot()


elif model_select == 'tailorshop_edgeGWESP': #edges + gwESP(0.25)
    reader_inst = BSTERGM_posterior_work()
    reader_inst.read_from_csv("example_results_"+ prior_select +"Prior/tailorshop_social_"+ prior_select +"Prior_edgeGWESP_"+str(chain_idx)+"chain", 2, 2)
    
    reader_inst.MC_formation_samples = reader_inst.MC_formation_samples[10000::50]
    reader_inst.MC_dissolution_samples = reader_inst.MC_dissolution_samples[10000::50]

    print(np.mean(reader_inst.MC_sample_trace()[0][0]), np.mean(reader_inst.MC_sample_trace()[0][1]))
    # STERGM formation: -2.5998, 0.9106
    
    print(np.mean(reader_inst.MC_sample_trace()[1][0]), np.mean(reader_inst.MC_sample_trace()[1][1]))
    # STERGM dissolution: -0.1921, 0.5155

    reader_inst.show_traceplot()
    reader_inst.show_histogram(formation_mark=[-2.5998, 0.9106], dissolution_mark=[-0.1921, 0.5155])
    reader_inst.show_acfplot()


elif model_select == 'tailorshop_edgeonly':
    reader_inst = BSTERGM_posterior_work()
    reader_inst.read_from_csv("example_results_"+ prior_select +"Prior/tailorshop_social_"+ prior_select +"Prior_edgeonly_"+str(chain_idx)+"chain", 1, 1)
    
    reader_inst.MC_formation_samples = reader_inst.MC_formation_samples[10000::50]
    reader_inst.MC_dissolution_samples = reader_inst.MC_dissolution_samples[10000::50]

    print(np.mean(reader_inst.MC_sample_trace()[0][0]))
    # STERGM formation: -1.3502
    
    print(np.mean(reader_inst.MC_sample_trace()[1][0]))
    # STERGM dissolution: 0.6274

    reader_inst.show_traceplot()
    reader_inst.show_histogram(formation_mark=[-1.3502],
        dissolution_mark=[0.6274])
    reader_inst.show_acfplot()
