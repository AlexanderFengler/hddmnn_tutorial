import hddm
import pandas as pd
import numpy as np
#import re
import argparse
import sys
import pickle
from statsmodels.distributions.empirical_distribution import ECDF


sys.path.append('simulators')

from cddm_data_simulation import ddm 
from cddm_data_simulation import ddm_flexbound
from cddm_data_simulation import levy_flexbound
from cddm_data_simulation import ornstein_uhlenbeck
from cddm_data_simulation import full_ddm
from cddm_data_simulation import ddm_sdv
from cddm_data_simulation import ddm_flexbound_pre
import cddm_data_simulation as cds

import boundary_functions as bf

# plotting 
import matplotlib.pyplot as plt
import seaborn as sns

# DATA SIMULATION ------------------------------------------------------------------------------
def bin_simulator_output(out = [0, 0],
                         bin_dt = 0.04,
                         nbins = 0): # ['v', 'a', 'w', 'ndt', 'angle']
        
    # Generate bins
    if nbins == 0:
        nbins = int(out[2]['max_t'] / bin_dt)
        bins = np.zeros(nbins + 1)
        bins[:nbins] = np.linspace(0, out[2]['max_t'], nbins)
        bins[nbins] = np.inf
    else:  
        bins = np.zeros(nbins + 1)
        bins[:nbins] = np.linspace(0, out[2]['max_t'], nbins)
        bins[nbins] = np.inf

    cnt = 0
    counts = np.zeros( (nbins, len(out[2]['possible_choices']) ) )

    for choice in out[2]['possible_choices']:
        counts[:, cnt] = np.histogram(out[0][out[1] == choice], bins = bins)[0] / out[2]['n_samples']
        cnt += 1
    return counts


def simulator(theta, model = 'angle', n_samples = 1000, bin_dim = None):
    
    # Useful for sbi
    if type(theta) == list or type(theta) == np.ndarray:
        pass
    else:
        theta = theta.numpy()
    
    if model == 'ddm':
        x = ddm(v = theta[0], a = theta[1], w = theta[2], ndt = theta[3], n_samples = n_samples)
    
    if model == 'angle':
        x = ddm_flexbound(v = theta[0], a = theta[1], w = theta[2], ndt = theta[3], 
                          boundary_fun = bf.angle, 
                          boundary_multiplicative = False,
                          boundary_params = {'theta': theta[4]}, 
                          n_samples = n_samples)
    
    if model == 'weibull_cdf':
        x = ddm_flexbound(v = theta[0], a = theta[1], w = theta[2], ndt = theta[3], 
                          boundary_fun = bf.weibull_cdf, 
                          boundary_multiplicative = True, 
                          boundary_params = {'alpha': theta[4], 'beta': theta[5]}, 
                          n_samples = n_samples)
    
    if model == 'levy':
        x = levy_flexbound(v = theta[0], a = theta[1], w = theta[2], alpha_diff = theta[3], ndt = theta[4], 
                           boundary_fun = bf.constant, 
                           boundary_multiplicative = True, 
                           boundary_params = {}, 
                           n_samples = n_samples)
    
    if model == 'full_ddm':
        x = full_ddm(v = theta[0], a = theta[1], w = theta[2], ndt = theta[3], dw = theta[4], sdv = theta[5], dndt = theta[6], 
                     boundary_fun = bf.constant, 
                     boundary_multiplicative = True, 
                     boundary_params = {}, 
                     n_samples = n_samples)

    if model == 'ddm_sdv':
        x = ddm_sdv(v = theta[0], a = theta[1], w = theta[2], ndt = theta[3], sdv = theta[4],
                    boundary_fun = bf.constant,
                    boundary_multiplicative = True, 
                    boundary_params = {},
                    n_samples = n_samples)
        
    if model == 'ornstein_uhlenbeck':
        x = ornstein_uhlenbeck(v = theta[0], a = theta[1], w = theta[2], g = theta[3], ndt = theta[4],
                               boundary_fun = bf.constant,
                               boundary_multiplicative = True,
                               boundary_params = {},
                               n_samples = n_samples)

    if model == 'pre':
        x = ddm_flexbound_pre(v = theta[0], a = theta[1], w = theta[2], ndt = theta[3],
                              boundary_fun = bf.angle,
                              boundary_multiplicative = False,
                              boundary_params = {'theta': theta[4]},
                              n_samples = n_samples)
    if bin_dim == None:
        return x
    else:
        return bin_simulator_output(x, nbins = bin_dim).flatten()
    
def simulator_condition_effects(n_conditions = 4, 
                                n_samples_by_condition = 1000,
                                condition_effect_on_param = [0], 
                                model = 'angle',
                                ):
    
    config = {'ddm': {'params':['v', 'a', 'z', 't'],
                      'param_bounds': [[-2, 0.5, 0.3, 0.2], [2, 2, 0.7, 1.8]],
                     
                     },
              'angle':{'params': ['v', 'a', 'z', 't', 'theta'],
                       'param_bounds': [[-2, -.5, 0.3, 0.2, 0.2], [2, 1.8, 0.7, 1.8, np.pi / 2 - 0.5]],
                      
                      },
              'weibull_cdf':{'params': ['v', 'a', 'z', 't', 'alpha', 'beta'],
                              'param_bounds': [[-2, 0.5, 0.3, 0.2, 1.0, 1.0], [2, 1.7, 0.7, 1.8, 4.0, 6.0]]
                            
                            },
             }
     
    param_base = np.tile(np.random.uniform(low = config[model]['param_bounds'][0],
                                           high = config[model]['param_bounds'][1], 
                                           size = (1, len(config[model]['params']))),
                                           (n_conditions, 1))
                         
    #len(config[model]['params']                   
    #print(param_base)
    gt = {}
    for i in range(n_conditions):
        for c_eff in condition_effect_on_param:
            id_tmp = config[model]['params'].index(c_eff)
            print(id_tmp)
            print(config[model]['param_bounds'][0])
            param_base[i, id_tmp] = np.random.uniform(low = config[model]['param_bounds'][0][id_tmp], 
                                                      high = config[model]['param_bounds'][1][id_tmp])
            gt[c_eff + '(' + str(i) + ')'] = param_base[i, id_tmp]
    
    for param in config[model]['params']:
        if param in condition_effect_on_param:
            pass
        else:
            id_tmp = config[model]['params'].index(param)
            gt[param] = param_base[0, id_tmp]
            
    #print(param_base)
    dataframes = []
    for i in range(n_conditions):
        sim_out = simulator(param_base[i, :], 
                            model = model, 
                            n_samples = n_samples_by_condition,
                            bin_dim = None)
        
        dataframes.append(hddm_preprocess(simulator_data = sim_out, subj_id = i))
    
    data_out = pd.concat(dataframes)
    data_out = data_out.rename(columns = {'subj_idx': "condition"})
    # print(param_base.shape)
    return (data_out, gt, param_base)
   
def hddm_preprocess(simulator_data = None, subj_id = 0):
    
    df = pd.DataFrame(simulator_data[0].astype(np.double), columns = ['rt'])
    df['response'] = simulator_data[1].astype(int)
    df['nn_response'] = df['response']
    df.loc[df['response'] == -1.0, 'response'] = 0.0
    df['subj_idx'] = subj_id
    
    return df


def hddm_preprocess_hierarchical(model = None, datasetid = 0):

    data = pickle.load(open('data_storage/' + model + '_tutorial_nsubj_5_n_1000.pickle', 'rb'))
    n_parti = data[1].shape[2]
    masterlist = []
    df = pd.DataFrame(columns = ['rt', 'response'])
    
    for j in range(n_parti):
        masterlist.append(pd.DataFrame(data[1][0][datasetid][j], columns = ['rt', 'response']))
        masterlist[j].insert(0,'subj_idx',j)
        df = df.append(masterlist[j], ignore_index = True, sort = True)
    
    df['nn_response'] = df['response']
    df.loc[df['response'] == -1.0, 'response'] = 0.0
    
    if model == 'angle':
        gt_subj = pd.DataFrame(data[0][0][datasetid], columns = ['v','a','z','t', 'theta'])
        gt_global_sds = pd.DataFrame(np.array([data[0][1][datasetid]]), columns = ['v','a','z','t', 'theta'])
        gt_global_means = pd.DataFrame(np.array([data[0][2][datasetid]]), columns = ['v', 'a', 'z', 't', 'theta'])
    
    elif model == 'ddm':
        gt_subj = pd.DataFrame(data[0][0][datasetid], columns = ['v','a','z','t'])
        gt_global_sds = pd.DataFrame(np.array([data[0][1][datasetid]]), columns = ['v','a','z','t'])
        gt_global_means = pd.DataFrame(np.array([data[0][2][datasetid]]), columns = ['v', 'a', 'z', 't'])
    
    elif model == 'weibull_cdf':
        gt_subj = pd.DataFrame(data[0][0][datasetid], columns = ['v', 'a', 'z', 't', 'alpha', 'beta'])
        gt_global_sds = pd.DataFrame(np.array([data[0][1][datasetid]]), columns = ['v', 'a', 'z', 't', 'alpha', 'beta'])
        gt_global_means = pd.DataFrame(np.array([data[0][2][datasetid]]), columns = ['v', 'a', 'z', 't', 'alpha', 'beta'])    
    return (df, gt_subj, gt_global_sds, gt_global_means)

def _make_trace_plotready(hddm_trace = None, model = ''):
    param_order = {'ddm': ['v', 'a', 'z', 't'],
                   'angle': ['v', 'a', 'z', 't', 'theta'],
                   'weibull_cdf': ['v', 'a', 'z', 't', 'alpha', 'beta'],
                  }
    
    posterior_samples = np.zeros(hddm_trace.shape)
    
    cnt = 0
    for param in param_order[model]:
        if param == 'z':
            posterior_samples[:, cnt] = 1 / (1 + np.exp( - hddm_trace['z_trans']))
        else:
            posterior_samples[:, cnt] = hddm_trace[param]
        cnt += 1
    
    return posterior_samples

def _make_trace_plotready_hierarchical(hddm_trace = None, model = ''):
    param_order = {'ddm': ['v', 'a', 'z', 't'],
                   'angle': ['v', 'a', 'z', 't', 'theta'],
                   'weibull_cdf': ['v', 'a', 'z', 't', 'alpha', 'beta'],
                  }
    
    subj_l = []
    for key in hddm_trace.keys():
        if '_subj' in key:
            subj_l.append(int(float(key[-3:])))

    dat = np.zeros((max(subj_l) + 1, hddm_trace.shape[0], len(param_order[model])))
    for key in hddm_trace.keys():
        if '_subj' in key:
            id_tmp = int(float(key[-3:]))
            if '_trans' in key:
                val_tmp = 1 / ( 1 + np.exp(- hddm_trace[key]))
            else:
                val_tmp = hddm_trace[key]
            dat[id_tmp, : ,param_order[model].index(key[:key.find('_')])] = val_tmp   
            
    return dat

def _make_trace_plotready_condition(hddm_trace = None, model = ''):
    param_order = {'ddm': ['v', 'a', 'z', 't'],
                   'angle': ['v', 'a', 'z', 't', 'theta'],
                   'weibull_cdf': ['v', 'a', 'z', 't', 'alpha', 'beta'],
                  }
    cond_l = []
    for key in hddm_trace.keys():
        if '(' in key:
            cond_l.append(int(float(key[-2])))
    
    dat = np.zeros((max(cond_l) + 1, hddm_trace.shape[0], len(param_order[model])))
                   
    for key in hddm_trace.keys():
        if '(' in key:
            id_tmp = int(float(key[-2]))
            if '_trans' in key:
                val_tmp = 1 / ( 1 + np.exp(- hddm_trace[key]))
            else:
                val_tmp = hddm_trace[key]
            
            dat[id_tmp, : ,param_order[model].index(key[:key.find('(')])] = val_tmp   
        else:
            if '_trans' in key:
                val_tmp = 1 / ( 1 + np.exp(- hddm_trace[key]))
                key = key[:key.find('_trans')]
            else:
                val_tmp = hddm_trace[key]
                   
            dat[:, :, param_order[model].index(key)] = val_tmp
            
    return dat

# --------------------------------------------------------------------------------------------

# Plot bound
# Mean posterior predictives
def model_plot(posterior_samples = None,
               ground_truths = [],
               cols = 3,
               model = 'weibull_cdf',
               n_post_params = 500,
               n_plots = 4,
               samples_by_param = 10,
               max_t = 5,
               input_hddm_trace = False,
               datatype = 'single_subject', # 'hierarchical', 'single_subject', 'condition'
               show_model = True):
    
    # Inputs are hddm_traces --> make plot ready
    if input_hddm_trace and posterior_samples is not None:
        if datatype == 'hierarchical':
            posterior_samples = _make_trace_plotready_hierarchical(posterior_samples, 
                                                                   model = model)
            n_plots = posterior_samples.shape[0]
#             print(posterior_samples)
            
        if datatype == 'single_subject':
            posterior_samples = _make_trace_plotready(posterior_samples, 
                                                      model = model)
        if datatype == 'condition':
            posterior_samples = _make_trace_plotready_condition(posterior_samples, 
                                                                model = model)
            n_plots = posterior_samples.shape[0]
            #print(posterior_samples)
            #n_plots = posterior_samples.shape[0]

    
    # Taking care of special case with 1 plot
    print(n_plots)
    if n_plots == 1:
        ground_truths = np.expand_dims(ground_truths, 0)
        posterior_samples = np.expand_dims(posterior_samples, 0)
#         print(ground_truths)
#         print(ground_truths.shape)
    
    plot_titles = {'ddm': 'DDM', 
                   'angle': 'ANGLE',
                   'full_ddm': 'FULL DDM',
                   'weibull_cdf': 'WEIBULL',
                   'levy': 'LEVY',
                   'ornstein': 'ORNSTEIN UHLENBECK',
                   'ddm_sdv': 'DDM RANDOM SLOPE',
                  }
    
    title = 'Model Plot: '
    
    ax_titles = {'ddm': ['v', 'a' 'z', 'ndt'],
                 'angle': ['v', 'a', 'z', 'ndt', 'theta'],
                 'full_ddm': ['v', 'a', 'z', 'ndt', 'dw', 'sdv', 'dndt'],
                 'weibull_cdf': ['v', 'a', 'z' , 'ndt', 'alpha', 'beta'],
                 'levy': ['v', 'a', 'z', 'ndt', 'alpha'],
                 'ornstein': ['v', 'a', 'z', 'g', 'ndt'],
                 'ddm_sdv': ['v', 'a', 'z', 'ndt', 'sdv'],
                }
    
    rows = int(np.ceil(n_plots / cols))
#     if posterior_samples is not None:
#         sub_idx = np.random.choice(posterior_samples.shape[1], size = n_post_params)
#         posterior_samples = posterior_samples[:, sub_idx, :]
    
    sns.set(style = "white", 
            palette = "muted", 
            color_codes = True,
            font_scale = 2)

    fig, ax = plt.subplots(rows, cols, 
                           figsize = (20, 20), 
                           sharex = False, 
                           sharey = False)
    
    my_suptitle = fig.suptitle(title + plot_titles[model], fontsize = 40)
    sns.despine(right = True)
    
    t_s = np.arange(0, max_t, 0.01)
    for i in range(n_plots):
        row_tmp = int(np.floor(i / cols))
        col_tmp = i - (cols * row_tmp)
        
        if rows > 1 and cols > 1:
            ax[row_tmp, col_tmp].set_xlim(0, max_t)
            ax[row_tmp, col_tmp].set_ylim(-2, 2)
        else:
            ax.set_xlim(0, max_t)
            ax.set_ylim(-2, 2)
        
        # Run simulations and add histograms
        # True params
        if model == 'angle' or model == 'angle2':
            out = cds.ddm_flexbound(v = ground_truths[i, 0],
                                    a = ground_truths[i, 1],
                                    w = ground_truths[i, 2],
                                    ndt = ground_truths[i, 3],
                                    s = 1,
                                    delta_t = 0.01, 
                                    max_t = 20,
                                    n_samples = 20000,
                                    print_info = False,
                                    boundary_fun = bf.angle,
                                    boundary_multiplicative = False,
                                    boundary_params = {'theta': ground_truths[i, 4]})
            
        if model == 'weibull_cdf' or model == 'weibull_cdf2':
            out = cds.ddm_flexbound(v = ground_truths[i, 0],
                                    a = ground_truths[i, 1],
                                    w = ground_truths[i, 2],
                                    ndt = ground_truths[i, 3],
                                    s = 1,
                                    delta_t = 0.01, 
                                    max_t = 20,
                                    n_samples = 20000,
                                    print_info = False,
                                    boundary_fun = bf.weibull_cdf,
                                    boundary_multiplicative = True,
                                    boundary_params = {'alpha': ground_truths[i, 4],
                                                       'beta': ground_truths[i, 5]})
        
        if model == 'ddm':
            out = cds.ddm_flexbound(v = ground_truths[i, 0],
                                    a = ground_truths[i, 1],
                                    w = ground_truths[i, 2],
                                    ndt = ground_truths[i, 3],
                                    s = 1,
                                    delta_t = 0.01,
                                    max_t = 20, 
                                    n_samples = 20000,
                                    print_info = False,
                                    boundary_fun = bf.constant,
                                    boundary_multiplicative = True,
                                    boundary_params = {})
            
        
        tmp_true = np.concatenate([out[0], out[1]], axis = 1)
        choice_p_up_true = np.sum(tmp_true[:, 1] == 1) / tmp_true.shape[0]
        
        if posterior_samples is not None:
            # Run Model simulations for posterior samples
            tmp_post = np.zeros((n_post_params*samples_by_param, 2))
            idx = np.random.choice(posterior_samples.shape[1], size = n_post_params, replace = False)

            for j in range(n_post_params):
                if model == 'angle' or model == 'angle2':
                    out = cds.ddm_flexbound(v = posterior_samples[i, idx[j], 0],
                                            a = posterior_samples[i, idx[j], 1],
                                            w = posterior_samples[i, idx[j], 2],
                                            ndt = posterior_samples[i, idx[j], 3],
                                            s = 1,
                                            delta_t = 0.01, 
                                            max_t = 20,
                                            n_samples = samples_by_param,
                                            print_info = False,
                                            boundary_fun = bf.angle,
                                            boundary_multiplicative = False,
                                            boundary_params = {'theta': posterior_samples[i, idx[j], 4]})

                if model == 'weibull_cdf' or model == 'weibull_cdf2':
                    out = cds.ddm_flexbound(v = posterior_samples[i, idx[j], 0],
                                            a = posterior_samples[i, idx[j], 1],
                                            w = posterior_samples[i, idx[j], 2],
                                            ndt = posterior_samples[i, idx[j], 3],
                                            s = 1,
                                            delta_t = 0.01, 
                                            max_t = 20,
                                            n_samples = samples_by_param,
                                            print_info = False,
                                            boundary_fun = bf.weibull_cdf,
                                            boundary_multiplicative = True,
                                            boundary_params = {'alpha': posterior_samples[i, idx[j], 4],
                                                               'beta': posterior_samples[i, idx[j], 5]})

                if model == 'ddm':
                    out = cds.ddm_flexbound(v = posterior_samples[i, idx[j], 0],
                                            a = posterior_samples[i, idx[j], 1],
                                            w = posterior_samples[i, idx[j], 2],
                                            ndt = posterior_samples[i, idx[j], 3],
                                            s = 1,
                                            delta_t = 0.01,
                                            max_t = 20, 
                                            n_samples = samples_by_param,
                                            print_info = False,
                                            boundary_fun = bf.constant,
                                            boundary_multiplicative = True,
                                            boundary_params = {})

                tmp_post[(10 * j):(10 * (j + 1)), :] = np.concatenate([out[0], out[1]], axis = 1)
        
         #ax.set_ylim(-4, 2)
        if rows > 1 and cols > 1:
            ax_tmp = ax[row_tmp, col_tmp].twinx()
        else:
            ax_tmp = ax.twinx()
        
        ax_tmp.set_ylim(-2, 2)
        ax_tmp.set_yticks([])
        
        if posterior_samples is not None:
            choice_p_up_post = np.sum(tmp_post[:, 1] == 1) / tmp_post.shape[0]



            counts, bins = np.histogram(tmp_post[tmp_post[:, 1] == 1, 0],
                                        bins = np.linspace(0, 10, 100))

            counts_2, bins = np.histogram(tmp_post[tmp_post[:, 1] == 1, 0],
                                          bins = np.linspace(0, 10, 100),
                                          density = True)
            
            if j == (n_post_params - 1):
                ax_tmp.hist(bins[:-1], 
                            bins, 
                            weights = choice_p_up_post * counts_2,
                            histtype = 'step',
                            alpha = 0.5, 
                            color = 'black',
                            edgecolor = 'black',
                            zorder = -1,
                            label = 'Posterior Predictive')
                
            else:
                ax_tmp.hist(bins[:-1], 
                            bins, 
                            weights = choice_p_up_post * counts_2,
                            histtype = 'step',
                            alpha = 0.5, 
                            color = 'black',
                            edgecolor = 'black',
                            zorder = -1)
                        

        counts, bins = np.histogram(tmp_true[tmp_true[:, 1] == 1, 0],
                                bins = np.linspace(0, 10, 100))

        counts_2, bins = np.histogram(tmp_true[tmp_true[:, 1] == 1, 0],
                                      bins = np.linspace(0, 10, 100),
                                      density = True)
        ax_tmp.hist(bins[:-1], 
                    bins, 
                    weights = choice_p_up_true * counts_2,
                    histtype = 'step',
                    alpha = 0.5, 
                    color = 'red',
                    edgecolor = 'red',
                    zorder = -1,
                    label = 'Ground Truth Data')
        ax_tmp.legend(loc = 'lower right')
             
        #ax.invert_xaxis()
        if rows > 1 and cols > 1:
            ax_tmp = ax[row_tmp, col_tmp].twinx()
        else:
            ax_tmp = ax.twinx()
            
        ax_tmp.set_ylim(2, -2)
        ax_tmp.set_yticks([])
        
        if posterior_samples is not None:
            counts, bins = np.histogram(tmp_post[tmp_post[:, 1] == -1, 0],
                            bins = np.linspace(0, 10, 100))

            counts_2, bins = np.histogram(tmp_post[tmp_post[:, 1] == -1, 0],
                                          bins = np.linspace(0, 10, 100),
                                          density = True)
            ax_tmp.hist(bins[:-1], 
                        bins, 
                        weights = (1 - choice_p_up_post) * counts_2,
                        histtype = 'step',
                        alpha = 0.5, 
                        color = 'black',
                        edgecolor = 'black',
                        zorder = -1)
        
        counts, bins = np.histogram(tmp_true[tmp_true[:, 1] == -1, 0],
                                bins = np.linspace(0, 10, 100))
    
        counts_2, bins = np.histogram(tmp_true[tmp_true[:, 1] == -1, 0],
                                      bins = np.linspace(0, 10, 100),
                                      density = True)
        ax_tmp.hist(bins[:-1], 
                    bins, 
                    weights = (1 - choice_p_up_true) * counts_2,
                    histtype = 'step',
                    alpha = 0.5, 
                    color = 'red',
                    edgecolor = 'red',
                    zorder = -1)
        
        # Plot posterior samples of bounds and slopes (model)
        if show_model:
            if posterior_samples is not None:
                for j in range(n_post_params):
                    if model == 'weibull_cdf' or model == 'weibull_cdf2':
                        b = posterior_samples[i, idx[j], 1] * bf.weibull_cdf(t = t_s, 
                                                                             alpha = posterior_samples[i, idx[j], 4],
                                                                             beta = posterior_samples[i, idx[j], 5])
                    if model == 'angle' or model == 'angle2':
                        b = np.maximum(posterior_samples[i, idx[j], 1] + bf.angle(t = t_s, 
                                                                                  theta = posterior_samples[i, idx[j], 4]), 0)
                    if model == 'ddm':
                        b = posterior_samples[i, idx[j], 1] * np.ones(t_s.shape[0])


                    start_point_tmp = - posterior_samples[i, idx[j], 1] + \
                                      (2 * posterior_samples[i, idx[j], 1] * posterior_samples[i, idx[j], 2])

                    slope_tmp = posterior_samples[i, idx[j], 0]

                    if rows > 1 and cols > 1:
                        ax[row_tmp, col_tmp].plot(t_s + posterior_samples[i, idx[j], 3], b, 'black',
                                                  t_s + posterior_samples[i, idx[j], 3], - b, 'black', 
                                                  alpha = 0.05,
                                                  zorder = 1000)
                    else:
                        ax.plot(t_s + posterior_samples[i, idx[j], 3], b, 'black',
                                t_s + posterior_samples[i, idx[j], 3], - b, 'black', 
                                alpha = 0.05,
                                zorder = 1000)

                    for m in range(len(t_s)):
                        if (start_point_tmp + (slope_tmp * t_s[m])) > b[m] or (start_point_tmp + (slope_tmp * t_s[m])) < -b[m]:
                            maxid = m
                            break
                        maxid = m

                    if rows > 1 and cols > 1:
                        ax[row_tmp, col_tmp].plot(t_s[:maxid] + posterior_samples[i, idx[j], 3],
                                                  start_point_tmp + slope_tmp * t_s[:maxid], 
                                                  'black', 
                                                  alpha = 0.05,
                                                  zorder = 1000)
                        if j == (n_post_params - 1):
                            ax[row_tmp, col_tmp].plot(t_s[:maxid] + posterior_samples[i, idx[j], 3],
                                                      start_point_tmp + slope_tmp * t_s[:maxid], 
                                                      'black', 
                                                      alpha = 0.05,
                                                      zorder = 1000,
                                                      label = 'Model Samples')

                    else:
                        ax.plot(t_s[:maxid] + posterior_samples[i, idx[j], 3],
                                start_point_tmp + slope_tmp * t_s[:maxid], 
                                'black', 
                                alpha = 0.05,
                                zorder = 1000)
                        if j ==(n_post_params - 1):
                            ax.plot(t_s[:maxid] + posterior_samples[i, idx[j], 3],
                                    start_point_tmp + slope_tmp * t_s[:maxid], 
                                    'black', 
                                    alpha = 0.05,
                                    zorder = 1000,
                                    label = 'Model Samples')
                            
        # Plot ground_truths bounds
        if show_model:
            if model == 'weibull_cdf' or model == 'weibull_cdf2':
                b = ground_truths[i, 1] * bf.weibull_cdf(t = t_s,
                                                         alpha = ground_truths[i, 4],
                                                         beta = ground_truths[i, 5])

            if model == 'angle' or model == 'angle2':
                b = np.maximum(ground_truths[i, 1] + bf.angle(t = t_s, theta = ground_truths[i, 4]), 0)

            if model == 'ddm':
                b = ground_truths[i, 1] * np.ones(t_s.shape[0])

            start_point_tmp = - ground_truths[i, 1] + \
                              (2 * ground_truths[i, 1] * ground_truths[i, 2])
            slope_tmp = ground_truths[i, 0]

            if rows > 1 and cols > 1:
                if row_tmp == 0 and col_tmp == 0:
                    ax[row_tmp, col_tmp].plot(t_s + ground_truths[i, 3], b, 'red', 
                                              alpha = 1, 
                                              linewidth = 3, 
                                              zorder = 1000)
                    ax[row_tmp, col_tmp].plot(t_s + ground_truths[i, 3], -b, 'red', 
                                              alpha = 1,
                                              linewidth = 3,
                                              zorder = 1000, 
                                              label = 'Grund Truth Model')
                    ax[row_tmp, col_tmp].legend()
                else:
                    ax[row_tmp, col_tmp].plot(t_s + ground_truths[i, 3], b, 'red', 
                              t_s + ground_truths[i, 3], -b, 'red', 
                              alpha = 1,
                              linewidth = 3,
                              zorder = 1000)
            else:
                ax.plot(t_s + ground_truths[i, 3], b, 'red', 
                        alpha = 1, 
                        linewidth = 3, 
                        zorder = 1000)
                ax.plot(t_s + ground_truths[i, 3], -b, 'red', 
                        alpha = 1,
                        linewidth = 3,
                        zorder = 1000,
                        label = 'Ground Truth Model')
                print('passed through legend part')
                print(row_tmp)
                print(col_tmp)
                ax.legend(loc = 'upper right')

            # Ground truth slope:
            for m in range(len(t_s)):
                if (start_point_tmp + (slope_tmp * t_s[m])) > b[m] or (start_point_tmp + (slope_tmp * t_s[m])) < -b[m]:
                    maxid = m
                    break
                maxid = m

            # print('maxid', maxid)
            if rows > 1 and cols > 1:
                ax[row_tmp, col_tmp].plot(t_s[:maxid] + ground_truths[i, 3], 
                                          start_point_tmp + slope_tmp * t_s[:maxid], 
                                          'red', 
                                          alpha = 1, 
                                          linewidth = 3, 
                                          zorder = 1000)

                ax[row_tmp, col_tmp].set_zorder(ax_tmp.get_zorder() + 1)
                ax[row_tmp, col_tmp].patch.set_visible(False)
                # print('passed through')

#                 #ax[row_tmp, col_tmp].legend(labels = [model, 'bg_stn'], fontsize = 20)
#                 if row_tmp == rows:
#                     ax[row_tmp, col_tmp].set_xlabel('rt', 
#                                                     fontsize = 20);
#                 ax[row_tmp, col_tmp].set_ylabel('', 
#                                                 fontsize = 20);


#                 title_tmp = ''
#                 for k in range(len(ax_titles[model])):
#                     title_tmp += ax_titles[model][k] + ': '
#                     title_tmp += str(round(ground_truths[i, k], 2)) + ', ' 

#                 ax[row_tmp, col_tmp].set_title(title_tmp,
#                                                fontsize = 24)
#                 ax[row_tmp, col_tmp].tick_params(axis = 'y', size = 20)
#                 ax[row_tmp, col_tmp].tick_params(axis = 'x', size = 20)

#                 # Some extra styling:
#                 ax[row_tmp, col_tmp].axvline(x = ground_truths[i, 3], ymin = -2, ymax = 2, c = 'red', linestyle = '--')
#                 ax[row_tmp, col_tmp].axhline(y = 0, xmin = 0, xmax = ground_truths[i, 3] / max_t, c = 'red',  linestyle = '--')

            else:
                ax.plot(t_s[:maxid] + ground_truths[i, 3], 
                                          start_point_tmp + slope_tmp * t_s[:maxid], 
                                          'red', 
                                          alpha = 1, 
                                          linewidth = 3, 
                                          zorder = 1000)

                ax.set_zorder(ax_tmp.get_zorder() + 1)
                ax.patch.set_visible(False)
                # print('passed through')

                #ax[row_tmp, col_tmp].legend(labels = [model, 'bg_stn'], fontsize = 20)
#                 if row_tmp == rows:
#                     ax.set_xlabel('rt', 
#                                   fontsize = 20);
#                 ax.set_ylabel('', 
#                               fontsize = 20);

#                 ax.set_title(title_tmp,
#                              fontsize = 24)

#                 ax.tick_params(axis = 'y', size = 20)
#                 ax.tick_params(axis = 'x', size = 20)

#                 # Some extra styling:
#                 ax.axvline(x = ground_truths[i, 3], ymin = -2, ymax = 2, c = 'red', linestyle = '--')
#                 ax.axhline(y = 0, xmin = 0, xmax = ground_truths[i, 3] / max_t, c = 'red',  linestyle = '--')
                
        # Set plot title
        title_tmp = ''
        for k in range(len(ax_titles[model])):
            title_tmp += ax_titles[model][k] + ': '
            title_tmp += str(round(ground_truths[i, k], 2)) + ', ' 

        if rows > 1 and cols > 1:
            if row_tmp == rows:
                ax[row_tmp, col_tmp].set_xlabel('rt', 
                                                 fontsize = 20);
            ax[row_tmp, col_tmp].set_ylabel('', 
                                            fontsize = 20);


            ax[row_tmp, col_tmp].set_title(title_tmp,
                                           fontsize = 24)
            ax[row_tmp, col_tmp].tick_params(axis = 'y', size = 20)
            ax[row_tmp, col_tmp].tick_params(axis = 'x', size = 20)

            # Some extra styling:
            ax[row_tmp, col_tmp].axvline(x = ground_truths[i, 3], ymin = -2, ymax = 2, c = 'red', linestyle = '--')
            ax[row_tmp, col_tmp].axhline(y = 0, xmin = 0, xmax = ground_truths[i, 3] / max_t, c = 'red',  linestyle = '--')

        else:
            if row_tmp == rows:
                ax.set_xlabel('rt', 
                              fontsize = 20);
            ax.set_ylabel('', 
                          fontsize = 20);

            ax.set_title(title_tmp,
                         fontsize = 24)

            ax.tick_params(axis = 'y', size = 20)
            ax.tick_params(axis = 'x', size = 20)

            # Some extra styling:
            ax.axvline(x = ground_truths[i, 3], ymin = -2, ymax = 2, c = 'red', linestyle = '--')
            ax.axhline(y = 0, xmin = 0, xmax = ground_truths[i, 3] / max_t, c = 'red',  linestyle = '--')

    
    if rows > 1 and cols > 1:
        for i in range(n_plots, rows * cols, 1):
            row_tmp = int(np.floor(i / cols))
            col_tmp = i - (cols * row_tmp)
            ax[row_tmp, col_tmp].axis('off')

    plt.tight_layout(rect = [0, 0.03, 1, 0.9])
    
    return plt.show()

def caterpillar_plot(trace = [],
                     gt = [],
                     model = 'angle',
                     datatype = 'hierarchical', # 'hierarchical', 'single_subject', 'condition'
                     drop_sd = True):
    
    sns.set(style = "white", 
        palette = "muted", 
        color_codes = True,
        font_scale = 2)

    fig, ax = plt.subplots(1, 1, 
                           figsize = (10, 10), 
                           sharex = False, 
                           sharey = False)
    
    my_suptitle = fig.suptitle('Caterpillar plot: ' + model, fontsize = 40)
    sns.despine(right = True)
    
    trace = trace.copy()
    gt_dict = {}
    cnt = 0
    
    if datatype == 'single_subject':
        if model == 'ddm':
            for v in ['v', 'a', 'z', 't']:
                gt_dict[v] = gt[cnt]
                cnt += 1

        if model == 'weibull_cdf':
            for v in ['v', 'a', 'z', 't', 'alpha', 'beta']:
                gt_dict[v] = gt[cnt]
                cnt += 1

        if model == 'angle':
            for v in ['v', 'a', 'z', 't', 'theta']:
                gt_dict[v] = gt[cnt]
                cnt += 1
    if datatype == 'hierarchical':
        tmp = {}
        tmp['subj'] = gt[1]
        tmp['global_means'] = gt[2]
        tmp['global_sds'] = gt[3]
        gt = tmp
        
        gt_dict = {}
        for param in gt['subj'].keys():
            for i in range(gt['subj'].shape[0]):
                gt_dict[param + '_subj.' + str(i) + '.0'] = gt['subj'][param][i]
        for param in gt['global_means'].keys():
            gt_dict[param] = gt['global_means'][param][0]
            
    if datatype == 'condition':
        gt_dict = gt
        
            
    ecdfs = {}
    plot_vals = {} # [0.01, 0.9], [0.01, 0.99], [mean]
    for k in trace.keys():
        if 'std' in k:
            pass
        else:
            if '_trans' in k:
                label_tmp = k.replace('_trans', '')
                trace[label_tmp] = 1 / (1 + np.exp(- trace[k]))
                k = label_tmp
                #print(trace[k].mean())
            #print(k)
            #print(trace[k])
            ok_ = 1
            if drop_sd == True:
                if 'sd' in k:
                    ok_ = 0
            if ok_:
                ecdfs[k] = ECDF(trace[k])
                tmp_sorted = sorted(trace[k])
                _p01 =  tmp_sorted[np.sum(ecdfs[k](tmp_sorted) <= 0.01) - 1]
                _p99 = tmp_sorted[np.sum(ecdfs[k](tmp_sorted) <= 0.99) - 1]
                _p1 = tmp_sorted[np.sum(ecdfs[k](tmp_sorted) <= 0.1) - 1]
                _p9 = tmp_sorted[np.sum(ecdfs[k](tmp_sorted) <= 0.9) - 1]
                _pmean = trace[k].mean()
                plot_vals[k] = [[_p01, _p99], [_p1, _p9], _pmean]
        
    x = [plot_vals[k][2] for k in plot_vals.keys()]
    ax.scatter(x, plot_vals.keys(), c = 'black', marker = 's', alpha = 0)
    for k in plot_vals.keys():
        ax.plot(plot_vals[k][1], [k, k], c = 'grey', zorder = -1, linewidth = 5)
        ax.plot(plot_vals[k][0] , [k, k], c = 'black', zorder = -1)
        #print(k)
        #print(gt_dict[k])
        ax.scatter(gt_dict[k], k,  c = 'red', marker = "|")

    return plt.show()
#         if save:
#             if machine == 'home':
#                 fig_dir = "/users/afengler/OneDrive/git_repos/nn_likelihoods/figures/" + method + "/model_uncertainty"
#                 if not os.path.isdir(fig_dir):
#                     os.mkdir(fig_dir)

#             figure_name = 'model_uncertainty_plot_'
#             plt.savefig(fig_dir + '/' + figure_name + model + data_signature + '_' + train_data_type + '.png',
#                         dpi = 150, 
#                         transparent = False,
#                         bbox_inches = 'tight',
#                         bbox_extra_artists = [my_suptitle])
#             plt.close()
#         if show:
#             return #plt.show(block = False)

