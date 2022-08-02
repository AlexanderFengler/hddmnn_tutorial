# MODULE IMPORTS ----

# warning settings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Data management
import pandas as pd
import numpy as np
import pickle

# Plotting
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

# Stats functionality
from statsmodels.distributions.empirical_distribution import ECDF

import multiprocessing

# HDDM
import hddm

def run_model(id):
    import hddm
    from hddm.simulators.hddm_dataset_generators import simulator_h_c
    
    def v_link(x):
        return x
    
    model = 'angle'
    n_trials_per_subject = 500
    n_subjects = 10
    
    data, full_parameter_dict = simulator_h_c(data = None, 
                                              n_subjects = n_subjects,
                                              n_trials_per_subject = n_trials_per_subject,
                                              model = model,
                                              p_outlier = 0.00,
                                              conditions = None, 
                                              depends_on = None, 
                                              regression_models = ['v ~ 1',
                                                                  'a ~ 1'],
                                              regression_covariates = {'prevresp': {'type': 'categorical', 'range': (0, 2)},
                                                                       'stimulus': {'type': 'categorical', 'range': (0, 2)}},
                                              group_only_regressors = True,
                                              group_only = None,
                                              fixed_at_default = ['z'])

    #ToDo: transform the z-param so it can find the right bounds?
    regr_md = [ #{'model': 'v ~ 1 + stimulus + prevresp', 'link_func': v_link},
               {'model': 'v ~ 1', 'link_func': v_link},
               {'model': 'a ~ 1', 'link_func': v_link}]

    
    from copy import deepcopy
    my_model_config = hddm.model_config.model_config[model].copy()
    my_include = deepcopy(hddm.model_config.model_config[model]['hddm_include'])
    my_include.remove('z')
    hddmnn_reg = hddm.HDDMnnRegressor(data,
                                      regr_md,
                                      include = my_include, # 'sv' is not allowed here
                                      model = model,
                                      model_config = my_model_config,
                                      informative = False,
                                      is_group_model = True, # hierarchical model
                                      # group_only_nodes = ['vx'],
                                      group_only_regressors = True, # fit one parameter for each subject
                                      # indirect_regressors = indirect_regressors, #indirect_regressors,
                                      indirect_betas = None, #indirect_betas,
                                      p_outlier = 0.05)
    
    print('start sampling')
    hddmnn_reg.sample(100, burn = 50, dbname = 'test_model.db', db='pickle')
    print('end_sampling')
    hddmnn_reg.save('test_model.pickle')



if __name__ == "__main__":
    pool = multiprocessing.get_context('spawn').Pool()
    models = pool.map(run_model, range(2))
    pool.close()

