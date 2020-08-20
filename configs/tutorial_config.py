import numpy as np
import pickle
import cddm_data_simulation as cd
import boundary_functions as bf
import os

tutorial_config = {
"ddm":{ 
    "parameters": ['v', 'a', 'z', 't'],
    "param_bounds_sampler": [[-2.0, 2.0], [0.5, 2.0], [0.3, 0.7], [0.2, 1.8]],
    "param_bounds_cnn": [[-3.0, 3.0], [0.3, 2.5], [0.1, 0.9], [0.0, 2.0]],
    "boundary_param_bounds_network": [],
    "boundary_param_bounds_sampler": [],
    "boundary_param_bounds_cnn":[],
    },
"angle":{
    "parameters": ['v', 'a', 'z', 't', 'theta'],
    "param_bounds": [[-2.0, 2.0], [0.5, 1.8], [0.3, 0.7], [0.2, 1.8]],
    'boundary_param_bounds_network':[[0, (np.pi / 2 - .1)]],
    "boundary_param_bounds_sampler": [[0.2, np.pi / 2 - .5]],
    "boundary_param_bounds_cnn": [[0, (np.pi / 2 - .2)]],
    },
"weibull_cdf":{
    "parameters": ['v', 'a', 'z', 't', 'alpha', 'beta'],
    "param_bounds": [[-2.0, 2.0], [0.5, 1.7], [0.3, 0.7], [0.2, 1.8]],
    "boundary_param_bounds_network": [[0.3, 5.0], [0.3, 7.0]],
    "boundary_param_bounds": [[1.0, 4.0], [1.0, 6.0]],
    "boundary_param_bounds_cnn": [[0.3, 5.0], [0.3, 7.0]],
    },
}