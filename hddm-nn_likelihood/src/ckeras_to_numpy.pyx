# Load packages
#import tensorflow as tf
#from tensorflow import keras
import cython
import numpy as np
import pickle  

# Activations
def relu(x):
    return np.maximum(0, x)

def linear(x):
    return x

def sigmoid(x):
    return 1 / (1 + np.exp(- x))

# def tanh(x):
#     return (2 / (1 + np.exp(- 2 * x))) - 1

# Function to extract network architecture 
def extract_architecture(model, 
                         save = False,
                         save_path = ''):
    
    biases = []
    activations = []
    weights = []
    
    for layer in model.layers:
        if layer.name == "input_1":
            continue
        weights.append(layer.get_weights()[0])
        biases.append(layer.get_weights()[1])
        activations.append(layer.get_config()["activation"])
            
    if save == True:
       pickle.dump(weights, open(save_path + "weights.pickle", "wb"))
       pickle.dump(biases, open(save_path + "biases.pickle", "wb"))
       pickle.dump(activations, open(save_path + "activations.pickle", "wb"))
            
    return weights, biases, activations

# Function to perform forward pass given architecture
def predict(x, weights, biases, activations, n_layers):
    # Activation dict
    activation_fns = {"relu":relu, "linear":linear, 'sigmoid':sigmoid, 'tanh':np.tanh}

    #for i in range(len(weights)):
    for i in range(n_layers):
        x = activation_fns[activations[i]](
            np.dot(x, weights[i]) + biases[i])
    return x

def log_p(params, 
          weights, 
          biases, 
          activations, 
          data, 
          orig_output_log_l = True, 
          ll_min = 1e-29):
    
    param_grid = np.tile(params, (data.shape[0], 1))
    inp = np.concatenate([param_grid, data], axis = 1)
    
    # TD add ll_min back in ?
    return - np.sum(predict(inp, 
                            weights, 
                            biases, 
                            activations))
    # else:
    #     out = np.maximum(predict(inp, 
    #                              weights, 
    #                              biases, 
    #                              activations), ll_min)
    #     return - np.sum(np.log(out))

def group_log_p(params, 
                weights, 
                biases, 
                activations, 
                data, 
                param_varies = [0, 0, 1], 
                params_ordered = ['v', 'a', 'w'],
                params_names = ['v', 'a', 'w_0', 'w_1', 'w_2'],
                orig_output_log_l = True,
                ll_min = 1e-29):
    
    n_datasets = len(data)
    n_params = len(params_ordered)
    log_p_out = 0
    for i in range(n_datasets):
        
        # Parameters for data id
        params_tmp = get_tmp_params(params = params, 
                                    params_ordered = params_ordered,
                                    param_varies = param_varies,
                                    params_names = params_names,
                                    idx = i)
               
        # Compute log_likelihood for current dataset
        log_p_out += log_p(params = params_tmp, 
                           weights = weights, 
                           biases = biases, 
                           activations = activations, 
                           data = data[str(i)], 
                           ll_min = ll_min,
                           orig_output_log_l = orig_output_log_l)
    return log_p_out

# Support functions -----------------
def get_tmp_params(params = [0, 1, 2, 3, 4],
                   params_ordered = ['v', 'a', 'w'],
                   param_varies = [0, 0, 1],
                   params_names = ['v', 'a', 'w_0', 'w_1', 'w_2'],
                   idx = 0):
    
    # Get parameters for current dataset
        n_params = len(params_ordered)
        params_tmp = []
        for j in range(n_params):
            if param_varies[j] == 1:
                params_tmp.append(params[params_names.index(params_ordered[j] + '_' + str(idx))])
            else:
                params_tmp.append(params[params_names.index(params_ordered[j])])
                
#         print('parameter_names_adj: ', params_names)
#         print('parameters: ', params)
#         print('idx: ', idx)
#         print('params out: ', params_tmp)
        return params_tmp
# ------------------------------------ 