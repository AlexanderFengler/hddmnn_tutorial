# Globaly settings for cython
# cython: cdivision=True
# cython: wraparound=False
# cython: boundscheck=False

# Functions for DDM data simulation
import cython
from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport log, sqrt, pow, fmax, atan, sin, cos, tan, M_PI, M_PI_2

import numpy as np
#import pandas as pd
from time import time
import inspect
import pickle

DTYPE = np.float32

# Method to draw random samples from a gaussian
cdef float random_uniform():
    cdef float r = rand()
    return r / RAND_MAX

cdef float random_exponential():
    return - log(random_uniform())

cdef float random_stable(float alpha_diff):
    cdef float eta, u, w, x
    # chi = - tan(M_PI_2 * alpha_diff)

    u = M_PI * (random_uniform() - 0.5)
    w = random_exponential()

    if alpha_diff == 1.0:
        eta = M_PI_2 # useless but kept to remain faithful to wikipedia entry
        x = (1.0 / eta) * ((M_PI_2) * tan(u))
        # x = (1.0 / eta) * ((M_PI_2 + u) * tan(u) - log((M_PI_2 * w * cos(u)) / (M_PI_2 + u)))
    else:
        # eta = (1.0 / alpha_diff) * atan(- chi)
        x = (sin(alpha_diff * u) / (pow(cos(u), 1 / alpha_diff))) * pow(cos(u - (alpha_diff * u)) / w, (1.0 - alpha_diff) / alpha_diff)
        # x = pow((1.0 + chi * chi), 1.0 / (2.0 * alpha_diff)) * \
        #        (sin(alpha_diff * (u + eta)) / pow(cos(u), 1.0 / alpha_diff)) * \
        #        pow(cos(u - (alpha_diff * (u + eta))) / w, (1.0 - alpha_diff) / alpha_diff)
    return x

cdef float[:] draw_random_stable(int n, float alpha_diff):
    cdef int i
    cdef float[:] result = np.zeros(n, dtype = DTYPE)

    for i in range(n):
        result[i] = random_stable(alpha_diff)
    return result

cdef float random_gaussian():
    cdef float x1, x2, w
    w = 2.0

    while(w >= 1.0):
        x1 = 2.0 * random_uniform() - 1.0
        x2 = 2.0 * random_uniform() - 1.0
        w = x1 * x1 + x2 * x2

    w = ((-2.0 * log(w)) / w) ** 0.5
    return x1 * w

cdef int sign(float x):
    return (x > 0) - (x < 0)

cdef float csum(float[:] x):
    cdef int i
    cdef int n = x.shape[0]
    cdef float total = 0
    
    for i in range(n):
        total += x[i]
    
    return total

## @cythonboundscheck(False)
cdef void assign_random_gaussian_pair(float[:] out, int assign_ix):
    cdef float x1, x2, w
    w = 2.0

    while(w >= 1.0):
        x1 = (2.0 * random_uniform()) - 1.0
        x2 = (2.0 * random_uniform()) - 1.0
        w = (x1 * x1) + (x2 * x2)

    w = ((-2.0 * log(w)) / w) ** 0.5
    out[assign_ix] = x1 * w
    out[assign_ix + 1] = x2 * w # this was x2 * 2 ..... :0 

# @cythonboundscheck(False)
cdef float[:] draw_gaussian(int n):
    # Draws standard normal variables - need to have the variance rescaled
    cdef int i
    cdef float[:] result = np.zeros(n, dtype=DTYPE)
    for i in range(n // 2):
        assign_random_gaussian_pair(result, i * 2)
    if n % 2 == 1:
        result[n - 1] = random_gaussian()
    return result

# DUMMY TEST SIMULATOR ------------------------------------------------------------------------
# Simulate (rt, choice) tuples from: SIMPLE DDM -----------------------------------------------
# Simplest algorithm
# delete random comment
# delete random comment 2
#@cython.boundscheck(False)
#@cython.wraparound(False)

def test(float v = 0, # drift by timestep 'delta_t'
         float a = 1, # boundary separation
         float w = 0.5,  # between 0 and 1
         float ndt = 0.0, # non-decision time
         float s = 1, # noise sigma
         float delta_t = 0.001, # timesteps fraction of seconds
         float max_t = 20, # maximum rt allowed
         int n_samples = 20000, # number of samples considered
         print_info = True # timesteps fraction of seconds
         ):

    rts = np.zeros((n_samples, 1), dtype = DTYPE)
    choices = np.zeros((n_samples, 1), dtype = np.intc)
    cdef float[:, :] rts_view = rts
    cdef int[:, :] choices_view = choices

    cdef float delta_t_sqrt = sqrt(delta_t)
    cdef float sqrt_st = delta_t_sqrt * s

    cdef float y, t

    cdef int n
    cdef int m = 0
    cdef int num_draws = int(max_t / delta_t + 1)
    cdef float[:] gaussian_values = draw_gaussian(num_draws)

    # Loop over samples
    for n in range(n_samples):
        y = w * a # reset starting point
        t = 0.0 # reset time

        # Random walker
        while y <= a and y >= 0 and t <= max_t:
            y += v * delta_t + sqrt_st * gaussian_values[m] # update particle position
            t += delta_t
            m += 1
            if m == num_draws:
                gaussian_values = draw_gaussian(num_draws)
                m = 0

        # Note that for purposes of consistency with Navarro and Fuss, 
        # the choice corresponding the lower barrier is +1, higher barrier is -1
        rts_view[n, 0] = t + ndt # store rt
        choices_view[n, 0] = (-1) * sign(y) # store choice
        
    return (rts, choices, {'v': v,
                           'a': a,
                           'w': w,
                           'ndt': ndt,
                           's': s,
                           'delta_t': delta_t,
                           'max_t': max_t,
                           'n_samples': n_samples,
                           'simulator': 'ddm',
                           'boundary_fun_type': 'constant',
                           'possible_choices': [-1, 1]})
# ---------------------------------------------------------------------------------------------

# Simulate (rt, choice) tuples from: SIMPLE DDM -----------------------------------------------
# Simplest algorithm
# delete random comment
# delete random comment 2
#@cython.boundscheck(False)
#@cython.wraparound(False)

def ddm(float v = 0, # drift by timestep 'delta_t'
        float a = 1, # boundary separation
        float w = 0.5,  # between 0 and 1
        float ndt = 0.0, # non-decision time
        float s = 1, # noise sigma
        float delta_t = 0.001, # timesteps fraction of seconds
        float max_t = 20, # maximum rt allowed
        int n_samples = 20000, # number of samples considered
        print_info = True # timesteps fraction of seconds
        ):

    rts = np.zeros((n_samples, 1), dtype = DTYPE)
    choices = np.zeros((n_samples, 1), dtype = np.intc)
    cdef float[:, :] rts_view = rts
    cdef int[:, :] choices_view = choices

    cdef float delta_t_sqrt = sqrt(delta_t)
    cdef float sqrt_st = delta_t_sqrt * s

    cdef float y, t

    cdef int n
    cdef int m = 0
    cdef int num_draws = int(max_t / delta_t + 1)
    cdef float[:] gaussian_values = draw_gaussian(num_draws)

    # Loop over samples
    for n in range(n_samples):
        y = w * a # reset starting point
        t = 0.0 # reset time

        # Random walker
        while y <= a and y >= 0 and t <= max_t:
            y += v * delta_t + sqrt_st * gaussian_values[m] # update particle position
            t += delta_t
            m += 1
            if m == num_draws:
                gaussian_values = draw_gaussian(num_draws)
                m = 0

        # Note that for purposes of consistency with Navarro and Fuss, 
        # the choice corresponding the lower barrier is +1, higher barrier is -1
        rts_view[n, 0] = t + ndt # store rt
        choices_view[n, 0] = (-1) * sign(y) # store choice
        
    return (rts, choices, {'v': v,
                           'a': a,
                           'w': w,
                           'ndt': ndt,
                           's': s,
                           'delta_t': delta_t,
                           'max_t': max_t,
                           'n_samples': n_samples,
                           'simulator': 'ddm',
                           'boundary_fun_type': 'constant',
                           'possible_choices': [-1, 1]})


# Simulate (rt, choice) tuples from: DDM WITH FLEXIBLE BOUNDARIES ------------------------------------
# @cythonboundscheck(False)
# @cythonwraparound(False)
def ddm_flexbound(float v = 0,
                  float a = 1,
                  float w = 0.5,
                  float ndt = 0.0,
                  float s = 1,
                  float delta_t = 0.001,
                  float max_t = 20,
                  int n_samples = 20000,
                  print_info = True,
                  boundary_fun = None, # function of t (and potentially other parameters) that takes in (t, *args)
                  boundary_multiplicative = True,
                  boundary_params = {}
                  ):

    rts = np.zeros((n_samples, 1), dtype = DTYPE)
    choices = np.zeros((n_samples, 1), dtype = np.intc)

    cdef float[:,:] rts_view = rts
    cdef int[:,:] choices_view = choices

    cdef float delta_t_sqrt = sqrt(delta_t) # correct scalar so we can use standard normal samples for the brownian motion
    cdef float sqrt_st = delta_t_sqrt * s # scalar to ensure the correct variance for the gaussian step

    # Boundary storage for the upper bound
    cdef int num_draws = int((max_t / delta_t) + 1)
    boundary = np.zeros(num_draws, dtype = DTYPE)
    cdef float[:] boundary_view = boundary
    cdef int i
    cdef float tmp

    # Precompute boundary evaluations
    if boundary_multiplicative:
        for i in range(num_draws):
            tmp = a * boundary_fun(t = i * delta_t, **boundary_params)
            if tmp > 0:
                boundary_view[i] = tmp
    else:
        for i in range(num_draws):
            tmp = a + boundary_fun(t = i * delta_t, **boundary_params)
            if tmp > 0:
                boundary_view[i] = tmp

    cdef float y, t
    cdef int n, ix
    cdef int m = 0
    cdef float[:] gaussian_values = draw_gaussian(num_draws)

    # Loop over samples
    for n in range(n_samples):
        y = (-1) * boundary_view[0] + (w * 2 * (boundary_view[0]))  # reset starting position 
        t = 0 # reset time
        ix = 0 # reset boundary index

        # Random walker
        while y >= (-1) * boundary_view[ix] and y <= boundary_view[ix] and t <= max_t:
            y += (v * delta_t) + (sqrt_st * gaussian_values[m])
            t += delta_t
            ix += 1
            m += 1
            if m == num_draws:
                gaussian_values = draw_gaussian(num_draws)
                m = 0

        rts_view[n, 0] = t + ndt # Store rt
        choices_view[n, 0] = sign(y) # Store choice

    return (rts, choices,  {'v': v,
                            'a': a,
                            'w': w,
                            'ndt': ndt,
                            's': s,
                            **boundary_params,
                            'delta_t': delta_t,
                            'max_t': max_t,
                            'n_samples': n_samples,
                            'simulator': 'ddm_flexbound',
                            'boundary_fun_type': boundary_fun.__name__,
                            'possible_choices': [-1, 1]})
# ----------------------------------------------------------------------------------------------------

# Simulate (rt, choice) tuples from: Levy Flight with Flex Bound -------------------------------------
# @cythonboundscheck(False)
# @cythonwraparound(False)
def levy_flexbound(float v = 0,
                   float a = 1,
                   float w = 0.5,
                   float alpha_diff = 1,
                   float ndt = 0.0,
                   float s = 1,
                   float delta_t = 0.001,
                   float max_t = 20,
                   int n_samples = 20000,
                   print_info = True,
                   boundary_fun = None, # function of t (and potentially other parameters) that takes in (t, *args)
                   boundary_multiplicative = True,
                   boundary_params = {}
                   ):

    rts = np.zeros((n_samples, 1), dtype = DTYPE)
    choices = np.zeros((n_samples, 1), dtype = np.intc)

    cdef float[:,:] rts_view = rts
    cdef int[:,:] choices_view = choices

    cdef float delta_t_alpha = pow(delta_t, 1.0 / alpha_diff) # correct scalar so we can use standard normal samples for the brownian motion
    #cdef float sqrt_st = delta_t_sqrt * s # scalar to ensure the correct variance for the gaussian step

    # Boundary storage for the upper bound
    cdef int num_draws = int((max_t / delta_t) + 1)
    boundary = np.zeros(num_draws, dtype = DTYPE)
    cdef float[:] boundary_view = boundary
    cdef int i
    cdef float tmp

    # Precompute boundary evaluations
    if boundary_multiplicative:
        for i in range(num_draws):
            tmp = a * boundary_fun(t = i * delta_t, **boundary_params)
            if tmp > 0:
                boundary_view[i] = tmp
    else:
        for i in range(num_draws):
            tmp = a + boundary_fun(t = i * delta_t, **boundary_params)
            if tmp > 0:
                boundary_view[i] = tmp

    cdef float y, t
    cdef int n, ix
    cdef int m = 0
    cdef float[:] gaussian_values = draw_random_stable(num_draws, alpha_diff)

    # Loop over samples
    for n in range(n_samples):
        y = (-1) * boundary_view[0] + (w * 2 * (boundary_view[0]))  # reset starting position 
        t = 0 # reset time
        ix = 0 # reset boundary index

        # Random walker
        while y >= (-1) * boundary_view[ix] and y <= boundary_view[ix] and t <= max_t:
            y += (v * delta_t) + (delta_t_alpha * gaussian_values[m])
            t += delta_t
            ix += 1
            m += 1
            if m == num_draws:
                gaussian_values = draw_random_stable(num_draws, alpha_diff)
                m = 0

        rts_view[n, 0] = t + ndt # Store rt
        choices_view[n, 0] = sign(y) # Store choice

    return (rts, choices,  {'v': v,
                            'a': a,
                            'w': w,
                            'ndt': ndt,
                            'alpha_diff': alpha_diff,
                            's': s,
                            **boundary_params,
                            'delta_t': delta_t,
                            'max_t': max_t,
                            'n_samples': n_samples,
                            'simulator': 'levy_flexbound',
                            'boundary_fun_type': boundary_fun.__name__,
                            'possible_choices': [-1, 1]})
# -------------------------------------------------------------------------------------------------

# Simulate (rt, choice) tuples from: Full DDM with flexible bounds --------------------------------
# @cythonboundscheck(False)
# @cythonwraparound(False)
def full_ddm(float v = 0,
             float a = 1,
             float w = 0.5,
             float ndt = 0.0,
             float dw = 0.05,
             float sdv = 0.1,
             float dndt = 0.0,
             float s = 1,
             float delta_t = 0.001,
             float max_t = 20,
             int n_samples = 20000,
             print_info = True,
             boundary_fun = None, # function of t (and potentially other parameters) that takes in (t, *args)
             boundary_multiplicative = True,
             boundary_params = {}
             ):

    rts = np.zeros((n_samples, 1), dtype = DTYPE)
    choices = np.zeros((n_samples, 1), dtype = np.intc)

    cdef float[:, :] rts_view = rts
    cdef int[:, :] choices_view = choices

    cdef float delta_t_sqrt = sqrt(delta_t) # correct scalar so we can use standard normal samples for the brownian motion
    cdef float sqrt_st = delta_t_sqrt * s # scalar to ensure the correct variance for the gaussian step

    # Boundary storage for the upper bound
    cdef int num_draws = int((max_t / delta_t) + 1)
    boundary = np.zeros(num_draws, dtype = DTYPE)
    cdef float[:] boundary_view = boundary
    
    cdef int i
    cdef float tmp

    # Precompute boundary evaluations
    if boundary_multiplicative:
        for i in range(num_draws):
            tmp = a * boundary_fun(t = i * delta_t, **boundary_params)
            if tmp > 0:
                boundary_view[i] = tmp
    else:
        for i in range(num_draws):
            tmp = a + boundary_fun(t = i * delta_t, **boundary_params)
            if tmp > 0:
                boundary_view[i] = tmp
    
    cdef float y, t, ndt_tmp
    cdef int n, ix
    cdef int m = 0
    cdef float drift_increment = 0.0
    cdef float[:] gaussian_values = draw_gaussian(num_draws) 

    # Loop over samples
    for n in range(n_samples):
        # initialize starting point
        y = ((-1) * boundary_view[0]) + (w * 2.0 * (boundary_view[0]))  # reset starting position
        
        # get drift by random displacement of v 
        drift_increment = (v + sdv * gaussian_values[m]) * delta_t
        ndt_tmp = ndt + (2 * (random_uniform() - 0.5) * dndt)
        
        # apply uniform displacement on y
        y += 2 * (random_uniform() - 0.5) * dw
        
        # increment m appropriately
        m += 1
        if m == num_draws:
                gaussian_values = draw_gaussian(num_draws)
                m = 0
        
        t = 0 # reset time
        ix = 0 # reset boundary index
        
        
        # Random walker
        while y >= (-1) * boundary_view[ix] and y <= boundary_view[ix] and t <= max_t:
            y += drift_increment + (sqrt_st * gaussian_values[m])
            t += delta_t
            ix += 1
            m += 1
            
            if m == num_draws:
                gaussian_values = draw_gaussian(num_draws)
                m = 0

        rts_view[n, 0] = t + ndt_tmp # Store rt
        choices_view[n, 0] = np.sign(y) # Store choice

    return (rts, choices,  {'v': v,
                            'a': a,
                            'w': w,
                            'ndt': ndt,
                            'dw': dw,
                            'sdv': sdv,
                            'dndt': dndt,
                            's': s,
                            **boundary_params,
                            'delta_t': delta_t,
                            'max_t': max_t,
                            'n_samples': n_samples,
                            'simulator': 'full_ddm',
                            'boundary_fun_type': boundary_fun.__name__,
                            'possible_choices': [-1, 1]})

# -------------------------------------------------------------------------------------------------

# Simulate (rt, choice) tuples from: Full DDM with flexible bounds --------------------------------
# @cythonboundscheck(False)
# @cythonwraparound(False)
def ddm_sdv(float v = 0,
            float a = 1,
            float w = 0.5,
            float ndt = 0.0,
            float sdv = 0.1,
            float s = 1,
            float delta_t = 0.001,
            float max_t = 20,
            int n_samples = 20000,
            print_info = True,
            boundary_fun = None, # function of t (and potentially other parameters) that takes in (t, *args)
            boundary_multiplicative = True,
            boundary_params = {}
            ):

    rts = np.zeros((n_samples, 1), dtype = DTYPE)
    choices = np.zeros((n_samples, 1), dtype = np.intc)

    cdef float[:, :] rts_view = rts
    cdef int[:, :] choices_view = choices

    cdef float delta_t_sqrt = sqrt(delta_t) # correct scalar so we can use standard normal samples for the brownian motion
    cdef float sqrt_st = delta_t_sqrt * s # scalar to ensure the correct variance for the gaussian step

    # Boundary storage for the upper bound
    cdef int num_draws = int((max_t / delta_t) + 1)
    boundary = np.zeros(num_draws, dtype = DTYPE)
    cdef float[:] boundary_view = boundary
    
    cdef int i
    cdef float tmp

    # Precompute boundary evaluations
    if boundary_multiplicative:
        for i in range(num_draws):
            tmp = a * boundary_fun(t = i * delta_t, **boundary_params)
            if tmp > 0:
                boundary_view[i] = tmp
    else:
        for i in range(num_draws):
            tmp = a + boundary_fun(t = i * delta_t, **boundary_params)
            if tmp > 0:
                boundary_view[i] = tmp
    
    cdef float y, t
    cdef int n, ix
    cdef int m = 0
    cdef float drift_increment = 0.0
    cdef float[:] gaussian_values = draw_gaussian(num_draws) 

    # Loop over samples
    for n in range(n_samples):
        # initialize starting point
        y = ((-1) * boundary_view[0]) + (w * 2.0 * (boundary_view[0]))  # reset starting position
        
        # get drift by random displacement of v 
        drift_increment = (v + sdv * gaussian_values[m]) * delta_t
        
        # increment m appropriately
        m += 1
        if m == num_draws:
                gaussian_values = draw_gaussian(num_draws)
                m = 0
        
        t = 0 # reset time
        ix = 0 # reset boundary index
        
        # Random walker
        while y >= (-1) * boundary_view[ix] and y <= boundary_view[ix] and t <= max_t:
            y += drift_increment + (sqrt_st * gaussian_values[m])
            t += delta_t
            ix += 1
            m += 1
            
            if m == num_draws:
                gaussian_values = draw_gaussian(num_draws)
                m = 0

        rts_view[n, 0] = t + ndt # Store rt
        choices_view[n, 0] = np.sign(y) # Store choice

    return (rts, choices,  {'v': v,
                            'a': a,
                            'w': w,
                            'ndt': ndt,
                            'sdv': sdv,
                            's': s,
                            **boundary_params,
                            'delta_t': delta_t,
                            'max_t': max_t,
                            'n_samples': n_samples,
                            'simulator': 'ddm_sdv',
                            'boundary_fun_type': boundary_fun.__name__,
                            'possible_choices': [-1, 1]})

# -------------------------------------------------------------------------------------------------

# Simulate (rt, choice) tuples from: Onstein-Uhlenbeck with flexible bounds -----------------------
# @cythonboundscheck(False)
# @cythonwraparound(False)
def ornstein_uhlenbeck(float v = 0, # drift parameter
                       float a = 1, # initial boundary separation
                       float w = 0.5, # starting point bias
                       float g = 0.1, # decay parameter
                       float ndt = 0.0,
                       float s = 1, # standard deviation
                       float delta_t = 0.001, # size of timestep
                       float max_t = 20, # maximal time in trial
                       int n_samples = 20000, # number of samples from process
                       print_info = True, # whether or not to print periodic update on number of samples generated
                       boundary_fun = None, # function of t (and potentially other parameters) that takes in (t, *args)
                       boundary_multiplicative = True,
                       boundary_params = {}
                      ):
    
    # Initializations
    rts = np.zeros((n_samples, 1), dtype = DTYPE) # rt storage
    choices = np.zeros((n_samples, 1), dtype = np.intc) # choice storage

    cdef float[:,:] rts_view = rts
    cdef int[:,:] choices_view = choices

    cdef float delta_t_sqrt = np.sqrt(delta_t) # correct scalar so we can use standard normal samples for the brownian motion
    cdef float sqrt_st = s * delta_t_sqrt

    # Boundary Storage
    cdef int num_draws = int((max_t / delta_t) + 1)
    cdef int i
    cdef float tmp
    boundary = np.zeros(num_draws, dtype = DTYPE)
    cdef float[:] boundary_view = boundary

    # Precompute boundary evaluations
    if boundary_multiplicative:
        for i in range(num_draws):
            tmp = a * boundary_fun(t = i * delta_t, **boundary_params)
            if tmp > 0:
                boundary_view[i] = tmp
    else:
        for i in range(num_draws):
            tmp = a + boundary_fun(t = i * delta_t, **boundary_params)
            if tmp > 0:
                boundary_view[i] = tmp

    cdef float y, t
    cdef int n, ix
    cdef int m = 0
    cdef float[:] gaussian_values = draw_gaussian(num_draws)

    # Loop over samples
    for n in range(n_samples):
        y = (-1) * boundary_view[0] + (w * 2 * boundary_view[0])
        t = 0
        ix = 0

        # Random walker
        while y >= (-1) * boundary_view[ix] and y <= boundary_view[ix] and t <= max_t:
            y += ((v - (g * y)) * delta_t) + sqrt_st * gaussian_values[m]
            t += delta_t
            ix += 1
            m += 1
            if m == num_draws:
                gaussian_values = draw_gaussian(num_draws)
                m = 0

        rts_view[n, 0] = ndt + t
        choices_view[n, 0] = sign(y)

        # if print_info == True:
        #     if n % 1000 == 0:
        #         print(n, ' datapoints sampled')

    return (rts, choices, {'v': v,
                           'a': a,
                           'w': w,
                           'g': g,
                           'ndt': ndt,
                           's': s,
                           **boundary_params,
                           'delta_t': delta_t,
                           'max_t': max_t,
                           'n_samples': n_samples,
                           'simulator': 'ornstein_uhlenbeck',
                           'boundary_fun_type': boundary_fun.__name__,
                           'possible_choices': [-1, 1]})
# --------------------------------------------------------------------------------------------------

# Simulate (rt, choice) tuples from: DDM WITH FLEXIBLE BOUNDARIES ------------------------------------
# @cythonboundscheck(False)
# @cythonwraparound(False)
def ddm_flexbound_seq2(float v_h = 0,
                       float v_l_1 = 0,
                       float v_l_2 = 0,
                       float a = 1,
                       float w_h = 0.5,
                       float w_l_1 = 0,
                       float w_l_2 = 0.5,
                       float ndt = 0.5,
                       float s = 1,
                       float delta_t = 0.001,
                       float max_t = 20,
                       int n_samples = 20000,
                       print_info = True,
                       boundary_fun = None, # function of t (and potentially other parameters) that takes in (t, *args)
                       boundary_multiplicative = True,
                       boundary_params = {}
                       ):

    rts = np.zeros((n_samples, 1), dtype = DTYPE)
    choices = np.zeros((n_samples, 1), dtype = np.intc)

    cdef float[:,:] rts_view = rts
    cdef int[:,:] choices_view = choices

    cdef float delta_t_sqrt = sqrt(delta_t) # correct scalar so we can use standard normal samples for the brownian motion
    cdef float sqrt_st = delta_t_sqrt * s # scalar to ensure the correct variance for the gaussian step

    # Boundary storage for the upper bound
    cdef int num_draws = int((max_t / delta_t) + 1)
    boundary = np.zeros(num_draws, dtype = DTYPE)
    cdef float[:] boundary_view = boundary
    cdef int i
    cdef float tmp

    # Precompute boundary evaluations
    if boundary_multiplicative:
        for i in range(num_draws):
            tmp = a * boundary_fun(t = i * delta_t, **boundary_params)
            if tmp > 0:
                boundary_view[i] = tmp
    else:
        for i in range(num_draws):
            tmp = a + boundary_fun(t = i * delta_t, **boundary_params)
            if tmp > 0:
                boundary_view[i] = tmp

    cdef float y_h, t, y_l
    cdef int n, ix
    cdef int m = 0
    cdef float[:] gaussian_values = draw_gaussian(num_draws)

    # Loop over samples
    for n in range(n_samples):
        t = 0 # reset time
        ix = 0 # reset boundary index

        # Random walker 1
        y_h = (-1) * boundary_view[0] + (w_h * 2 * (boundary_view[0]))  # reset starting position 
        while y_h >= (-1) * boundary_view[ix] and y_h <= boundary_view[ix] and t <= max_t:
            y_h += (v_h * delta_t) + (sqrt_st * gaussian_values[m])
            t += delta_t
            ix += 1
            m += 1
            if m == num_draws:
                gaussian_values = draw_gaussian(num_draws)
                m = 0
  
        # If we are already at maximum t, to generate a choice we just sample from a bernoulli
        if t >= max_t:
            if random_uniform() > 0.5:
                choices_view[n, 0] = choices_view[n, 0] + 1
        else:
            if sign(y_h) < 0: # Store intermediate choice
                choices_view[n, 0] = 0 
                
                # In case boundary is negative already, we flip a coin with bias determined by w_l_ parameter
                if boundary_view[ix] <= 0:
                    if random_uniform() < w_l_1:
                        choices_view[n, 0] += 1
                else:
                    y_l = (-1) * boundary_view[ix] + (w_l_1 * 2 * (boundary_view[ix])) 
                    v_l = v_l_1
            else:
                choices_view[n, 0] = 2
                
                # In case boundary is negative already, we flip a coin with bias determined by w_l_ parameter
                if boundary_view[ix] <= 0:
                    if random_uniform() < w_l_2:
                        choices_view[n, 0] += 1
                else:
                    y_l = (-1) * boundary_view[ix] + (w_l_2 * 2 * (boundary_view[ix])) 
                    v_l = v_l_2

        # Random walker 2
        while y_l >= (-1) * boundary_view[ix] and y_l <= boundary_view[ix] and t <= max_t:
            y_l += (v_l * delta_t) + (sqrt_st * gaussian_values[m])
            t += delta_t
            ix += 1
            m += 1
            if m == num_draws:
                gaussian_values = draw_gaussian(num_draws)
                m = 0

        rts_view[n, 0] = t + ndt
        if sign(y_l) >= 0: # store choice update
            choices_view[n, 0] += 1

    return (rts, choices,  {'v_h': v_h,
                            'v_l_1': v_l_1,
                            'v_l_2': v_l_2,
                            'a': a,
                            'w_h': w_h,
                            'w_l_1': w_l_1,
                            'w_l_2': w_l_2,
                            'ndt': ndt,
                            's': s,
                            **boundary_params,
                            'delta_t': delta_t,
                            'max_t': max_t,
                            'n_samples': n_samples,
                            'simulator': 'ddm_flexbound',
                            'boundary_fun_type': boundary_fun.__name__,
                            'possible_choices': [0, 1, 2, 3]})
# -----------------------------------------------------------------------------------------------

# Simulate (rt, choice) tuples from: DDM WITH FLEXIBLE BOUNDARIES ------------------------------------
# @cythonboundscheck(False)
# @cythonwraparound(False)
def ddm_flexbound_par2(float v_h = 0, 
                       float v_l_1 = 0,
                       float v_l_2 = 0,
                       float a = 1,
                       float w_h = 0.5,
                       float w_l_1 = 0.5,
                       float w_l_2 = 0.5,
                       float ndt = 0.5,
                       float s = 1,
                       float delta_t = 0.001,
                       float max_t = 20,
                       int n_samples = 20000,
                       print_info = True,
                       boundary_fun = None, # function of t (and potentially other parameters) that takes in (t, *args)
                       boundary_multiplicative = True,
                       boundary_params = {}
                       ):

    rts = np.zeros((n_samples, 1), dtype = DTYPE)
    choices = np.zeros((n_samples, 1), dtype = np.intc)

    cdef float[:, :] rts_view = rts
    cdef int[:, :] choices_view = choices

    cdef float delta_t_sqrt = sqrt(delta_t) # correct scalar so we can use standard normal samples for the brownian motion
    cdef float sqrt_st = delta_t_sqrt * s # scalar to ensure the correct variance for the gaussian step

    # Boundary storage for the upper bound
    cdef int num_draws = int((max_t / delta_t) + 1)
    boundary = np.zeros(num_draws, dtype = DTYPE)
    cdef float[:] boundary_view = boundary
    cdef int i
    cdef float tmp

    # Precompute boundary evaluations
    if boundary_multiplicative:
        for i in range(num_draws):
            tmp = a * boundary_fun(t = i * delta_t, **boundary_params)
            if tmp > 0:
                boundary_view[i] = tmp
    else:
        for i in range(num_draws):
            tmp = a + boundary_fun(t = i * delta_t, **boundary_params)
            if tmp > 0:
                boundary_view[i] = tmp

    cdef float y_h, y_l, v_l, t_h, t_l
    cdef int n, ix
    cdef int m = 0
    cdef float[:] gaussian_values = draw_gaussian(num_draws)

    # Loop over samples
    for n in range(n_samples):
        t_h = 0 # reset time high dimension
        t_l = 0 # reset time low dimension
        ix = 0 # reset boundary index

        # Initialize walkers
        y_h = (-1) * boundary_view[0] + (w_h * 2 * (boundary_view[0])) 

        # Random walks until y_h hits bound
        while y_h >= (-1) * boundary_view[ix] and y_h <= boundary_view[ix] and t_h <= max_t:
            y_h += (v_h * delta_t) + (sqrt_st * gaussian_values[m])
            t_h += delta_t
            ix += 1
            m += 1
            if m == num_draws:
                gaussian_values = draw_gaussian(num_draws)
                m = 0

        if sign(y_h) < 0: # Store intermediate choice
            choices_view[n, 0] = 0 
            y_l = (-1) * boundary_view[0] + (w_l_1 * 2 * (boundary_view[0])) 
            v_l = v_l_1
        
        else:
            choices_view[n, 0] = 2
            y_l = (-1) * boundary_view[0] + (w_l_2 * 2 * (boundary_view[0])) 
            v_l = v_l_2

        # Random walks until the y_l corresponding to y_h hits bound
        ix = 0
        while y_l >= (-1) * boundary_view[ix] and y_l <= boundary_view[ix] and t_l <= max_t:
            y_l += (v_l * delta_t) + (sqrt_st * gaussian_values[m])
            t_l += delta_t
            ix += 1
            m += 1
            if m == num_draws:
                gaussian_values = draw_gaussian(num_draws)
                m = 0

        rts_view[n, 0] = fmax(t_h, t_l) + ndt

        if sign(y_l) >= 0: # store choice update
            choices_view[n, 0] = choices_view[n, 0] + 1

    return (rts, choices,  {'v_h': v_h,
                            'v_l_1': v_l_1,
                            'v_l_2': v_l_2,
                            'a': a,
                            'w_h': w_h,
                            'w_l_1': w_l_1,
                            'w_l_2': w_l_2,
                            'ndt': ndt,
                            's': s,
                            **boundary_params,
                            'delta_t': delta_t,
                            'max_t': max_t,
                            'n_samples': n_samples,
                            'simulator': 'ddm_flexbound',
                            'boundary_fun_type': boundary_fun.__name__,
                            'possible_choices': [0, 1, 2, 3]})
# -----------------------------------------------------------------------------------------------

# Simulate (rt, choice) tuples from: DDM WITH FLEXIBLE BOUNDARIES ------------------------------------
# @cythonboundscheck(False)
# @cythonwraparound(False)
def ddm_flexbound_mic2(float v_h = 0, 
                       float v_l_1 = 0,
                       float v_l_2 = 0,
                       float a = 1,
                       float w_h = 0.5,
                       float w_l_1 = 0.5,
                       float w_l_2 = 0.5,
                       float d = 0.5, # d for 'dampen' effect on drift parameter
                       float ndt = 0.5,
                       float s = 1,
                       float delta_t = 0.001,
                       float max_t = 20,
                       int n_samples = 20000,
                       print_info = True,
                       boundary_fun = None, # function of t (and potentially other parameters) that takes in (t, *args)
                       boundary_multiplicative = True,
                       boundary_params = {}
                       ):

    rts = np.zeros((n_samples, 1), dtype = DTYPE)
    choices = np.zeros((n_samples, 1), dtype = np.intc)

    cdef float[:, :] rts_view = rts
    cdef int[:, :] choices_view = choices

    cdef float delta_t_sqrt = sqrt(delta_t) # correct scalar so we can use standard normal samples for the brownian motion
    cdef float sqrt_st = delta_t_sqrt * s # scalar to ensure the correct variance for the gaussian step

    # Boundary storage for the upper bound
    cdef int num_draws = int((max_t / delta_t) + 1)
    boundary = np.zeros(num_draws, dtype = DTYPE)
    cdef float[:] boundary_view = boundary
    cdef int i
    cdef float tmp

    # Y particle trace
    bias_trace = np.zeros(num_draws, dtype = DTYPE)
    cdef float[:] bias_trace_view = bias_trace

    # Precompute boundary evaluations
    if boundary_multiplicative:
        for i in range(num_draws):
            tmp = a * boundary_fun(t = i * delta_t, **boundary_params)
            if tmp > 0:
                boundary_view[i] = tmp
    else:
        for i in range(num_draws):
            tmp = a + boundary_fun(t = i * delta_t, **boundary_params)
            if tmp > 0:
                boundary_view[i] = tmp

    cdef float y_h, y_l, v_l, t_h, t_l
    cdef int n, ix, ix_tmp
    cdef int m = 0
    cdef float[:] gaussian_values = draw_gaussian(num_draws)

    # Loop over samples
    for n in range(n_samples):
        t_h = 0 # reset time high dimension
        t_l = 0 # reset time low dimension
        ix = 0 # reset boundary index

        # Initialize walkers
        y_h = (-1) * boundary_view[0] + (w_h * 2 * (boundary_view[0])) 
        bias_trace_view[0] = ((boundary_view[0] - y_h) / (2 * boundary_view[0]))

        # Random walks until y_h hits bound
        while y_h >= (-1) * boundary_view[ix] and y_h <= boundary_view[ix] and t_h <= max_t:
            y_h += (v_h * delta_t) + (sqrt_st * gaussian_values[m])
            bias_trace_view[ix] = ((boundary_view[ix] - y_h) / (2 * boundary_view[ix]))
            t_h += delta_t
            ix += 1
            m += 1
            if m == num_draws:
                gaussian_values = draw_gaussian(num_draws)
                m = 0

        if sign(y_h) < 0: # Store intermediate choice
            choices_view[n, 0] = 0 
            y_l = (- 1) * boundary_view[0] + (w_l_1 * 2 * (boundary_view[0])) 
            v_l = v_l_1
            ix_tmp = ix + 1

            while ix_tmp < num_draws:
                bias_trace_view[ix_tmp] = 1.0
                ix_tmp += 1

            # We need to reverse the bias if we took the lower choice
            ix_tmp = 0 
            while ix_tmp < num_draws:
                bias_trace_view[ix_tmp] = bias_trace_view[ix_tmp]
                ix_tmp += 1

        else:
            choices_view[n, 0] = 2
            y_l = (- 1) * boundary_view[0] + (w_l_2 * 2 * (boundary_view[0])) 
            v_l = v_l_2
            ix_tmp = ix + 1
            while ix_tmp < num_draws:
                bias_trace_view[ix_tmp] = 0.0
                ix_tmp += 1

        # Random walks until the y_l corresponding to y_h hits bound
        ix = 0
        while y_l >= (-1) * boundary_view[ix] and y_l <= boundary_view[ix] and t_l <= max_t:
            #y_l += (bias_trace_view[ix] * v_l * delta_t) + (sqrt_st * gaussian_values[m])
            y_l += (v_l * (1.0 - bias_trace_view[ix] * d) * delta_t) + (sqrt_st * gaussian_values[m])
            t_l += delta_t
            ix += 1
            m += 1
            if m == num_draws:
                gaussian_values = draw_gaussian(num_draws)
                m = 0

        rts_view[n, 0] = fmax(t_h, t_l) + ndt

        if sign(y_l) >= 0: # store choice update
            choices_view[n, 0] = choices_view[n, 0] + 1

    return (rts, choices,  {'v_h': v_h,
                            'v_l_1': v_l_1,
                            'v_l_2': v_l_2,
                            'a': a,
                            'w_h': w_h,
                            'w_l_1': w_l_1,
                            'w_l_2': w_l_2,
                            'ndt': ndt,
                            's': s,
                            **boundary_params,
                            'delta_t': delta_t,
                            'max_t': max_t,
                            'n_samples': n_samples,
                            'simulator': 'ddm_flexbound',
                            'boundary_fun_type': boundary_fun.__name__,
                            'possible_choices': [0, 1, 2, 3]})
# -----------------------------------------------------------------------------------------------

# Simulate (rt, choice) tuples from: RACE MODEL WITH N SAMPLES ----------------------------------

# Check if any of the particles in the race model have crossed
# @cythonboundscheck(False)
# @cythonwraparound(False)

# Function that checks boundary crossing of particles
cdef bint check_finished(float[:] particles, float boundary):
    cdef int i,n
    n = particles.shape[0]
    for i in range(n):
        if particles[i] > boundary:
            return True
    return False

def test_check():
    # Quick sanity check for the check_finished function
    temp = np.random.normal(0,1, 10).astype(DTYPE)
    cdef float[:] temp_view = temp
    start = time()
    [check_finished(temp_view, 3) for _ in range(1000000)]
    print(check_finished(temp_view, 3))
    end = time()
    print("cython check: {}".format(start - end))
    start = time()
    [(temp > 3).any() for _ in range(1000000)]
    end = time()
    print("numpy check: {}".format(start - end))

# @cythonboundscheck(False)
# @cythonwraparound(False)
def race_model(v = np.array([0, 0, 0], dtype = DTYPE), # np.array expected, one column of floats
               float a = 1, # initial boundary separation
               w = np.array([0, 0, 0], dtype = DTYPE), # np.array expected, one column of floats
               float ndt = 1, # for now we we don't allow ndt by choice
               #ndt = np.array([0.0, 0.0, 0.0], dtype = DTYPE),
               s = np.array([1, 1, 1], dtype = DTYPE), # np.array expected, one column of floats
               float delta_t = 0.001, # time increment step
               float max_t = 20, # maximum rt allowed
               int n_samples = 2000, 
               print_info = True,
               boundary_fun = None,
               boundary_multiplicative = True,
               boundary_params = {}):

    # Initializations
    cdef float[:] v_view = v
    cdef float[:] w_view = w
    cdef float delta_t_sqrt = sqrt(delta_t)
    sqrt_st = delta_t_sqrt * s
    cdef float[:] sqrt_st_view = sqrt_st

    cdef int n_particles = len(v)
    rts = np.zeros((n_samples, 1), dtype = DTYPE)
    cdef float[:,:] rts_view = rts
    choices = np.zeros((n_samples, 1), dtype = np.intc)
    cdef int[:,:] choices_view = choices
    cdef float [:] particles_view

    # Boundary storage
    cdef int num_steps = int((max_t / delta_t) + 1)
    cdef int i
    cdef float tmp
    boundary = np.zeros(num_steps, dtype = DTYPE)
    cdef float[:] boundary_view = boundary

    # Precompute boundary evaluations
    if boundary_multiplicative:
        for i in range(num_steps):
            tmp = a * boundary_fun(t = i * delta_t, **boundary_params)
            if tmp > 0:
                boundary_view[i] = tmp
    else:
        for i in range(num_steps):
            tmp = a + boundary_fun(t = i * delta_t, **boundary_params)
            if tmp > 0:
                boundary_view[i] = tmp
    
    # Initialize variables needed for for loop 
    cdef float t
    cdef int n, ix, j
    cdef int m = 0

    cdef int num_draws = num_steps * n_particles
    cdef float[:] gaussian_values = draw_gaussian(num_draws)

    # Loop over samples
    for n in range(n_samples):
        particles = w * boundary_view[0] # Reset particle starting points
        particles_view = particles
        t = 0 # reset time
        ix = 0

        # Random walker
        while not check_finished(particles_view, boundary_view[ix]) and t <= max_t:
            for j in range(n_particles):
                particles_view[j] += (v_view[j] * delta_t) + sqrt_st_view[j] * gaussian_values[m]
                m += 1
                if m == num_draws:
                    m = 0
                    gaussian_values = draw_gaussian(num_draws)
            t += delta_t
            ix += 1

        choices_view[n, 0] = np.argmax(particles)
        #rts_view[n, 0] = t + ndt[choices_view[n, 0]]
        rts_view[n , 0] = t + ndt # for now no ndt per choice option

    # Create some dics
    v_dict = {}
    w_dict = {}
    #ndt_dict = {}
    for i in range(n_particles):
        v_dict['v_' + str(i)] = v[i]
        w_dict['w_' + str(i)] = w[i]
        #ndt_dict['ndt_' + str(i)] = ndt[i] # for now no ndt by choice

    return (rts, choices, {**v_dict,
                           'a': a, 
                           **w_dict,
                           'ndt': ndt,
                           # **ndt_dict, # for now no ndt by choice
                           's': s,
                           **boundary_params,
                           'delta_t': delta_t,
                           'max_t': max_t,
                           'n_samples': n_samples,
                           'simulator': 'race_model',
                           'boundary_fun_type': boundary_fun.__name__,
                           'possible_choices': list(np.arange(0, n_particles, 1))})
# -------------------------------------------------------------------------------------------------
# @cythonboundscheck(False)
# @cythonwraparound(False)

# Simulate (rt, choice) tuples from: Leaky Competing Accumulator Model -----------------------------
def lca(v = np.array([0, 0, 0], dtype = DTYPE), # drift parameters (np.array expect: one column of floats)
        float a = 1.01, # criterion height
        w = np.array([0, 0, 0], dtype = DTYPE), # initial bias parameters (np.array expect: one column of floats)
        float g = 0.0, # decay parameter
        float b = 1.0, # inhibition parameter
        float ndt = 1.0,
        float s = 1.0, # variance (can be one value or np.array of size as v and w)
        float delta_t = 0.001, # time-step size in simulator
        float max_t = 20, # maximal time
        int n_samples = 2000, # number of samples to produce
        print_info = True, # whether or not to periodically report the number of samples generated thus far
        boundary_fun = None, # function of t (and potentially other parameters) that takes in (t, *args)
        boundary_multiplicative = True,
        boundary_params = {}):

    # Initializations
    cdef int n_particles = v.shape[0]
    
    rts = np.zeros((n_samples, 1), dtype = DTYPE)
    cdef float[:,:] rts_view = rts
    
    choices = np.zeros((n_samples, 1), dtype = np.intc)
    cdef int[:,:] choices_view = choices
    
    cdef float[:] v_view = v
    cdef float[:] w_view = w
    
    particles = np.zeros(n_particles, dtype = DTYPE)
    cdef float[:] particles_view = particles
    
    particles_reduced_sum = np.zeros(n_particles, dtype = DTYPE)
    cdef float[:] particles_reduced_sum_view = particles_reduced_sum
    
    cdef float delta_t_sqrt = sqrt(delta_t)
    cdef float sqrt_st = s * delta_t_sqrt
    
    cdef int n, i, ix
    cdef int m = 0
    cdef float t, particles_sum
    
    # Boundary storage                                                             
    cdef int num_steps = int((max_t / delta_t) + 2)
    cdef float tmp
    boundary = np.zeros(num_steps, dtype = DTYPE)
    cdef float[:] boundary_view = boundary

    # Precompute boundary evaluations
    if boundary_multiplicative:
        for i in range(num_steps):
            tmp = a * boundary_fun(t = i * delta_t, **boundary_params)
            if tmp > 0:
                boundary_view[i] = tmp
    else:
        for i in range(num_steps):
            tmp = a + boundary_fun(t = i * delta_t, **boundary_params)
            if tmp > 0:
                boundary_view[i] = tmp

    cdef int num_draws = num_steps * n_particles
    cdef float[:] gaussian_values = draw_gaussian(num_draws)

    for n in range(n_samples):
        # Reset particle starting points
        for i in range(n_particles):
            particles_view[i] = w_view[i] * boundary_view[0]
        
        t = 0.0 # reset time
        ix = 0 # reset boundary index

        while not check_finished(particles_view, boundary_view[ix]) and t <= max_t:
            # calculate current sum over particle positions
            particles_sum = csum(particles_view)
            # update particle positions 
            for i in range(n_particles):
                particles_reduced_sum_view[i] = (- 1) * particles_view[i] + particles_sum
                particles_view[i] += ((v_view[i] - (g * particles_view[i]) - \
                        (b * particles_reduced_sum_view[i])) * delta_t) + (sqrt_st * gaussian_values[m])
                particles_view[i] = fmax(0.0, particles_view[i])
                m += 1
                
                if m == num_draws:
                    gaussian_values = draw_gaussian(num_draws)
                    m = 0
            
            t += delta_t # increment time
            ix += 1 # increment boundary index
            
        choices_view[n, 0] = particles.argmax() # store choices for sample n
        rts_view[n, 0] = t + ndt # ndt[choices_view[n, 0]] # store reaction time for sample n
        
    # Create some dics
    v_dict = {}
    w_dict = {}
    #ndt_dict = {}
    
    for i in range(n_particles):
        v_dict['v_' + str(i)] = v[i]
        w_dict['w_' + str(i)] = w[i]
        #ndt_dict['ndt_' + str(i)] = ndt[i]

    return (rts, choices, {**v_dict,
                           'a': a,
                           **w_dict,
                           'g': g,
                           'b': b,
                           'ndt': ndt,
                           's': s,
                           **boundary_params,
                           'delta_t': delta_t,
                           'max_t': max_t,
                           'n_samples': n_samples,
                           'simulator' : 'lca',
                           'boundary_fun_type': boundary_fun.__name__,
                           'possible_choices': list(np.arange(0, n_particles, 1))})



# Alternative boundary specification
# Simulate (rt, choice) tuples from: DDM WITH FLEXIBLE BOUNDARIES ------------------------------------
#@cythonboundscheck(False)
#@cythonwraparound(False)
def ddm_flexbound_pre(float v = 0,
                      float a = 1,
                      float w = 0.5,
                      float ndt = 0.0,
                      float s = 1,
                      float delta_t = 0.001,
                      float max_t = 20,
                      int n_samples = 20000,
                      print_info = True,
                      boundary_fun = None, # function of t (and potentially other parameters) that takes in (t, *args)
                      boundary_multiplicative = True,
                      boundary_params = {}
                      ):

    rts = np.zeros((n_samples, 1), dtype = DTYPE)
    choices = np.zeros((n_samples, 1), dtype = np.intc)
    t_s = np.arange(0, max_t + delta_t, delta_t).astype(DTYPE)
    boundary = np.zeros(t_s.shape, dtype = DTYPE)
    
    cdef float[:,:] rts_view = rts
    cdef int[:,:] choices_view = choices
    cdef float delta_t_sqrt = sqrt(delta_t) # correct scalar so we can use standard normal samples for the brownian motion
    cdef float sqrt_st = delta_t_sqrt * s # scalar to ensure the correct variance for the gaussian step

    # Boundary storage for the upper bound
    cdef int num_draws = int((max_t / delta_t) + 1)
    
    #boundary = np.zeros(num_draws, dtype = DTYPE)
    #cdef float[:] boundary_view = boundary
    #cdef int i
    #cdef float tmp

    # Precompute boundary evaluations
    if boundary_multiplicative:
        # print(a)
        boundary[:] = np.multiply(a, boundary_fun(t = t_s, **boundary_params)).astype(DTYPE)
    else:
        # print(a)
        boundary[:] = np.add(a, boundary_fun(t = t_s, **boundary_params)).astype(DTYPE)
    
#     if boundary_multiplicative:
#         for i in range(num_draws):
#             tmp = a * boundary_fun(t = i * delta_t, **boundary_params)
#             if tmp > 0:
#                 boundary_view[i] = tmp
#     else:
#         for i in range(num_draws):
#             tmp = a + boundary_fun(t = i * delta_t, **boundary_params)
#             if tmp > 0:
#                 boundary_view[i] = tmp

    cdef float y, t
    cdef int n, ix
    cdef int m = 0
    cdef float[:] gaussian_values = draw_gaussian(num_draws)
    cdef float[:] boundary_view = boundary
    #ixs = np.zeros(n_samples, dtype = np.intc)
    #cdef int[:] ixs_view = ixs
    
    # Loop over samples
    for n in range(n_samples):
        y = (-1) * boundary_view[0] + (w * 2 * (boundary_view[0]))  # reset starting position 
        t = 0 # reset time
        ix = 0 # reset boundary index

        # Random walker
        while y >= (-1) * boundary_view[ix] and y <= boundary_view[ix] and t <= max_t:
            y += (v * delta_t) + (sqrt_st * gaussian_values[m])
            t += delta_t
            ix += 1
            m += 1
            if m == num_draws:
                gaussian_values = draw_gaussian(num_draws)
                m = 0

        rts_view[n, 0] = t + ndt # Store rt
        choices_view[n, 0] = sign(y) # Store choice
        #ixs_view[n] = ix

    return (rts, choices, {'v': v,
                                'a': a,
                                'w': w,
                                'ndt': ndt,
                                's': s,
                                **boundary_params,
                                'delta_t': delta_t,
                                'max_t': max_t,
                                'n_samples': n_samples,
                                'simulator': 'ddm_flexbound',
                                'boundary_fun_type': boundary_fun.__name__,
                                'possible_choices': [-1, 1]})
# # ----------------------------------------------------------------------------------------------------


# Simulate (rt, choice) tuples from: Onstein-Uhlenbeck with flexible bounds -----------------------
# @cythonboundscheck(False)
# @cythonwraparound(False)
def ornstein_uhlenbeck_pre(float v = 0, # drift parameter
                           float a = 1, # initial boundary separation
                           float w = 0.5, # starting point bias
                           float g = 0.1, # decay parameter
                           float ndt = 0.0,
                           float s = 1, # standard deviation
                           float delta_t = 0.001, # size of timestep
                           float max_t = 20, # maximal time in trial
                           int n_samples = 20000, # number of samples from process
                           print_info = True, # whether or not to print periodic update on number of samples generated
                           boundary_fun = None, # function of t (and potentially other parameters) that takes in (t, *args)
                           boundary_multiplicative = True,
                           boundary_params = {}
                          ):
    
    # Initializations
    rts = np.zeros((n_samples, 1), dtype = DTYPE) # rt storage
    choices = np.zeros((n_samples, 1), dtype = np.intc) # choice storage
    t_s = np.arange(0, max_t + delta_t, delta_t).astype(DTYPE)
    boundary = np.zeros(t_s.shape, dtype = DTYPE)

    cdef float[:,:] rts_view = rts
    cdef int[:,:] choices_view = choices

    cdef float delta_t_sqrt = np.sqrt(delta_t) # correct scalar so we can use standard normal samples for the brownian motion
    cdef float sqrt_st = s * delta_t_sqrt

    # Boundary Storage
    cdef int num_draws = int((max_t / delta_t) + 1)
    
    # Precompute boundary evaluations
    if boundary_multiplicative:
        # print(a)
        boundary[:] = np.multiply(a, boundary_fun(t = t_s, **boundary_params)).astype(DTYPE)
    else:
        # print(a)
        boundary[:] = np.add(a, boundary_fun(t = t_s, **boundary_params)).astype(DTYPE)
    
    cdef float y, t
    cdef int n, ix
    cdef int m = 0
    cdef float[:] gaussian_values = draw_gaussian(num_draws)
    cdef float[:] boundary_view = boundary


    # Loop over samples
    for n in range(n_samples):
        y = (-1) * boundary_view[0] + (w * 2 * boundary_view[0])
        t = 0
        ix = 0

        # Random walker
        while y >= (-1) * boundary_view[ix] and y <= boundary_view[ix] and t <= max_t:
            y += ((v - (g * y)) * delta_t) + sqrt_st * gaussian_values[m]
            t += delta_t
            ix += 1
            m += 1
            if m == num_draws:
                gaussian_values = draw_gaussian(num_draws)
                m = 0

        rts_view[n, 0] = ndt + t
        choices_view[n, 0] = sign(y)

        # if print_info == True:
        #     if n % 1000 == 0:
        #         print(n, ' datapoints sampled')

    return (rts, choices, {'v': v,
                           'a': a,
                           'w': w,
                           'g': g,
                           'ndt': ndt,
                           's': s,
                           **boundary_params,
                           'delta_t': delta_t,
                           'max_t': max_t,
                           'n_samples': n_samples,
                           'simulator': 'ornstein_uhlenbeck',
                           'boundary_fun_type': boundary_fun.__name__,
                           'possible_choices': [-1, 1]})
# --------------------------------------------------------------------------------------------------

# Simulate (rt, choice) tuples from: Levy Flight with Flex Bound -------------------------------------
# @cythonboundscheck(False)
# @cythonwraparound(False)
def levy_flexbound_pre(float v = 0,
                       float a = 1,
                       float w = 0.5,
                       float alpha_diff = 1,
                       float ndt = 0.0,
                       float s = 1,
                       float delta_t = 0.001,
                       float max_t = 20,
                       int n_samples = 20000,
                       print_info = True,
                       boundary_fun = None, # function of t (and potentially other parameters) that takes in (t, *args)
                       boundary_multiplicative = True,
                       boundary_params = {}
                       ):

    rts = np.zeros((n_samples, 1), dtype = DTYPE)
    choices = np.zeros((n_samples, 1), dtype = np.intc)
    t_s = np.arange(0, max_t + delta_t, delta_t).astype(DTYPE)

    cdef float[:,:] rts_view = rts
    cdef int[:,:] choices_view = choices

    cdef float delta_t_alpha = pow(delta_t, 1.0 / alpha_diff) # correct scalar so we can use standard normal samples for the brownian motion
    #cdef float sqrt_st = delta_t_sqrt * s # scalar to ensure the correct variance for the gaussian step

    # Boundary storage for the upper bound
    cdef int num_draws = int((max_t / delta_t) + 1)
    boundary = np.zeros(t_s.shape, dtype = DTYPE)    
    
    cdef int i
    cdef float tmp

    # Precompute boundary evaluations
    if boundary_multiplicative:
        # print(a)
        boundary[:] = np.multiply(a, boundary_fun(t = t_s, **boundary_params)).astype(DTYPE)
    else:
        # print(a)
        boundary[:] = np.add(a, boundary_fun(t = t_s, **boundary_params)).astype(DTYPE)

    cdef float[:] boundary_view = boundary
    cdef float y, t
    cdef int n, ix
    cdef int m = 0
    cdef float[:] gaussian_values = draw_random_stable(num_draws, alpha_diff)

    # Loop over samples
    for n in range(n_samples):
        y = (-1) * boundary_view[0] + (w * 2 * (boundary_view[0]))  # reset starting position 
        t = 0 # reset time
        ix = 0 # reset boundary index

        # Random walker
        while y >= (-1) * boundary_view[ix] and y <= boundary_view[ix] and t <= max_t:
            y += (v * delta_t) + (delta_t_alpha * gaussian_values[m])
            t += delta_t
            ix += 1
            m += 1
            if m == num_draws:
                gaussian_values = draw_random_stable(num_draws, alpha_diff)
                m = 0

        rts_view[n, 0] = t + ndt # Store rt
        choices_view[n, 0] = sign(y) # Store choice

    return (rts, choices,  {'v': v,
                            'a': a,
                            'w': w,
                            'ndt': ndt,
                            'alpha_diff': alpha_diff,
                            's': s,
                            **boundary_params,
                            'delta_t': delta_t,
                            'max_t': max_t,
                            'n_samples': n_samples,
                            'simulator': 'levy_flexbound',
                            'boundary_fun_type': boundary_fun.__name__,
                            'possible_choices': [-1, 1]})
# -------------------------------------------------------------------------------------------------

# Simulate (rt, choice) tuples from: Full DDM with flexible bounds --------------------------------
# @cythonboundscheck(False)
# @cythonwraparound(False)
def full_ddm_pre(float v = 0,
                 float a = 1,
                 float w = 0.5,
                 float ndt = 0.0,
                 float dw = 0.05,
                 float sdv = 0.1,
                 float dndt = 0.0,
                 float s = 1,
                 float delta_t = 0.001,
                 float max_t = 20,
                 int n_samples = 20000,
                 print_info = True,
                 boundary_fun = None, # function of t (and potentially other parameters) that takes in (t, *args)
                 boundary_multiplicative = True,
                 boundary_params = {}
                 ):

    rts = np.zeros((n_samples, 1), dtype = DTYPE)
    choices = np.zeros((n_samples, 1), dtype = np.intc)
    t_s = np.arange(0, max_t + delta_t, delta_t).astype(DTYPE)
    boundary = np.zeros(t_s.shape, dtype = DTYPE)    

    cdef float[:, :] rts_view = rts
    cdef int[:, :] choices_view = choices

    cdef float delta_t_sqrt = sqrt(delta_t) # correct scalar so we can use standard normal samples for the brownian motion
    cdef float sqrt_st = delta_t_sqrt * s # scalar to ensure the correct variance for the gaussian step

    # Boundary storage for the upper bound
    cdef int num_draws = int((max_t / delta_t) + 1)
    
    cdef int i
    cdef float tmp

    
    # Precompute boundary evaluations
    if boundary_multiplicative:
        # print(a)
        boundary[:] = np.multiply(a, boundary_fun(t = t_s, **boundary_params)).astype(DTYPE)
    else:
        # print(a)
        boundary[:] = np.add(a, boundary_fun(t = t_s, **boundary_params)).astype(DTYPE)
        
    cdef float[:] boundary_view = boundary
    cdef float y, t, ndt_tmp
    cdef int n, ix
    cdef int m = 0
    cdef float drift_increment = 0.0
    cdef float[:] gaussian_values = draw_gaussian(num_draws) 

    # Loop over samples
    for n in range(n_samples):
        # initialize starting point
        y = ((-1) * boundary_view[0]) + (w * 2.0 * (boundary_view[0]))  # reset starting position
        
        # get drift by random displacement of v 
        drift_increment = (v + sdv * gaussian_values[m]) * delta_t
        ndt_tmp = ndt + (2 * (random_uniform() - 0.5) * dndt)
        
        # apply uniform displacement on y
        y += 2 * (random_uniform() - 0.5) * dw
        
        # increment m appropriately
        m += 1
        if m == num_draws:
                gaussian_values = draw_gaussian(num_draws)
                m = 0
        
        t = 0 # reset time
        ix = 0 # reset boundary index
        
        # Random walker
        while y >= (-1) * boundary_view[ix] and y <= boundary_view[ix] and t <= max_t:
            y += drift_increment + (sqrt_st * gaussian_values[m])
            t += delta_t
            ix += 1
            m += 1
            
            if m == num_draws:
                gaussian_values = draw_gaussian(num_draws)
                m = 0

        rts_view[n, 0] = t + ndt_tmp # Store rt
        choices_view[n, 0] = np.sign(y) # Store choice

    return (rts, choices,  {'v': v,
                            'a': a,
                            'w': w,
                            'ndt': ndt,
                            'dw': dw,
                            'sdv': sdv,
                            'dndt': dndt,
                            's': s,
                            **boundary_params,
                            'delta_t': delta_t,
                            'max_t': max_t,
                            'n_samples': n_samples,
                            'simulator': 'full_ddm',
                            'boundary_fun_type': boundary_fun.__name__,
                            'possible_choices': [-1, 1]})

# -------------------------------------------------------------------------------------------------

# Simulate (rt, choice) tuples from: Full DDM with flexible bounds --------------------------------
# @cythonboundscheck(False)
# @cythonwraparound(False)
def ddm_sdv_pre(float v = 0,
                float a = 1,
                float w = 0.5,
                float ndt = 0.0,
                float sdv = 0.1,
                float s = 1,
                float delta_t = 0.001,
                float max_t = 20,
                int n_samples = 20000,
                print_info = True,
                boundary_fun = None, # function of t (and potentially other parameters) that takes in (t, *args)
                boundary_multiplicative = True,
                boundary_params = {}
                ):

    rts = np.zeros((n_samples, 1), dtype = DTYPE)
    choices = np.zeros((n_samples, 1), dtype = np.intc)
    t_s = np.arange(0, max_t + delta_t, delta_t).astype(DTYPE)
    boundary = np.zeros(t_s.shape, dtype = DTYPE)      

    cdef float[:, :] rts_view = rts
    cdef int[:, :] choices_view = choices

    cdef float delta_t_sqrt = sqrt(delta_t) # correct scalar so we can use standard normal samples for the brownian motion
    cdef float sqrt_st = delta_t_sqrt * s # scalar to ensure the correct variance for the gaussian step

    # Boundary storage for the upper bound
    cdef int num_draws = int((max_t / delta_t) + 1)
    
    cdef int i
    cdef float tmp

    # Precompute boundary evaluations
    if boundary_multiplicative:
        # print(a)
        boundary[:] = np.multiply(a, boundary_fun(t = t_s, **boundary_params)).astype(DTYPE)
    else:
        # print(a)
        boundary[:] = np.add(a, boundary_fun(t = t_s, **boundary_params)).astype(DTYPE)
    
    cdef float[:] boundary_view = boundary
    cdef float y, t
    cdef int n, ix
    cdef int m = 0
    cdef float drift_increment = 0.0
    cdef float[:] gaussian_values = draw_gaussian(num_draws) 

    # Loop over samples
    for n in range(n_samples):
        # initialize starting point
        y = ((-1) * boundary_view[0]) + (w * 2.0 * (boundary_view[0]))  # reset starting position
        
        # get drift by random displacement of v 
        drift_increment = (v + sdv * gaussian_values[m]) * delta_t
        
        # increment m appropriately
        m += 1
        if m == num_draws:
                gaussian_values = draw_gaussian(num_draws)
                m = 0
        
        t = 0 # reset time
        ix = 0 # reset boundary index
        
        # Random walker
        while y >= (-1) * boundary_view[ix] and y <= boundary_view[ix] and t <= max_t:
            y += drift_increment + (sqrt_st * gaussian_values[m])
            t += delta_t
            ix += 1
            m += 1
            
            if m == num_draws:
                gaussian_values = draw_gaussian(num_draws)
                m = 0

        rts_view[n, 0] = t + ndt # Store rt
        choices_view[n, 0] = np.sign(y) # Store choice

    return (rts, choices,  {'v': v,
                            'a': a,
                            'w': w,
                            'ndt': ndt,
                            'sdv': sdv,
                            's': s,
                            **boundary_params,
                            'delta_t': delta_t,
                            'max_t': max_t,
                            'n_samples': n_samples,
                            'simulator': 'ddm_sdv',
                            'boundary_fun_type': boundary_fun.__name__,
                            'possible_choices': [-1, 1]}
           )

# -------------------------------------------------------------------------------------------------