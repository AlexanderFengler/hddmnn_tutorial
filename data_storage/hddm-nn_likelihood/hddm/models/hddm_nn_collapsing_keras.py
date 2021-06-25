
"""
"""

from copy import copy
import numpy as np
import pymc
import wfpt
import pickle

from kabuki.hierarchical import Knode
from kabuki.utils import stochastic_from_dist
from hddm.models import HDDM
from wfpt import wiener_like_nn_collapsing_keras


class HDDMnn_collapsing_keras(HDDM):
    """HDDM model that uses neural net likelihood

    """

    def __init__(self, *args, **kwargs):
        self.non_centered = kwargs.pop('non_centered', False)
        self.free = kwargs.pop('free',True)
        self.k = kwargs.pop('k',False)
        self.wfpt_nn_collapsing_class_keras = Wienernn_collapsing_keras

        super(HDDMnn_collapsing_keras, self).__init__(*args, **kwargs)

    def _create_stochastic_knodes(self, include):
        knodes = super(HDDMnn_collapsing_keras, self)._create_stochastic_knodes(include)
        if self.free:
            knodes.update(self._create_family_gamma_gamma_hnormal('beta', g_mean=1.5, g_std=0.75, std_std=2, std_value=0.1, value=1))
            if self.k:
                knodes.update(self._create_family_gamma_gamma_hnormal('alpha', g_mean=1.5, g_std=0.75, std_std=2, std_value=0.1, value=1))
        else:
            knodes.update(self._create_family_trunc_normal('beta', lower=0.3, upper=7, value=1))
            if self.k:
                knodes.update(self._create_family_trunc_normal('alpha', lower=0.3, upper=5, value=1))
        return knodes

    def _create_wfpt_parents_dict(self, knodes):
        wfpt_parents = super(HDDMnn_collapsing_keras, self)._create_wfpt_parents_dict(knodes)
        wfpt_parents['beta'] = knodes['beta_bottom']
        wfpt_parents['alpha'] = knodes['alpha_bottom'] if self.k else 3.00
        return wfpt_parents

    def _create_wfpt_knode(self, knodes):
        wfpt_parents = self._create_wfpt_parents_dict(knodes)
        return Knode(self.wfpt_nn_collapsing_class_keras, 'wfpt', observed=True, col_name=['nn_response', 'rt'], **wfpt_parents)


def wienernn_like_collapsing_keras(x, v, sv, a, alpha, beta, z, sz, t, st, p_outlier=0): #theta

    wiener_params = {'err': 1e-4, 'n_st': 2, 'n_sz': 2,
                     'use_adaptive': 1,
                     'simps_err': 1e-3,
                     'w_outlier': 0.1}
    wp = wiener_params

    #with open("weights.pickle", "rb") as tmp_file:
    #    weights = pickle.load(tmp_file)
    #with open('biases.pickle', 'rb') as tmp_file:
    #    biases = pickle.load(tmp_file)
    #with open('activations.pickle', 'rb') as tmp_file:
    #    activations = pickle.load(tmp_file)

    #print('hei')
    nn_response = x['nn_response'].values.astype(int)
    return wiener_like_nn_collapsing_keras(np.absolute(x['rt'].values), nn_response, v, sv, a, alpha, beta, z, sz, t, st, p_outlier=p_outlier, **wp)
Wienernn_collapsing_keras = stochastic_from_dist('Wienernn_collapsing_keras', wienernn_like_collapsing_keras)
