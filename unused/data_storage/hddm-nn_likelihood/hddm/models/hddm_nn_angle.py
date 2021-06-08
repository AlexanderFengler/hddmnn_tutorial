
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
from wfpt import wiener_like_nn_angle


class HDDMnn_angle(HDDM):
    """HDDM model that uses neural net likelihood

    """

    def __init__(self, *args, **kwargs):
        self.non_centered = kwargs.pop('non_centered', False)
        self.free = kwargs.pop('free',True)
        self.wfpt_nn_angle_class = Wienernn_angle

        super(HDDMnn_angle, self).__init__(*args, **kwargs)

    def _create_stochastic_knodes(self, include):
        knodes = super(HDDMnn_angle, self)._create_stochastic_knodes(include)
        if self.free:
            knodes.update(self._create_family_gamma_gamma_hnormal('theta', g_mean=1.5, g_std=0.75, std_std=2, std_value=0.1, value=1))
        else:
            knodes.update(self._create_family_trunc_normal('theta', lower=0, upper=1.2, value=0.5))
        return knodes

    def _create_wfpt_parents_dict(self, knodes):
        wfpt_parents = super(HDDMnn_angle, self)._create_wfpt_parents_dict(knodes)
        wfpt_parents['theta'] = knodes['theta_bottom']
        return wfpt_parents

    def _create_wfpt_knode(self, knodes):
        wfpt_parents = self._create_wfpt_parents_dict(knodes)
        return Knode(self.wfpt_nn_angle_class, 'wfpt', observed=True, col_name=['nn_response', 'rt'], **wfpt_parents)


def wienernn_like_angle(x, v, sv, a, theta, z, sz, t, st, p_outlier=0):

    wiener_params = {'err': 1e-4, 'n_st': 2, 'n_sz': 2,
                     'use_adaptive': 1,
                     'simps_err': 1e-3,
                     'w_outlier': 0.1}
    wp = wiener_params

    nn_response = x['nn_response'].values.astype(int)
    return wiener_like_nn_angle(np.absolute(x['rt'].values), nn_response, v, sv, a, theta, z, sz, t, st, p_outlier=p_outlier, **wp)
Wienernn_angle = stochastic_from_dist('Wienernn_angle', wienernn_like_angle)
