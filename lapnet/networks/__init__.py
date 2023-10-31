# Copyright 2023 Bytedance Ltd. and/or its affiliate
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import functools

from .orig_ferminet import fermi_net_orbitals, make_fermi_net
from .protocol import *
from .psiformer import make_psiformer, psiformer_orbitals
from .lapnet import make_lapnet, lapnet_orbitals
from .utils import construct_input_features

def network_provider(cfg) -> MakeNetwork:
    '''
    This function uses the configuration to decide which network is used 
    and assign the network architecture hyperparameters. 
    The returned function takes the systems dependent parameters as input,
    including atoms, nspins, charges and hf_solution. The output of the 
    returned function is the wavefunctions.
    '''
    name = cfg.network.name
    if name.lower() == 'ferminet':
        return functools.partial(make_fermi_net,
                envelope=cfg.network.envelope_type,
                feature_layer=cfg.network.get('feature_layer', 'standard'),
                bias_orbitals=cfg.network.bias_orbitals,
                use_last_layer=cfg.network.use_last_layer,
                full_det=cfg.network.full_det,
                **cfg.network.detnet
            )
    if name.lower() == 'lapnet':
        return functools.partial(make_lapnet,
                envelope=cfg.network.envelope_type,
                bias_orbitals=cfg.network.bias_orbitals,
                use_layernorm=cfg.network.use_layernorm,
                jas_w_init=cfg.network.jas_w_init,
                orbitals_spin_split=cfg.network.orbitals_spin_split,
                **cfg.network.detnet
            )
    if name.lower() == 'psiformer':
        return functools.partial(make_psiformer,
                envelope=cfg.network.envelope_type,
                bias_orbitals=cfg.network.bias_orbitals,
                use_layernorm=cfg.network.use_layernorm,
                jas_w_init=cfg.network.jas_w_init,
                orbitals_spin_split=cfg.network.orbitals_spin_split,
                **cfg.network.detnet
            )
    raise NotImplementedError

def network_orbital_provider(cfg) -> OrbitalLike:
    '''
    return OrbitalLike function. Used in the pretraining stage.
    '''
    name = cfg.network.name
    if name.lower() == 'ferminet':
        return fermi_net_orbitals
    if name.lower() == 'lapnet':
        return lapnet_orbitals
    if name.lower() == 'psiformer':
        return psiformer_orbitals
    raise NotImplementedError
