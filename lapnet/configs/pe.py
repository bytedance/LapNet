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

from lapnet import base_config
from lapnet.utils import system
from lapnet.utils.system import Atom
import numpy as np

# C-C length
lcc = 1.55
# C-H length
lch = 1.09
theta = 109.4712/180.0 * 3.1415926

# repeat config
dxc = lcc * np.sin(theta/2.0)
dyc = lcc * np.cos(theta/2.0)
dxh = lch * np.sin(theta/2.0)
dyh = lch * np.cos(theta/2.0)

pos_C1 = np.array((-0.5 * dxc, -0.5 * dyc, 0.0))
pos_C2 = np.array((0.5 * dxc, 0.5 * dyc, 0.0))

pos_H11 = np.array((-0.5 * dxc, -0.5 * dyc - dyh, dxh))
pos_H12 = np.array((-0.5 * dxc, -0.5 * dyc - dyh, -dxh))
pos_H21 = np.array((0.5 * dxc, 0.5 * dyc + dyh, dxh))
pos_H22 = np.array((0.5 * dxc, 0.5 * dyc + dyh, -dxh))

rep_dis = np.array((2*dxc,0,0))

dpos_Hl = np.array((-dxh,dyh,0)) + pos_C1
dpos_Hr = np.array((dxh,-dyh,0)) + pos_C2

def get_config(input_str):
    r_str= input_str
    repeat_num = int(r_str)

    # Get default options.
    cfg = base_config.default()
    # Set up molecule
    cfg.system.electrons = (repeat_num * 8 + 1, repeat_num * 8 + 1)  # Spins
    molecule = [Atom(symbol='H', coords=dpos_Hl, units='angstrom')]
    for i in range(repeat_num):
        molecule.extend([
                Atom(symbol='C', coords=pos_C1+i*rep_dis, units='angstrom'),
                Atom(symbol='C', coords=pos_C2+i*rep_dis, units='angstrom'),
                Atom(symbol='H', coords=pos_H11+i*rep_dis, units='angstrom'),
                Atom(symbol='H', coords=pos_H12+i*rep_dis, units='angstrom'),
                Atom(symbol='H', coords=pos_H21+i*rep_dis, units='angstrom'),
                Atom(symbol='H', coords=pos_H22+i*rep_dis, units='angstrom')]
                )
    molecule.extend([Atom(symbol='H', coords=dpos_Hr+(repeat_num-1)*rep_dis, units='angstrom'),])

    cfg.system.molecule = molecule

    cfg.system.molecule_name = 'C{}H{}'.format(2*repeat_num, 4*repeat_num + 2)
    spin_configs = [(1,0)]
    for i in range(repeat_num):
        spin_configs.extend([(3,3),(3,3),(0,1),(1,0),(0,1),(1,0)])
    spin_configs.extend([(0,1)])
    cfg.system.atom_spin_configs = spin_configs
    return cfg
