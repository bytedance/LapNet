# Copyright 2022 DeepMind Technologies Limited.
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

systems = {}
atom_spin_configs = {}
units = {}

systems['LiH'] = [['Li', (0.000000, 0.000000,  0.000000)],
                 ['H', (3.015,  0.000000,  0.000000)]]
atom_spin_configs['LiH'] = ((2,1),(0,1))
units['LiH'] = 'bohr'

systems['Li2'] = [['Li', (0.000000, 0.000000,  0.000000)],
                 ['Li', (5.051,  0.000000,  0.000000)]]
atom_spin_configs['Li2'] = ((2,1),(1,2))
units['Li2'] = 'bohr'

systems['NH3'] = [['N', (0.0, 0.0, 0.22013)],
                ['H', (0.0, 1.77583, -0.51364)],
                ['H', (1.53791, -0.88791, -0.51364)],
                ['H', (-1.53791, -0.88791, -0.51364)],]
atom_spin_configs['NH3'] = ((4,3),(1,0),(0,1),(0,1))
units['NH3'] = 'bohr'

systems['CH4'] = [('C', (0.0, 0.0, 0.0)),
                ('H', (1.18886, 1.18886, 1.18886)),
                ('H', (-1.18886, -1.18886, 1.18886)),
                ('H', (1.18886, -1.18886, -1.18886)),
                ('H', (-1.18886, 1.18886, -1.18886)),]
atom_spin_configs['CH4'] = ((3,3),(1,0),(0,1),(1,0),(0,1))
units['CH4'] = 'bohr'

systems['CO'] = [['C', (0.000000, 0.000000,  0.000000)],
                 ['O', (2.173,  0.000000,  0.000000)]]
atom_spin_configs['CO'] = ((3,3),(4,4))
units['CO'] = 'bohr'

systems['N2'] = [['N', (0.000000, 0.000000,  0.000000)],
                 ['N', (2.068,  0.000000,  0.000000)]]
atom_spin_configs['N2'] = ((4,3),(3,4))
units['N2'] = 'bohr'

systems['C2H4'] = [('C', (0.0, 0.0, 1.26135)),
      ('C', (0.0, 0.0, -1.26135)),
      ('H', (0.0, 1.74390, 2.33889)),
      ('H', (0.0, -1.74390, 2.33889)),
      ('H', (0.0, 1.74390, -2.33889)),
      ('H', (0.0, -1.74390, -2.33889)),]
atom_spin_configs['C2H4'] = ((3,3),(3,3),(1,0),(0,1),(1,0),(0,1))
units['C2H4'] = 'bohr'

systems['methylamine'] = [['C',(0.0517, 0.7044, 0.0)],
                    ['N',(0.0517, -0.7596, 0.0)],
                    ['H',(-0.9417, 1.1762, 0.0)],
                    ['H',(-0.4582, -1.0994, 0.8124)],
                    ['H',(-0.4582, -1.0994, -0.8124)],
                    ['H',(0.5928, 1.0567, 0.8807)],
                    ['H',(0.5928, 1.0567, -0.8807)]]
atom_spin_configs['methylamine'] = ((3,3),(4,3),(0,1),(1,0),(0,1),(1,0),(0,1))
units['methylamine'] = 'angstrom'


# WARNING: O3 here is assigned to a singlet state, but may also exchange to a triplet state after optimization
systems['O3'] = [['O',(0.0, 2.0859, -0.4319)],
                    ['O',(0.0, 0.0, 0.8638)],
                    ['O',(0.0, -2.0859, -0.4319)],]
atom_spin_configs['O3'] = ((4,4),(4,4),(4,4))
units['O3'] = 'bohr'


# TODO: ask for full config
systems['ethanol'] = [['C',(2.2075, -0.7566, 0.0)],
                      ['C',(0.0, 1.0572, 0.0)],
                      ['O',(-2.2489, -0.4302, 0.0)],
                      ['H',(-3.6786, 0.7210, 0.0)],
                      ['H',(0.0804, 2.2819, 1.6761)],
                      ['H',(0.0804, 2.2819, -1.6761)],
                      ['H',(3.9985, 0.2736, 0.0)],
                      ['H',(2.1327, -1.9601, 1.6741)],
                      ['H',(2.1327, -1.9601, -1.6741)]]
atom_spin_configs['ethanol'] = ((3,3),(3,3),(4,4),(1,0),(0,1),(1,0),(0,1),(1,0),(0,1))
units['ethanol'] = 'bohr'


systems['bicbut'] = [['C',(0.0, 2.13792, 0.58661)],
                      ['C',(0.0, -2.13792, 0.58661)],
                      ['C',(1.41342, 0.0, -0.58924)],
                      ['C',(-1.41342, 0.0, -0.58924)],
                      ['H',(0.0, 2.33765, 2.64110)],
                      ['H',(0.0, 3.92566, -0.43023)],
                      ['H',(0.0, -2.33765, 2.64110)],
                      ['H',(0.0, -3.92566, -0.43023)],
                      ['H',(2.67285, 0.0, -2.19514)],
                      ['H',(-2.67285, 0.0, -2.19514)]]
atom_spin_configs['bicbut'] = ((3,3),(3,3),(3,3),(3,3),(1,0),(0,1),(1,0),(0,1),(1,0),(0,1))
units['bicbut'] = 'bohr'


def set_ferminet_systems(cfg):
  """Sets molecule and electrons fields in ConfigDict."""

  if cfg.system.molecule_name not in systems.keys():
    raise ValueError(f'Unrecognized molecule: {cfg.system.molecule_name}')
  molecule = []
  for element, coords in systems[cfg.system.molecule_name]:
    molecule.append(system.Atom(symbol=element, coords=coords, units=units[cfg.system.molecule_name]))
  cfg.system.molecule = molecule

  # electrons = sum(int(round(atom.charge)) for atom in cfg.system.molecule)
  # nalpha = electrons // 2
  # cfg.system.electrons = (nalpha, nalpha)

  cfg.system.atom_spin_configs = atom_spin_configs[cfg.system.molecule_name] if \
               atom_spin_configs.__contains__(cfg.system.molecule_name) else None

  cfg.system.electrons = (sum(spin[0] for spin in cfg.system.atom_spin_configs), sum(spin[1] for spin in cfg.system.atom_spin_configs))


  return cfg


def get_config():
  """  
  Returns config for systems in FermiNet's paper.
  Using config.system.molecule_name to set the system.
  """
  name = 'NH3'
  cfg = base_config.default()
  cfg.system.molecule_name = name
  cfg.system.atom_spin_configs = None
  with cfg.ignore_type():
    cfg.system.set_molecule = set_ferminet_systems
    cfg.config_module = '.ferminet_system_configs'
  return cfg
