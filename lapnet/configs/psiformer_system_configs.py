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

systems['benzene'] = [['C', (0.00000,2.63664,0.00000)],
                    ['C', (2.28339,1.31832,0.00000)],
                    ['C', (2.28339,-1.31832,0.00000)],
                    ['C', (0.00000,-2.63664,0.00000)],
                    ['C', (-2.28339,-1.31832,0.00000)],
                    ['C', (-2.28339,1.31832,0.00000)],
                    ['H', (0.00000,4.69096,0.00000)],
                    ['H', (4.06250,2.34549,0.00000)],
                    ['H', (4.06250,-2.34549,0.00000)],
                    ['H', (0.00000,-4.69096,0.00000)],
                    ['H', (-4.06250,-2.34549,0.00000)],
                    ['H', (-4.06250,2.34549,0.00000)]]
atom_spin_configs['benzene'] = ((3,3),(3,3),(3,3),(3,3),(3,3),(3,3),(1,0),(0,1),(1,0),(0,1),(1,0),(0,1))
units['benzene'] = 'bohr'


systems['toluene'] = [['C', (-0.01831, 1.72486, 0.0)],
                    ['C', (-0.01297, 0.37035, 2.2706)],
                    ['C', (-0.01297, 0.37035, -2.2706)],
                    ['C', (-0.01297, -2.26452, 2.27703)],
                    ['C', (-0.01297, -2.26452, -2.27703)],
                    ['C', (-0.009, -3.59169, 0.0)],
                    ['C', (0.05583, 4.56877, 0.0)],
                    ['H', (2.00464, 5.26559, 0.0)],
                    ['H', (-0.88281, 5.33834, -1.67217)],
                    ['H', (-0.88281, 5.33834, 1.67217)],
                    ['H', (-0.02402, 1.3927, 4.05592)],
                    ['H', (-0.01841, -3.28187, 4.06225)],
                    ['H', (-0.01415, -5.64576, 0.0)],
                    ['H', (-0.01841, -3.28187, -4.06225)],
                    ['H', (-0.02402, 1.3927, -4.05592)]]
atom_spin_configs['toluene'] = ((3,3),(3,3),(3,3),(3,3),(3,3),(3,3),(3,3),(1,0),(0,1),(1,0),(0,1),(1,0),(0,1),(1,0),(0,1))
units['toluene'] = 'bohr'



systems['naphthalene'] = [['C', (0.0, 0.0, 1.35203)],
                          ['C', (0.0, 0.0, -1.35203)],
                          ['C', (0.0, 2.34349, 2.6493)],
                          ['C', (0.0, -2.34349, 2.6493)],
                          ['C', (0.0, 2.34349, -2.6493)],
                          ['C', (0.0, -2.34349, -2.6493)],
                          ['C', (0.0, -4.59147, 1.33509)],
                          ['C', (0.0, 4.59147, 1.33509)],
                          ['C', (0.0, -4.59147, -1.33509)],
                          ['C', (0.0, 4.59147, -1.33509)],
                          ['H', (0.0, 2.34107, 4.70689)],
                          ['H', (0.0, -2.34107, 4.70689)],
                          ['H', (0.0, -6.37654, 2.35261)],
                          ['H', (0.0, -6.37654, -2.35261)],
                          ['H', (0.0, -2.34107, -4.70689)],
                          ['H', (0.0, 2.34107, -4.70689)],
                          ['H', (0.0, 6.37654, -2.35261)],
                          ['H', (0.0, 6.37654, 2.35261)], ]
atom_spin_configs['naphthalene'] = ((3,3),(3,3),(3,3),(3,3),(3,3),(3,3),(3,3),(3,3),(3,3),(3,3),(1,0),(0,1),(1,0),(0,1),(1,0),(0,1),(1,0),(0,1))
units['naphthalene'] = 'bohr'

systems['CCl4'] = [['C', (0.0, 0.0, 0.0)],
                    ['Cl', (1.93005, 1.93005, 1.93005)],
                    ['Cl', (-1.93005, -1.93005, 1.93005)],
                    ['Cl', (-1.93005, 1.93005, -1.93005)],
                    ['Cl', (1.93005, -1.93005, -1.93005)]]
atom_spin_configs['CCl4'] = ((3,3),(8,9),(9,8),(9,8),(8,9))
units['CCl4'] = 'bohr'

def set_psiformer_systems(cfg):
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
  Returns config for systems in Psiformer's paper.
  Using config.system.molecule_name to set the system.
  """
  name = 'benzene'
  cfg = base_config.default()
  cfg.system.molecule_name = name
  cfg.system.atom_spin_configs = None
  with cfg.ignore_type():
    cfg.system.set_molecule = set_psiformer_systems
    cfg.config_module = '.psiformer_system_configs'
  return cfg
