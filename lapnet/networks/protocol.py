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

from typing_extensions import Protocol
import enum
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple, Union, List
import lapjax.numpy as jnp
import chex
import attr

LayerArgs = Tuple[Tuple[int, int], ...]

# Recursive types are not yet supported in pytype - b/109648354.
# pytype: disable=not-supported-yet
ParamTree = Union[jnp.ndarray, Iterable['ParamTree'], Mapping[Any, 'ParamTree']]
# pytype: enable=not-supported-yet
# Parameters for a single part of the network are just a dict.
Param = Mapping[str, jnp.ndarray]

## Interfaces (public) ##

class Options(Protocol):
    pass

class InitNetwork(Protocol):

  def __call__(self, key: chex.PRNGKey) -> ParamTree:
    """Returns initialized parameters for the network.

    Args:
      key: RNG state
    """


class WaveFuncLike(Protocol):

  def __call__(self, params: ParamTree,
               electrons: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Returns the sign and log magnitude of the wavefunction.

    Args:
      params: network parameters.
      electrons: electron positions, shape (nelectrons*ndim), where ndim is the
        dimensionality of the system.
    """


class LogWaveFuncLike(Protocol):

  def __call__(self, params: ParamTree, electrons: jnp.ndarray) -> jnp.ndarray:
    """Returns the log magnitude of the wavefunction.

    Args:
      params: network parameters.
      electrons: electron positions, shape (nelectrons*ndim), where ndim is the
        dimensionality of the system.
    """

class OrbitalLike(Protocol):
    def __call__(self, params: ParamTree, pos: jnp.ndarray, atoms: jnp.ndarray, nspins: Tuple[int, ...]) -> Tuple[List[jnp.ndarray], ...]:
        ''' Returns the orbital of the wavefunction.
        Args:
            params: network parameters.
            pos: electron positions, shape (nelectrons*ndim), where ndim is the
                dimensionality of the system.
            atoms: atom position.
            nspins: electron spins
        '''

class MakeNetwork(Protocol):
    def __call__(self, *args: Any, **kwds: Any) -> Tuple[InitNetwork, WaveFuncLike, Options]:
        ''' 
        Returns the network constructor.
        '''

## Interfaces (network components) ##


class FeatureInit(Protocol):

  def __call__(self) -> Tuple[Tuple[int, int], Param]:
    """Creates the learnable parameters for the feature input layer.

    Returns:
      Tuple of ((x, y), params), where x and y are the number of one-electron
      features per electron and number of two-electron features per pair of
      electrons respectively, and params is a (potentially empty) mapping of
      learnable parameters associated with the feature construction layer.
    """


class FeatureApply(Protocol):

  def __call__(self, ae: jnp.ndarray, r_ae: jnp.ndarray, ee: jnp.ndarray,
               r_ee: jnp.ndarray,
               **params: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Creates the features to pass into the network.

    Args:
      ae: electron-atom vectors. Shape: (nelectron, natom, 3).
      r_ae: electron-atom distances. Shape: (nelectron, natom, 1).
      ee: electron-electron vectors. Shape: (nelectron, nelectron, 3).
      r_ee: electron-electron distances. Shape: (nelectron, nelectron).
      **params: learnable parameters, as initialised in the corresponding
        FeatureInit function.
    """


@attr.s(auto_attribs=True)
class FeatureLayer:
  init: FeatureInit
  apply: FeatureApply


class FeatureLayerType(enum.Enum):
  STANDARD = enum.auto()
