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
from typing import Sequence, Tuple

import attr
import chex
import jax

import lapjax.numpy as jnp
from lapnet import envelopes
from lapnet.networks import network_blocks

from .protocol import *
from .transformer_blocks import (
    CrossAttentionLayer,
    LayerNormBlock,
    MultiheadCrossAttention,
)
from .utils import construct_input_features, init_jastrow_weights


@attr.s(auto_attribs=True, kw_only=True)
class LapNetOptions:
  """Options controlling the LapNet architecture.

  Attributes:
    ndim: dimension of system. Change only with caution.
    hidden_dims: Tuple of pairs, where each pair contains the number of hidden
      units and number of MultiheadCrossAttention. The number of layers is given 
      by the length of the tuple.
    determinants: Number of determinants to use.
    full_det: WARNING: please keep true for lapnet
    bias_orbitals: If true, include a bias in the final linear layer to shape
      the outputs into orbitals.
    envelope_label: Envelope to use to impose orbitals go to zero at infinity.
      See envelopes module.
    envelope: Envelope object to create and apply the multiplicative envelope.
    attn_layer: Transformer layers used by lapnet
    use_layernorm: If True, use layernorm in the attention block
    jas_w_init: Initialization Value of jastrow factor
    orbitals_spin_split: If true, use different parameters for alpha and beta 
    electrons in the orbital and envelope function.
  """
  ndim: int = 3
  hidden_dims: Tuple = ((256, 4), (256, 4), (256, 4), (256, 4))
  determinants: int = 16
  full_det: bool = True
  bias_orbitals: bool = False
  envelope_label: envelopes.EnvelopeLabel = envelopes.EnvelopeLabel.ABS_ISOTROPIC
  envelope: envelopes.Envelope = attr.ib(
      default=attr.Factory(
          lambda self: envelopes.get_envelope(self.envelope_label),
          takes_self=True))
  atten_layers: Sequence[CrossAttentionLayer] = []
  use_layernorm: bool = False
  jas_w_init: float = 0.0
  orbitals_spin_split: bool = True


def get_multihead_list(hidden_dims: LayerArgs,
                       layernorm: bool = False) -> Sequence[CrossAttentionLayer]:
  """Return the backbone of transformer as a list of multihead layers.

  Args:
      hidden_dims (LayerArgs): Each elecment is a tuple decribing (output_dim, num_heads).
      layernorm (bool): Whether to use laryorm in the attention block 

  Returns:
      list: Sequence of MultiheadCrossAttention.
  """
  atten_layers = [MultiheadCrossAttention(
                    output_dim=output_dim,
                    num_heads=num_heads,)
        for (output_dim, num_heads) in hidden_dims]

  ln1 = [LayerNormBlock(use_layernorm=layernorm) for _ in hidden_dims]
  ln2 = [LayerNormBlock(use_layernorm=layernorm) for _ in hidden_dims]
  ln3 = [LayerNormBlock(use_layernorm=layernorm) for _ in hidden_dims]

  return [CrossAttentionLayer(
            attention=u, layernorm1=v, layernorm2=w, layernorm3=x)
            for u, v, w, x in zip(atten_layers, ln1, ln2, ln3)]


def init_lapnet_params(
    key: chex.PRNGKey,
    atoms: jnp.ndarray,
    nspins: Tuple[int, ...],
    options: LapNetOptions = LapNetOptions(),
) -> ParamTree:
  """Initializes parameters for the LapNet Neural Network.

  Args:
      key (chex.PRNGKey): JAX RNG state.
      atoms (jnp.ndarray): (natom, ndim) array of atom positions.
      nspins (Tuple[int, ...]): A tuple representing the number of spin-up and spin-down electrons. Should have length 2.
      options (LapNetOptions): Network options.
  """
  if not isinstance(options, LapNetOptions):
    raise ValueError("options should be LapNetOptions")
  if options.envelope.apply_type != envelopes.EnvelopeType.PRE_DETERMINANT:
    raise ValueError('In LapNet, the envelope type must be `PRE_DETERMINANT`.')
  if not options.full_det:
    raise ValueError('In LapNet, the full_det option must be true.')

  natom, ndim = atoms.shape

  params = {}  # The dict of all parameters to be optimized.

  # num_features_in and num_features_out represent
  # the dimension of initial array as well as the Transformer input dimension.
  # params['input'] is a linear layer weights.
  num_features_in, num_features_out = natom * (ndim + 1) + 1, options.hidden_dims[0][0]
  key, subkey = jax.random.split(key)
  params['input'] = network_blocks.init_linear_layer(
    subkey, num_features_in, num_features_out, include_bias=True
  )

  # The input dimension of each layer
  dims_in = [num_features_out] + [w[0] for w in options.hidden_dims[:-1]]
  # Initialize the parameters for transformer backbone.
  params['transformer'] = []
  for dim_in, layer in zip(dims_in, options.atten_layers):
    dic = {}
    input_example = jnp.ones((sum(nspins), dim_in))
    output_example = jnp.ones((sum(nspins), layer.attention.output_dim))
    key, attkey, mlpkey, sparskey, lnkey = jax.random.split(key, num = 5)
    dic['attention'] = layer.attention.init(attkey, [input_example, input_example])
    dic['MLP'] = network_blocks.init_linear_layer(
        key=mlpkey,
        in_dim=layer.attention.output_dim,
        out_dim=layer.attention.output_dim,
        include_bias=True
    )

    dic['spars'] = [network_blocks.init_linear_layer(
        key=key,
        in_dim=layer.attention.output_dim,
        out_dim=layer.attention.output_dim,
        include_bias=True
    ) for key in jax.random.split(sparskey, num=2)]

    ln1key, ln2key, ln3key = jax.random.split(lnkey, num=3)
    dic['ln1'] = layer.layernorm1.init(ln1key, input_example)
    dic['ln2'] = layer.layernorm2.init(ln2key, input_example)
    dic['ln3'] = layer.layernorm3.init(ln3key, output_example)

    params['transformer'].append(dic)

  # Construct Orbital Projection
  output_dim = sum(nspins) * options.determinants
  if not options.orbitals_spin_split:
    # Construct Orbital Projection
    key, subkey = jax.random.split(key, num=2)
    params['orbital'] = network_blocks.init_linear_layer(
        key=subkey,
        in_dim=options.hidden_dims[-1][0],
        out_dim=output_dim,
        include_bias=options.bias_orbitals)

    # Construct Envelope
    params['envelope'] = options.envelope.init(
        natom=natom, output_dims=[output_dim], hf=None, ndim=ndim)[0]
  else:
    params['orbital'] = []
    params['envelope'] = []
    for i in range(len(nspins)):
      # Construct Orbital Projection
      key, subkey = jax.random.split(key, num=2)
      params['orbital'].append(network_blocks.init_linear_layer(
          key=subkey,
          in_dim=options.hidden_dims[-1][0],
          out_dim=output_dim,
          include_bias=options.bias_orbitals))

      # Construct Envelope
      params['envelope'].append(options.envelope.init(
          natom=natom, output_dims=[output_dim], hf=None, ndim=ndim)[0])

  # Construct Jastrow factor
  params['jastrow'] = init_jastrow_weights(key, options.jas_w_init)

  return params


def lapnet_orbitals(
    params,
    pos: jnp.ndarray,
    atoms: jnp.ndarray,
    nspins: Tuple[int, ...],
    options: LapNetOptions=LapNetOptions(),
):
  """Forward evaluation of the LapNet up to the orbitals. 

  Args:
      params: A dictionary of parameters, contain fileds:
        `input`: linear layer mapping initial array to transformer inputs.
        `transformer`: parameters used in transformer backbones.
        `orbital`: linear layer mapping transformer outputs to orbitals.
        `envelope`: parameters used in the envelope function.
        `jastrow`: parameters used in the Jastrow factor.
      pos (jnp.ndarray): The electron positions, with shape (3N,).
      atoms (jnp.ndarray): The atom positions.
      nspins (Tuple[int, ...]): Tuple with number of spin up and spin down electrons. Should have length 2.
      options (LapNetOptions): Network options.

  Returns:
    Binary tuple containg:
      One matrix with shape (K, N, N), where the second dimension is equivariant equivariant to the input.
      (ae, r_ae, r_ee), representing the atom-electron vectors, distrances and e-e distrances.

  """
  ae, ee, r_ae, r_ee = construct_input_features(pos, atoms)
  n_elec = r_ae.shape[0]

  # Construct the input of the transformer.
  scale_r_ae = jnp.log(1.0 + r_ae)
  input_features = jnp.concatenate((scale_r_ae, ae * scale_r_ae / r_ae), axis = 2).reshape(
    (n_elec, -1)
  )
  # concatenate nspin features
  input_spin = jnp.concatenate([
                  jnp.ones((nspins[0], 1)),
                  -jnp.ones((nspins[1], 1))], axis = 0)
  input_features = jnp.concatenate((input_features, input_spin), axis = -1)
  hs = network_blocks.linear_layer(input_features, **params['input'])
  hd = hs  # deepcopy(hs)

  # Start with the transformer backbone
  for layer, param in zip(
        options.atten_layers, params['transformer']):
    if 'ln1' in param:
      hs_norm = layer.layernorm1.apply(param['ln1'], hs)
      hd_norm = layer.layernorm2.apply(param['ln2'], hd)
    fd = hd + layer.attention.apply(param["attention"],
                                        [hs_norm, hd_norm])[0]
    if 'ln3' in param:
      fd_norm = layer.layernorm3.apply(param['ln3'], fd)
    hd = fd + jnp.tanh(network_blocks.linear_layer(
                              fd_norm, **param["MLP"]))
    # keep sparsity for hs
    for i in range(2):
      hs = hs + jnp.tanh(network_blocks.linear_layer(
                              hs, **param["spars"][i]))

  if not options.orbitals_spin_split:
    # Construct the orbitals
    orbitals = network_blocks.linear_layer(hd, **params['orbital'])

    # Apply PRE_DETERMINANT envelopes.
    orbitals = orbitals * options.envelope.apply(
      ae=ae, r_ae=r_ae, r_ee=r_ee, **params['envelope'])
    orbitals = jnp.transpose(orbitals.reshape((n_elec, -1, n_elec)), (1, 0, 2))
  else:
    h_to_orbitals = jnp.split(hd, network_blocks.array_partitions(nspins), axis=0)
    # Drop unoccupied spin channels
    h_to_orbitals = [h for h, spin in zip(h_to_orbitals, nspins) if spin > 0]
    active_spin_channels = [spin for spin in nspins if spin > 0]
    active_spin_partitions = network_blocks.array_partitions(active_spin_channels)
    # Create orbitals.
    orbitals = [
        network_blocks.linear_layer(h, **p)
        for h, p in zip(h_to_orbitals, params['orbital'])
    ]
    ae_channels = jnp.split(ae, active_spin_partitions, axis=0)
    r_ae_channels = jnp.split(r_ae, active_spin_partitions, axis=0)
    r_ee_channels = jnp.split(r_ee, active_spin_partitions, axis=0)
    for i in range(len(active_spin_channels)):
      orbitals[i] = orbitals[i] * options.envelope.apply(
          ae=ae_channels[i],
          r_ae=r_ae_channels[i],
          r_ee=r_ee_channels[i],
          **params['envelope'][i],
      )

    # Reshape into matrices.
    shapes = [(spin, -1, sum(nspins) if options.full_det else spin)
              for spin in active_spin_channels]
    orbitals = [
        jnp.reshape(orbital, shape) for orbital, shape in zip(orbitals, shapes)
    ]
    orbitals = [jnp.transpose(orbital, (1, 0, 2)) for orbital in orbitals]
    orbitals = jnp.concatenate(orbitals, axis=1)  # only support full_det

  return [orbitals], (ae, r_ae, r_ee)


def jastrow_factor(r_ee: jnp.ndarray, nspins: Tuple[int,int], alpha_par: jnp.ndarray, alpha_anti: jnp.ndarray):
  """Calculate the jastrow factor specified in PsiFormer paper.

  Specifically, 
  ```
  \sum_{i<j, \sigma_i = \sigma_j} -0.25 * \frac{par**2}{par + |r_ee[i,j]|} + 
  \sum_{\sigma_i\not=\sigma_j} -0.5 * \frac{anti**2}{anti+|r_ee[i,j]|}
  ```

  Args:
      r_ee (jnp.ndarray): (n_elec, n_elec) array. Represent the distance of electrons.
      par (jnp.ndarray): The paramter of electrons with identical spins.
      anti (jnp.ndarray): The paramter of electrons with different spins.
  """
  n_elec = sum(nspins)
  r_ee = r_ee.reshape((n_elec, n_elec))

  # Identical spins part
  par_mat = (alpha_par**2) / (alpha_par + r_ee + jnp.eye(n_elec))
  masks = [jnp.triu(jnp.ones((ele, ele)), k=1) > 0 for ele in nspins]

  par_jastrow = -0.25 * (
    jnp.sum(par_mat[:nspins[0], :nspins[0]], where=masks[0]) + \
    jnp.sum(par_mat[nspins[0]:, nspins[0]:], where=masks[1]))

  # Different spins part
  off_diag = r_ee[:nspins[0], nspins[0]:]  # upper triangle
  anti_jastrow = -0.5 * jnp.sum((alpha_anti**2)/(alpha_anti+off_diag))

  return par_jastrow + anti_jastrow


def lapnet_each_det(
    params,
    pos: jnp.ndarray,
    atoms: jnp.ndarray,
    nspins: Tuple[int, ...],
    options: LapNetOptions = LapNetOptions(),
):
  """Forward evaluation of the LapNet. Outputs all the determinants.

  Args:
      params: A dictionary of parameters, containing all fields needed in `lapnet_orbitals`, as well as 'jastrow' used in `jastrow_factor`.
      pos (jnp.ndarray): The electron positions, a 3N dimension vector.
      atoms (jnp.ndarray): The atom positions.
      nspins (Tuple[int, ...]): Tuple with number of spin up and spin down electrons. Should have length 2.
      options (LapNetOptions, optional): Network options. Defaults to LapNetOptions().

  Returns:
      Output of determinants in log space. The sign is not returned.
  """

  orbitals, (_, _, r_ee) = lapnet_orbitals(
      params,
      pos,
      atoms=atoms,
      nspins=nspins,
      options=options,
  )
  output = network_blocks.individual_slogdet(orbitals)[1]
  output = output + jastrow_factor(r_ee, nspins, **params['jastrow'])

  return output


def lapnet(
    params,
    pos: jnp.ndarray,
    atoms: jnp.ndarray,
    nspins: Tuple[int, ...],
    options: LapNetOptions = LapNetOptions(),
):
  """Forward evaluation of the LapNet wavefunction.

  This implementation is similar to `fermi_net`, but the orbitals and Jastrow factor are different.

  Args:
      params: A dictionary of parameters, containing all fields needed in `lapnet_orbitals`, as well as 'jastrow' used in `jastrow_factor`.
      pos (jnp.ndarray): The electron positions, a 3N dimension vector.
      atoms (jnp.ndarray): The atom positions.
      nspins (Tuple[int, ...]): Tuple with number of spin up and spin down electrons. Should have length 2.
      options (LapNetOptions, optional): Network options. Defaults to LapNetOptions().

  Returns:
      Output of antisymmetric NN in log space,
      i.e. (sign, log absolute of value).
  """

  orbitals, (_, _, r_ee) = lapnet_orbitals(
      params,
      pos,
      atoms=atoms,
      nspins=nspins,
      options=options,
  )
  output = network_blocks.logdet_matmul(orbitals)
  orbitals = orbitals[0]

  # multiply Jastrow factor
  output = output[0], output[1] + jastrow_factor(r_ee, nspins, **params['jastrow'])

  return output


def make_lapnet(
    atoms: jnp.ndarray,
    nspins: Tuple[int, int],
    charges: jnp.ndarray,
    *,
    envelope: Union[str, envelopes.EnvelopeLabel] = 'isotropic',
    bias_orbitals: bool = False,
    use_layernorm: bool = False,
    jas_w_init: float = 0.0,
    orbitals_spin_split: bool = True,

    hidden_dims: LayerArgs = ((256, 32), (256, 32), (256, 32)),
    determinants: int = 16,
    after_determinants: int = 1,

    hf_solution=None,
) -> Tuple[InitNetwork, WaveFuncLike, LapNetOptions]:
  """Creates functions for initializing parameters and evaluating lapnet.

  Args:
    atoms: (natom, ndim) array of atom positions.
    nspins: Tuple of the number of spin-up and spin-down electrons. Should have length 2.
    charges: (natom) array of atom nuclear charges.
    envelope: Envelope to use to impose orbitals go to zero at infinity.
    bias_orbitals: If true, include a bias in the final linear layer to shape
      the outputs into orbitals.
    use_layernorm: Wether or not to use layernorm in each layer.
    jas_w_int: initialization value for jastor factor
    orbitals_spin_split: If true, use different parameters for different spin in
    the envelope function and orbital.
    hidden_dims: Tuple of pairs, where each pair contains the dimension of attention
    blocks and the number of head for each layer.
    determinants: Number of determinants to use.

    after_determinants, hf_solution : place_holder

  Returns:
    init, network, options, network_each_det tuple, where init and network are callables which
    initialise the network parameters and apply the network respectively, and
    options specifies the settings used in the network. network_each_det output each determinants in lapnet.
  """

  if isinstance(envelope, str):
    envelope = envelope.upper().replace('-', '_')
    envelope_label = envelopes.EnvelopeLabel[envelope]
  else:
    # support naming scheme used in config files.
    envelope_label = envelope
  if envelope_label == envelopes.EnvelopeLabel.EXACT_CUSP:
    envelope_kwargs = {'nspins': nspins, 'charges': charges}
  else:
    envelope_kwargs = {}

  options = LapNetOptions(
    hidden_dims=hidden_dims,
    determinants=determinants,
    full_det=True,
    bias_orbitals=bias_orbitals,
    envelope_label=envelope_label,
    envelope=envelopes.get_envelope(envelope_label, **envelope_kwargs),
    atten_layers = get_multihead_list(hidden_dims,
                                      layernorm=use_layernorm),
    use_layernorm=use_layernorm,
    jas_w_init=jas_w_init,
    orbitals_spin_split=orbitals_spin_split,
  )

  init = functools.partial(
      init_lapnet_params,
      atoms=atoms,
      nspins=nspins,
      options=options
  )
  network = functools.partial(
      lapnet,
      atoms=atoms,
      nspins=nspins,
      options=options,
  )
  network_each_det = functools.partial(
      lapnet_each_det,
      atoms=atoms,
      nspins=nspins,
      options=options,
  )

  return init, network, options, network_each_det
