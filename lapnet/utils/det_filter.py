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

import copy

import jax
import jax.numpy as jnp
import kfac_jax
import numpy as np
from absl import logging
from lapnet import constants


def filter_idx(is_vanished):
  idx_domination = []
  for idx, vanish in enumerate(is_vanished):
    if not vanish:
      idx_domination.append(idx)
  return idx_domination


def params_pick(params, idx, num_det):
  shapes = list(params.shape[:-1])
  shapes.append(num_det)
  shapes.append(-1)
  if params.shape[-1] % num_det:
    raise ValueError('The orbital in params cannot be divided equally '
                      f'into pieces into the number of det {num_det}')
  params = params.reshape(shapes)
  params_picked = params[..., idx, :]
  shapes = shapes[:-2]
  shapes.append(-1)
  params_picked = params_picked.reshape(shapes)
  return params_picked


def filtering(network_each_det,
              init_step,
              params,
              opt_state,
              data,
              sharded_key,
              cfg,
              step,
              num_det,):

  network_each_det_pmapped = constants.pmap(
    jax.vmap(network_each_det, in_axes=(None, 0), out_axes=0))
  full_individual_logdet = network_each_det_pmapped(params, data)

  # Check whether num_det is the number of configurations in the wavefunction.
  if full_individual_logdet.shape[-1] != num_det:
    raise ValueError(f'Number of det inconsistent, supposed to be: {num_det}, '
                     f'calculated to be {full_individual_logdet.shape[-1]}')

  # Reshape the logdet as (num_walker, num_det)
  full_individual_logdet = full_individual_logdet.reshape((-1, num_det))

  # Obtain the lower and upper bond for each configurations.
  upper_bond = jnp.percentile(
    full_individual_logdet, 100 - cfg.network.det_filter.percent, axis=0)
  lower_bond = jnp.percentile(
    full_individual_logdet, cfg.network.det_filter.percent, axis=0)

  # List upper bond in row and lower bond in column.
  upper_bond = upper_bond.reshape((1, -1))
  lower_bond = lower_bond.reshape((-1, 1))

  # Find vanished configuration. If there exists one lower bond, subtracted by
  # the threshold, still bigger than the upper bond, the det should vanish.
  # Do the summation for each upper bond along the lower bond.
  # If it is True for some lower bonds, the configuration should vanish.
  is_vanished = jnp.sum(
    upper_bond < (lower_bond - cfg.network.det_filter.threshold), axis=0
  )

  # If there are vanished configurations, prune the orbital parameters and init optimizer.
  if not jnp.sum(is_vanished):
    logging.info('No need to filter.')
    return params, opt_state, step, num_det, sharded_key


  logging.info('Det will be filtered.')
  idx_survive = filter_idx(is_vanished)
  params['orbital'] = jax.tree_util.tree_map(
    lambda x: params_pick(x, idx=idx_survive, num_det=num_det), params['orbital'])

  params['envelope'] = jax.tree_util.tree_map(
    lambda x: params_pick(x, idx=idx_survive, num_det=num_det), params['envelope'])

  sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
  step_counter = opt_state.step_counter
  step, opt_state, sharded_key = init_step(cfg, params, data, subkeys, opt_state_ckpt=None)
  opt_state.step_counter = step_counter
  num_det = len(idx_survive)
  return params, opt_state, step, num_det, sharded_key
