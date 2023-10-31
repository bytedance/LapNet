# Copyright 2020 DeepMind Technologies Limited.
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

"""Utilities for pretraining and importing PySCF models."""

from typing import Callable, Optional, Sequence, Tuple, Union

from absl import logging
import chex
import functools
from lapnet import constants
from lapnet import envelopes
from lapnet import mcmc
from lapnet import networks
from lapnet.utils import scf
from lapnet.utils import system
import jax
from jax import numpy as jnp
import kfac_jax
import numpy as np
import optax
import pyscf


# Given the parameters and electron positions, return arrays(s) of the orbitals.
# See networks.fermi_net_orbitals. (Note only the orbitals, and not envelope
# parameters, are required.)
FermiNetOrbitals = Callable[[networks.ParamTree, jnp.ndarray],
                            Sequence[jnp.ndarray]]


def get_hf(molecule: Optional[Sequence[system.Atom]] = None,
           nspins: Optional[Tuple[int, int]] = None,
           basis: Optional[str] = 'sto-3g',
           pyscf_mol: Optional[pyscf.gto.Mole] = None,
           restricted: Optional[bool] = False) -> scf.Scf:
  """Returns an Scf object with the Hartree-Fock solution to the system.

  Args:
    molecule: the molecule in internal format.
    nspins: tuple with number of spin up and spin down electrons.
    basis: basis set to use in Hartree-Fock calculation.
    pyscf_mol: pyscf Mole object defining the molecule. If supplied,
      molecule, nspins and basis are ignored.
    restricted: If true, perform a restricted Hartree-Fock calculation,
      otherwise perform an unrestricted Hartree-Fock calculation.
  """
  if pyscf_mol:
    scf_approx = scf.Scf(pyscf_mol=pyscf_mol, restricted=restricted)
  else:
    scf_approx = scf.Scf(
        molecule, nelectrons=nspins, basis=basis, restricted=restricted)
  scf_approx.run()
  return scf_approx


def eval_orbitals(scf_approx: scf.Scf, pos: Union[np.ndarray, jnp.ndarray],
                  nspins: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
  """Evaluates SCF orbitals from PySCF at a set of positions.

  Args:
    scf_approx: an scf.Scf object that contains the result of a PySCF
      calculation.
    pos: an array of electron positions to evaluate the orbitals at, of shape
      (..., nelec*3), where the leading dimensions are arbitrary, nelec is the
      number of electrons and the spin up electrons are ordered before the spin
      down electrons.
    nspins: tuple with number of spin up and spin down electrons.

  Returns:
    tuple with matrices of orbitals for spin up and spin down electrons, with
    the same leading dimensions as in pos.
  """
  if not isinstance(pos, jnp.ndarray):  # works even with JAX array
    try:
      pos = pos.copy()
    except AttributeError as exc:
      raise ValueError('Input must be either NumPy or JAX array.') from exc
  leading_dims = pos.shape[:-1]
  # split into separate electrons
  pos = jnp.reshape(pos, [-1, 3])  # (batch*nelec, 3)

  # use pure_callback for scf_approx.eval_mos(pos) which is implemented using numpy
  if scf_approx.restricted:
    nbasis = scf_approx.mean_field.mo_coeff.shape[-1]
  else:
    nbasis = scf_approx.mean_field.mo_coeff[0].shape[-1]
  results_shape = jax.core.ShapedArray((pos.shape[0], nbasis), pos.dtype)
  mos = jax.pure_callback(scf_approx.eval_mos, (results_shape, results_shape), pos)  # (batch*nelec, nbasis), (batch*nelec, nbasis)

  # Reshape into (batch, nelec, nbasis) for each spin channel.
  mos = [jnp.reshape(mo, leading_dims + (sum(nspins), -1)) for mo in mos]
  # Return (using Aufbau principle) the matrices for the occupied alpha and
  # beta orbitals. Number of alpha electrons given by nspins[0].
  alpha_spin = mos[0][..., :nspins[0], :nspins[0]]
  beta_spin = mos[1][..., nspins[0]:, :nspins[1]]
  return alpha_spin, beta_spin


def jax_eval_orbitals(scf_approx: scf.Scf, pos: Union[np.ndarray, jnp.ndarray],
                  nspins: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
  """Evaluates SCF orbitals from PySCF at a set of positions.

  Args:
    scf_approx: an scf.Scf object that contains the result of a PySCF
      calculation.
    pos: an array of electron positions to evaluate the orbitals at, of shape
      (..., nelec*3), where the leading dimensions are arbitrary, nelec is the
      number of electrons and the spin up electrons are ordered before the spin
      down electrons.
    nspins: tuple with number of spin up and spin down electrons.

  Returns:
    tuple with matrices of orbitals for spin up and spin down electrons, with
    the same leading dimensions as in pos.
  """
  if not isinstance(pos, jnp.ndarray):  # works even with JAX array
    try:
      pos = pos.copy()
    except AttributeError as exc:
      raise ValueError('Input must be either NumPy or JAX array.') from exc
  alpha_spin, beta_spin = scf_approx.vmap_jax_scf(pos)
  return alpha_spin, beta_spin


def eval_slater(scf_approx: scf.Scf, pos: Union[jnp.ndarray, np.ndarray],
                nspins: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
  """Evaluates the Slater determinant.

  Args:
    scf_approx: an object that contains the result of a PySCF calculation.
    pos: an array of electron positions to evaluate the orbitals at.
    nspins: tuple with number of spin up and spin down electrons.

  Returns:
    tuple with sign and log absolute value of Slater determinant.
  """
  matrices = jax_eval_orbitals(scf_approx, pos, nspins)
  slogdets = [jnp.linalg.slogdet(elem) for elem in matrices]
  sign_alpha, sign_beta = [elem[0] for elem in slogdets]
  log_abs_wf_alpha, log_abs_wf_beta = [elem[1] for elem in slogdets]
  log_abs_slater_determinant = log_abs_wf_alpha + log_abs_wf_beta
  sign = sign_alpha * sign_beta
  return sign, log_abs_slater_determinant


def make_pretrain_step(batch_envelope_fn,
                       batch_orbitals: FermiNetOrbitals,
                       batch_network: networks.LogWaveFuncLike,
                       optimizer_update: optax.TransformUpdateFn,
                       full_det: bool = False):
  """Creates function for performing one step of Hartre-Fock pretraining.

  Args:
    batch_envelope_fn: callable with signature f(params, data) which, given a
      batch of electron positions and the tree of envelope network parameters,
      returns the multiplicative envelope to apply to the orbitals. See envelope
      functions in networks for details. Only required if the envelope is not
      included in batch_orbitals.
    batch_orbitals: callable with signature f(params, data), which given network
      parameters and a batch of electron positions, returns the orbitals in
      the network evaluated at those positions.
    batch_network: callable with signature f(params, data), which given network
      parameters and a batch of electron positions, returns the log of the
      magnitude of the (wavefunction) network  evaluated at those positions.
    optimizer_update: callable for transforming the gradients into an update (ie
      conforms to the optax API).
    full_det: If true, evaluate all electrons in a single determinant.
      Otherwise, evaluate products of alpha- and beta-spin determinants.

  Returns:
    Callable for performing a single pretraining optimisation step.
  """

  def pretrain_step(data, target, params, state, key, logprob):
    """One iteration of pretraining to match HF."""
    n = jnp.array([tgt.shape[-1] for tgt in target]).sum()

    def loss_fn(p, x, target):
      env = jnp.exp(batch_envelope_fn(p['envelope'], x) / n)
      env = jnp.reshape(env, [env.shape[-1], 1, 1, 1])
      if full_det:
        ndet = target[0].shape[0]
        na = target[0].shape[1]
        nb = target[1].shape[1]
        target = jnp.concatenate(
            (jnp.concatenate((target[0], jnp.zeros((ndet, na, nb))), axis=-1),
             jnp.concatenate((jnp.zeros((ndet, nb, na)), target[1]), axis=-1)),
            axis=-2)
        result = jnp.mean(
            (target[:, None, ...] - env * batch_orbitals(p, x)[0])**2)
      else:
        result = jnp.array([
            jnp.mean((t[:, None, ...] - env * o)**2)
            for t, o in zip(target, batch_orbitals(p, x))
        ]).sum()
      return constants.pmean(result)

    val_and_grad = jax.value_and_grad(loss_fn, argnums=0)
    loss_val, search_direction = val_and_grad(params, data, target)
    search_direction = constants.pmean(search_direction)
    updates, state = optimizer_update(search_direction, state, params)
    params = optax.apply_updates(params, updates)
    data, key, logprob, _ = mcmc.mh_update(params, batch_network, data, key,
                                           logprob, 0)
    return data, params, state, loss_val, logprob

  return pretrain_step

def make_pretrain_burn_in_step(batch_network: networks.LogWaveFuncLike,):

  def burn_in_step(data, target, params, state, key, logprob):
    data, key, logprob, _ = mcmc.mh_update(params, batch_network, data, key,
                                           logprob, 0)
    return data, params, None, None, logprob

  return burn_in_step


def pretrain_hartree_fock(
    *,
    params: networks.ParamTree,
    data: jnp.ndarray,
    batch_network: networks.WaveFuncLike,
    batch_orbitals: FermiNetOrbitals,
    network_options: networks.Options,
    sharded_key: chex.PRNGKey,
    atoms: jnp.ndarray,
    electrons: Tuple[int, int],
    scf_approx: scf.Scf,
    iterations: int = 1000,
    burn_in_iters: int = 5000,
    logger: Optional[Callable[[int, float], None]] = None,
    optim: str='adam',
):
  """Performs training to match initialization as closely as possible to HF.

  Args:
    params: Network parameters.
    data: MCMC configurations.
    batch_network: callable with signature f(params, data), which given network
      parameters and a batch of electron positions, returns the log of the
      magnitude of the (wavefunction) network  evaluated at those positions.
    batch_orbitals: callable with signature f(params, data), which given network
      parameters and a batch of electron positions, returns the orbitals in
      the network evaluated at those positions.
    network_options: FermiNet network options.
    sharded_key: JAX RNG state (sharded) per device.
    atoms: (natom, 3) array of atom positions.
    electrons: tuple of number of electrons of each spin.
    scf_approx: an scf.Scf object that contains the result of a PySCF
      calculation.
    iterations: number of pretraining iterations to perform.
    logger: Callable with signature (step, value) which externally logs the
      pretraining loss.

  Returns:
    params, data: Updated network parameters and MCMC configurations such that
    the orbitals in the network closely match Hartree-Foch and the MCMC
    configurations are drawn from the log probability of the network.
  """
  # Implementing the basis set in JAX would enable using GPUs and allow
  # eval_orbitals to be pmapped.
  if optim == 'adam':
    optimizer = optax.adam(3.e-4)
  elif optim == 'lamb':
    optimizer = optax.lamb(1e-3)
  else:
    raise NotImplementedError

  opt_state_pt = constants.pmap(optimizer.init)(params)

  if (network_options.envelope.apply_type ==
      envelopes.EnvelopeType.POST_DETERMINANT):

    def envelope_fn(params, x):
      ae, r_ae, _, r_ee = networks.construct_input_features(x, atoms)
      return network_options.envelope.apply(
          ae=ae, r_ae=r_ae, r_ee=r_ee, **params)
  else:
    envelope_fn = lambda p, x: 0.0
  batch_envelope_fn = jax.vmap(envelope_fn, (None, 0))

  pretrain_step = make_pretrain_step(
      batch_envelope_fn,
      batch_orbitals,
      batch_network,
      optimizer.update,
      full_det=network_options.full_det)
  pretrain_step = constants.pmap(pretrain_step)

  burn_in_step = make_pretrain_burn_in_step(
    batch_network,
  )
  burn_in_step = constants.pmap(burn_in_step)

  pnetwork = constants.pmap(batch_network)
  logprob = 2. * pnetwork(params, data)

  # pretrain burn in step
  for t in range(burn_in_iters):
    sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
    data, params, _, _, logprob = burn_in_step(
        data, None, params, None, subkeys, logprob)
    if t % 1000 == 0:
      logging.info('Pretrain burn in iter %05d, logprob, %.2f', t, jnp.mean(logprob))
    if logger:
      logger(t, loss[0])

  for t in range(iterations):
    target = jax.pmap(jax_eval_orbitals,
                      in_axes=(None, 0, None),
                      static_broadcasted_argnums=(0, 2))(scf_approx, data, electrons)
    sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
    data, params, opt_state_pt, loss, logprob = pretrain_step(
        data, target, params, opt_state_pt, subkeys, logprob)
    if t % 100 == 0:
      logging.info('Pretrain iter %05d: %g, logprob: %.2f', t, loss[0], jnp.mean(logprob))
    if logger:
      logger(t, loss[0])
  return params, data


def make_HF_ansatz(scf_approx: scf.Scf, electrons: Tuple[int, int]):
  """ For drawing samples from the HF orbitals,
      HF_sampling evaluates SCF orbitals from PySCF at a set of positions and computes corresponding Slater determinants.

  Args:
    scf_approx: an scf.Scf object that contains the result of a PySCF calculation.
    electrons: tuple with number of spin up and spin down electrons.
  """
  sampling_func = functools.partial(eval_slater, scf_approx=scf_approx, nspins=electrons)

  def HF_sampling(params: networks.ParamTree, data: Union[jnp.ndarray, np.ndarray]):
    """
      Args:
        params: unused
        data: MCMC configurations.

      Returns: ln |psi|
    """
    return sampling_func(pos=data)[1]

#  batch_sampling = jax.vmap(HF_sampling, in_axes=(None, 0), out_axes=0)
#  return batch_sampling
  return HF_sampling
