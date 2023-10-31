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

"""Core training loop for neural QMC in JAX."""
import functools
import importlib
import os
import shutil
import time
from typing import Optional, Sequence, Tuple, Union

import chex
import jax
import jax.numpy as jnp
import kfac_jax
import ml_collections
import numpy as np
import optax
from absl import logging
from lapnet import (
    checkpoint,
    constants,
    curvature_tags_and_blocks,
    hamiltonian,
)
from lapnet import loss as qmc_loss_functions
from lapnet import mcmc, networks, pretrain
from lapnet.utils import det_filter, multi_host, statistics, system, writers
from kfac_jax import utils as kfac_utils
from typing_extensions import Protocol


def init_electrons(
    key,
    molecule: Sequence[system.Atom],
    electrons: Sequence[int],
    batch_size: int,
    init_width=1.0,
    given_atomic_spin_configs: Sequence[Tuple[int, int]] = None
) -> jnp.ndarray:
  """Initializes electron positions around each atom.

  Args:
    key: JAX RNG state.
    molecule: system.Atom objects making up the molecule.
    electrons: tuple of number of alpha and beta electrons.
    batch_size: total number of MCMC configurations to generate across all
      devices.
    init_width: width of (atom-centred) Gaussian used to generate initial
      electron configurations.

  Returns:
    array of (batch_size, (nalpha+nbeta)*ndim) of initial (random) electron
    positions in the initial MCMC configurations and ndim is the dimensionality
    of the space (i.e. typically 3).
  """
  if given_atomic_spin_configs is None:
    print('WARNING: NO SPIN ASSIGNED')

  if sum(atom.charge for atom in molecule) != sum(electrons) and given_atomic_spin_configs is None:
    if len(molecule) == 1:
      atomic_spin_configs = [electrons]
    else:
      raise NotImplementedError('No initialization policy yet '
                                'exists for charged molecules.')
  else:
    atomic_spin_configs = [
        (atom.element.nalpha, atom.element.nbeta) for atom in molecule
    ] if given_atomic_spin_configs is None else given_atomic_spin_configs
    assert sum(sum(x) for x in atomic_spin_configs) == sum(electrons)
    while tuple(sum(x) for x in zip(*atomic_spin_configs)) != electrons:
      i = np.random.randint(len(atomic_spin_configs))
      nalpha, nbeta = atomic_spin_configs[i]
      atomic_spin_configs[i] = nbeta, nalpha

  # Assign each electron to an atom initially.
  electron_positions = []
  for i in range(2):
    for j in range(len(molecule)):
      atom_position = jnp.asarray(molecule[j].coords)
      electron_positions.append(
          jnp.tile(atom_position, atomic_spin_configs[j][i]))
  electron_positions = jnp.concatenate(electron_positions)
  # Create a batch of configurations with a Gaussian distribution about each
  # atom.
  key, subkey = jax.random.split(key)
  return (
      electron_positions +
      init_width *
      jax.random.normal(subkey, shape=(batch_size, electron_positions.size)))


# All optimizer states (KFAC and optax-based).
OptimizerState = Union[optax.OptState, kfac_jax.optimizer.OptimizerState]
OptUpdateResults = Tuple[networks.ParamTree, Optional[OptimizerState],
                         jnp.ndarray,
                         Optional[qmc_loss_functions.AuxiliaryLossData]]


class OptUpdate(Protocol):

  def __call__(self, params: networks.ParamTree,
               data: jnp.ndarray,
               opt_state: optax.OptState,
               key: chex.PRNGKey) -> OptUpdateResults:
    """Evaluates the loss and gradients and updates the parameters accordingly.

    Args:
      params: network parameters.
      data: electron positions.
      opt_state: optimizer internal state.
      key: RNG state.

    Returns:
      Tuple of (params, opt_state, loss, aux_data), where params and opt_state
      are the updated parameters and optimizer state, loss is the evaluated loss
      and aux_data auxiliary data (see AuxiliaryLossData docstring).
    """

StepResults = Tuple[jnp.ndarray, networks.ParamTree, Optional[optax.OptState],
                    jnp.ndarray, qmc_loss_functions.AuxiliaryLossData,
                    jnp.ndarray]


class Step(Protocol):

  def __call__(self,
               data: jnp.ndarray,
               params: networks.ParamTree,
               state: OptimizerState,
               key: chex.PRNGKey,
               mcmc_width: jnp.ndarray) -> StepResults:
    """Performs one set of MCMC moves and an optimization step.

    Args:
      data: batch of MCMC configurations.
      params: network parameters.
      state: optimizer internal state.
      key: JAX RNG state.
      mcmc_width: width of MCMC move proposal. See mcmc.make_mcmc_step.

    Returns:
      Tuple of (data, params, state, loss, aux_data, pmove).
        data: Updated MCMC configurations drawn from the network given the
          *input* network parameters.
        params: updated network parameters after the gradient update.
        state: updated optimization state.
        loss: energy of system based on input network parameters averaged over
          the entire set of MCMC configurations.
        aux_data: AuxiliaryLossData object also returned from evaluating the
          loss of the system.
        pmove: probability that a proposed MCMC move was accepted.
    """


def make_training_step(
    mcmc_step,
    optimizer_step: OptUpdate,
) -> Step:
  """Factory to create traning step for non-KFAC optimizers.

  Args:
    mcmc_step: Callable which performs the set of MCMC steps. See make_mcmc_step
      for creating the callable.
    optimizer_step: OptUpdate callable which evaluates the forward and backward
      passes and updates the parameters and optimizer state, as required.

  Returns:
    step, a callable which performs a set of MCMC steps and then an optimization
    update. See the Step protocol for details.
  """
  @functools.partial(constants.pmap, donate_argnums=(1, 2, 3, 4))
  def step(data: jnp.ndarray,
           params: networks.ParamTree, state: Optional[optax.OptState],
           key: chex.PRNGKey, mcmc_width: jnp.ndarray) -> StepResults:
    """A full update iteration (except for KFAC): MCMC steps + optimization."""
    # MCMC loop
    mcmc_key, loss_key = jax.random.split(key, num=2)
    data, pmove = mcmc_step(params, data, mcmc_key, mcmc_width)

    # Optimization step
    new_params, state, loss, aux_data = optimizer_step(params, data, state,
                                                       loss_key)
    return data, new_params, state, loss, aux_data, pmove

  return step


def make_kfac_training_step(mcmc_step, damping: float,
                            optimizer: kfac_jax.Optimizer) -> Step:
  """Factory to create traning step for KFAC optimizers.

  Args:
    mcmc_step: Callable which performs the set of MCMC steps. See make_mcmc_step
      for creating the callable.
    damping: value of damping to use for each KFAC update step.
    optimizer: KFAC optimizer instance.

  Returns:
    step, a callable which performs a set of MCMC steps and then an optimization
    update. See the Step protocol for details.
  """
  mcmc_step = constants.pmap(mcmc_step, donate_argnums=1)
  shared_mom = kfac_jax.utils.replicate_all_local_devices(jnp.zeros([]))
  shared_damping = kfac_jax.utils.replicate_all_local_devices(
      jnp.asarray(damping))

  def step(data: jnp.ndarray,
           params: networks.ParamTree, state: kfac_jax.optimizer.OptimizerState,
           key: chex.PRNGKey, mcmc_width: jnp.ndarray) -> StepResults:
    """A full update iteration for KFAC: MCMC steps + optimization."""
    # KFAC requires control of the loss and gradient eval, so everything called
    # here must be already pmapped.

    # MCMC loop
    mcmc_keys, loss_keys = kfac_jax.utils.p_split(key)
    new_data, pmove = mcmc_step(params, data, mcmc_keys, mcmc_width)

    # Optimization step
    new_params, state, stats = optimizer.step(
        params=params,
        state=state,
        rng=loss_keys,
        data_iterator=iter([new_data]),
        momentum=shared_mom,
        damping=shared_damping)
    return new_data, new_params, state, stats['loss'], stats['aux'], pmove

  return step


def train(cfg: ml_collections.ConfigDict, writer_manager=None):
  """Runs training loop for QMC.

  Args:
    cfg: ConfigDict containing the system and training parameters to run on. See
      base_config.default for more details.
    writer_manager: context manager with a write method for logging output. If
      None, a default writer (lapnet.utils.writers.Writer) is used.

  Raises:
    ValueError: if an illegal or unsupported value in cfg is detected.
  """
  # Device logging
  num_devices = jax.local_device_count()
  num_hosts, host_idx = jax.device_count() // num_devices, jax.process_index()
  local_batch_size = cfg.batch_size // num_hosts
  num_hosts = jax.device_count() // num_devices
  logging.info('Starting QMC with %i XLA devices per host '
               'across %i hosts.', num_devices, num_hosts)
  if local_batch_size % num_devices != 0:
    raise ValueError('Batch size must be divisible by number of devices, '
                     'got batch size {} for {} devices.'.format(
                         local_batch_size, num_devices))
  if cfg.system.ndim != 3:
    # The network (at least the input feature construction) and initial MCMC
    # molecule configuration (via system.Atom) assume 3D systems. This can be
    # lifted with a little work.
    raise ValueError('Only 3D systems are currently supported.')
  host_batch_size = cfg.batch_size // num_hosts  # batch size per host
  device_batch_size = host_batch_size // num_devices  # batch size per device
  data_shape = (num_devices, device_batch_size)

  # Check if mol is a pyscf molecule and convert to internal representation
  if cfg.system.pyscf_mol:
    cfg.update(
        system.pyscf_mol_to_internal_representation(cfg.system.pyscf_mol))

  # Convert mol config into array of atomic positions and charges
  atoms = jnp.stack([jnp.array(atom.coords) for atom in cfg.system.molecule])
  charges = jnp.array([atom.charge for atom in cfg.system.molecule])
  nspins = cfg.system.electrons

  if cfg.debug.deterministic:
    seed = 23
  else:
    seed = 1e6 * time.time()
    seed = int(multi_host.broadcast_to_hosts(seed))
  key = jax.random.PRNGKey(seed)
  # We want to make sure different host uses different keys to generate walkers.
  key = jax.random.fold_in(key, host_idx)

  if cfg.optim.el_partition_num < 0:
      cfg.optim.el_partition_num = find_el_partition_num(
        cfg,
        start=-cfg.optim.el_partition_num,
        n=3 * np.sum(cfg.system.electrons))
      print(f"Now use parititon number: {cfg.optim.el_partition_num}")

  # Create parameters, network, and vmaped/pmaped derivations

  if cfg.pretrain.method == 'direct_init' or (
      cfg.pretrain.method == 'hf' and cfg.pretrain.iterations > 0):
    hartree_fock = pretrain.get_hf(
        pyscf_mol=cfg.system.get('pyscf_mol'),
        molecule=cfg.system.molecule,
        nspins=nspins,
        restricted=False,
        basis=cfg.pretrain.basis)
    # broadcast the result of PySCF from host 0 to all other hosts
    hartree_fock.mean_field.mo_coeff = tuple([
        multi_host.broadcast_to_hosts(x)
        for x in hartree_fock.mean_field.mo_coeff
    ])

  hf_solution = hartree_fock if cfg.pretrain.method == 'direct_init' else None

  (network_init, signed_network,
   network_options, network_each_det) = networks.network_provider(cfg)(
      atoms, nspins, charges, hf_solution=hf_solution,)

  params_initialization_key = get_params_initialization_key(cfg.debug.deterministic)
  params = network_init(params_initialization_key)
  params = kfac_utils.replicate_all_local_devices(params)
  # Often just need log|psi(x)|.
  network = lambda *args, **kwargs: signed_network(*args, **kwargs)[1]  # type: networks.LogWaveFuncLike
  batch_network = jax.vmap(
      network, in_axes=(None, 0), out_axes=0)  # batched network
  # Set up checkpointing and restore params/data if necessary
  # Mirror behaviour of checkpoints in TF FermiNet.
  # Checkpoints are saved to save_path.
  # When restoring, we first check for a checkpoint in save_path. If none are
  # found, then we check in restore_path.  This enables calculations to be
  # started from a previous calculation but then resume from their own
  # checkpoints in the event of pre-emption.

  ckpt_save_path = checkpoint.create_save_path(cfg.log.save_path)
  ckpt_restore_path = checkpoint.get_restore_path(cfg.log.restore_path)

  # If restore path is available, we copy everything in restore path over to save path
  # so that it's easier to manage
  if ckpt_restore_path:
      for _f in os.listdir(ckpt_restore_path):
          src_filename = os.path.join(ckpt_restore_path, _f)
          logging.info(f'Copying {src_filename} to {ckpt_save_path}')
          shutil.copy(src_filename, ckpt_save_path)

  ckpt_restore_filename = checkpoint.find_last_checkpoint(ckpt_save_path)

  if ckpt_restore_filename:
    t_init, data, params, opt_state_ckpt, mcmc_width_ckpt, sharded_key_ckpt = checkpoint.restore(
        ckpt_restore_filename, local_batch_size)
  else:
    logging.info('No checkpoint found. Training new model.')
    key, subkey = jax.random.split(key)
    # make sure data on each host is initialized differently
    subkey = jax.random.fold_in(subkey, jax.process_index())
    data = init_electrons(subkey, cfg.system.molecule, cfg.system.electrons,
                          local_batch_size,
                          init_width=cfg.mcmc.init_width,
                          given_atomic_spin_configs=cfg.system.atom_spin_configs \
                            if hasattr(cfg.system, 'atom_spin_configs') \
                            else None,
                          )
    data = jnp.reshape(data, data_shape + data.shape[1:])
    data = kfac_jax.utils.broadcast_all_local_devices(data)
    t_init = 0
    opt_state_ckpt = None
    mcmc_width_ckpt = None
    sharded_key_ckpt = None

  # Set up logging
  train_schema = ['step', 'energy', 'var', 'ewmean', 'ewvar', 'pmove', 'num_outliers', 'num_det']

  # Initialization done. We now want to have different PRNG streams on each
  # device. Shard the key over devices
  tmp_sharded_key = kfac_jax.utils.make_different_rng_key_on_all_devices(key)
  sharded_key = tmp_sharded_key \
                if sharded_key_ckpt is None or tmp_sharded_key.shape != sharded_key_ckpt.shape \
                else sharded_key_ckpt

  # Pretraining to match Hartree-Fock

  if (t_init == 0 and cfg.pretrain.method == 'hf' and
      cfg.pretrain.iterations > 0):
    orbitals = functools.partial(
        networks.network_orbital_provider(cfg),
        atoms=atoms,
        nspins=cfg.system.electrons,
        options=network_options,
    )
    batch_orbitals = jax.vmap(
        lambda params, data: orbitals(params, data)[0],
        in_axes=(None, 0),
        out_axes=0)
    sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
    sampling_function = pretrain.make_HF_ansatz(hartree_fock, cfg.system.electrons) if cfg.pretrain.use_hf_sample else batch_network
    params, data = pretrain.pretrain_hartree_fock(
        params=params,
        data=data,
        batch_network=sampling_function,
        batch_orbitals=batch_orbitals,
        network_options=network_options,
        sharded_key=subkeys,
        atoms=atoms,
        electrons=cfg.system.electrons,
        scf_approx=hartree_fock,
        iterations=cfg.pretrain.iterations,
        burn_in_iters=cfg.pretrain.burn_in_iters,
        optim=cfg.pretrain.optim)

  # Main training

  # Construct MCMC step
  atoms_to_mcmc = atoms if cfg.mcmc.scale_by_nuclear_distance else None
  mcmc_step = mcmc.make_mcmc_step(
      batch_network,
      local_batch_size // num_devices,
      steps=cfg.mcmc.steps,
      blocks=cfg.mcmc.blocks,
      atoms=atoms_to_mcmc
  )
  # Construct loss and optimizer
  if cfg.system.make_local_energy_fn:
    local_energy_module, local_energy_fn = (
        cfg.system.make_local_energy_fn.rsplit('.', maxsplit=1))
    local_energy_module = importlib.import_module(local_energy_module)
    make_local_energy = getattr(local_energy_module, local_energy_fn)  # type: hamiltonian.MakeLocalEnergy
    local_energy = make_local_energy(
        f=signed_network,
        atoms=atoms,
        charges=charges,
        nspins=nspins,
        use_scan=False,
        **cfg.system.make_local_energy_kwargs)
  else:
    local_energy = hamiltonian.local_energy(
        f=signed_network,
        atoms=atoms,
        charges=charges,
        nspins=nspins,
        use_scan=False,
        forward_laplacian=cfg.optim.forward_laplacian)
  total_energy = qmc_loss_functions.make_loss(
      network,
      local_energy,
      clip_local_energy=cfg.optim.clip_el,
      rm_outlier=cfg.optim.rm_outlier,
      el_partition=cfg.optim.el_partition_num,
      local_energy_outlier_width=cfg.optim.local_energy_outlier_width)
  # Compute the learning rate
  def learning_rate_schedule(t_: jnp.ndarray) -> jnp.ndarray:
    fg = 1.0 * (t_ >= cfg.optim.lr.warmup)
    orig_lr = cfg.optim.lr.rate * jnp.power(
          (1.0 / (1.0 + fg * (t_ - cfg.optim.lr.warmup)/cfg.optim.lr.delay)), cfg.optim.lr.decay)
    linear_lr = cfg.optim.lr.rate * t_ / (cfg.optim.lr.warmup + (cfg.optim.lr.warmup == 0.0))
    return fg * orig_lr + (1 - fg) * linear_lr

  # Differentiate wrt parameters (argument 0)
  val_and_grad = jax.value_and_grad(total_energy, argnums=0, has_aux=True)


  # Construct and setup optimizer
  def init_step(cfg, params, data, sharded_key, opt_state_ckpt=None):
    if cfg.optim.optimizer == 'none':
      optimizer = None
    elif cfg.optim.optimizer == 'adam':
      optimizer = optax.chain(
        optax.scale_by_adam(**cfg.optim.adam),
        optax.scale_by_schedule(learning_rate_schedule),
        optax.scale(-1.))
    elif cfg.optim.optimizer == 'lamb':
      optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.scale_by_adam(eps=1e-7),
        optax.scale_by_trust_ratio(),
        optax.scale_by_schedule(learning_rate_schedule),
        optax.scale(-1))
    elif cfg.optim.optimizer == 'kfac':
      optimizer = kfac_jax.Optimizer(
        val_and_grad,
        l2_reg=cfg.optim.kfac.l2_reg,
        norm_constraint=cfg.optim.kfac.norm_constraint,
        value_func_has_aux=True,
        value_func_has_rng=True,
        learning_rate_schedule=learning_rate_schedule,
        curvature_ema=cfg.optim.kfac.cov_ema_decay,
        inverse_update_period=cfg.optim.kfac.invert_every,
        min_damping=cfg.optim.kfac.min_damping,
        num_burnin_steps=0,
        register_only_generic=cfg.optim.kfac.register_only_generic,
        estimation_mode='fisher_exact',
        multi_device=True,
        pmap_axis_name=constants.PMAP_AXIS_NAME,
        auto_register_kwargs=dict(
            graph_patterns=curvature_tags_and_blocks.GRAPH_PATTERNS,
        ),
        # debug=True
      )
      sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
      opt_state = optimizer.init(params, subkeys, data)
      opt_state = opt_state_ckpt or opt_state  # avoid overwriting ckpted state
    else:
      raise ValueError(f'Not a recognized optimizer: {cfg.optim.optimizer}')

    if not optimizer:
      opt_state = None

      def energy_eval(params: networks.ParamTree, data: jnp.ndarray,
                      opt_state: Optional[optax.OptState],
                      key: chex.PRNGKey) -> OptUpdateResults:
        loss, aux_data = total_energy(params, key, data)
        return params, opt_state, loss, aux_data

      step = make_training_step(
          mcmc_step=mcmc_step,
          optimizer_step=energy_eval)

    elif isinstance(optimizer, optax.GradientTransformation):
      # optax/optax-compatible optimizer (ADAM, LAMB, ...)
      opt_state = jax.pmap(optimizer.init)(params)
      opt_state = opt_state_ckpt or opt_state  # avoid overwriting ckpted state

      def opt_update(params: networks.ParamTree, data: jnp.ndarray,
                      opt_state: Optional[optax.OptState],
                      key: chex.PRNGKey) -> OptUpdateResults:
        (loss, aux_data), grad = val_and_grad(params, key, data)
        grad = constants.pmean(grad)
        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, aux_data

      step = make_training_step(mcmc_step=mcmc_step, optimizer_step=opt_update)
    elif isinstance(optimizer, kfac_jax.Optimizer):
      step = make_kfac_training_step(
          mcmc_step=mcmc_step,
          damping=cfg.optim.kfac.damping,
          optimizer=optimizer)
    else:
      raise ValueError(f'Unknown optimizer: {optimizer}')

    return step, opt_state, sharded_key


  step, opt_state, sharded_key = init_step(cfg, params, data, sharded_key,
                                           opt_state_ckpt=opt_state_ckpt)

  if mcmc_width_ckpt is not None:
    mcmc_width = kfac_jax.utils.replicate_all_local_devices(mcmc_width_ckpt[0])
  else:
    mcmc_width = kfac_jax.utils.replicate_all_local_devices(
        jnp.asarray(cfg.mcmc.move_width))
  pmoves = np.zeros(cfg.mcmc.adapt_frequency)

  if t_init == 0 or cfg.mcmc.force_burn_in:
    logging.info('Burning in MCMC chain for %d steps', cfg.mcmc.burn_in)

    def null_update(params: networks.ParamTree, data: jnp.ndarray,
                    opt_state: Optional[optax.OptState],
                    key: chex.PRNGKey) -> OptUpdateResults:
      del data, key
      return params, opt_state, jnp.zeros(1), None

    burn_in_step = make_training_step(
        mcmc_step=mcmc_step, optimizer_step=null_update)

    for t in range(cfg.mcmc.burn_in):
      sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
      data, params, *_ = burn_in_step(
          data=data,
          params=params,
          state=None,
          key=subkeys,
          mcmc_width=mcmc_width)
    logging.info('Completed burn-in MCMC steps')
    sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
    ptotal_energy = constants.pmap(total_energy)
    initial_energy, _ = ptotal_energy(params, subkeys, data)
    logging.info('Initial energy: %03.4f E_h', initial_energy[0])

  time_of_last_ckpt = time.time()
  should_save_ckpt = make_should_save_ckpt(cfg)
  weighted_stats = None

  if cfg.optim.optimizer == 'none' and opt_state_ckpt is not None:
    # If opt_state_ckpt is None, then we're restarting from a previous inference
    # run (most likely due to preemption) and so should continue from the last
    # iteration in the checkpoint. Otherwise, starting an inference run from a
    # training run.
    logging.info('No optimizer provided. Assuming inference run.')
    logging.info('Setting initial iteration to 0.')
    t_init = 0

  if writer_manager is None:
      writer_manager = writers.Writer(
          name=cfg.log.stats_file_name,
          schema=train_schema,
          directory=ckpt_save_path,
          iteration_key=None,
        log=False)

  with writer_manager as writer:
    # Main training loop
    if cfg.network.full_det:
      num_det = params['orbital'][0]['w'].shape[-1] // sum(nspins)
    else:
      num_det = params['orbital'][0]['w'].shape[-1] // nspins[0]

    if cfg.network.det_filter.step > 0 and cfg.optim.optimizer:
      logging.info('number of det will be updated per %d steps', cfg.network.det_filter.step)
    else:
      logging.info('number of det will not be changed.')

    for t in range(t_init, cfg.optim.iterations):
      init_time = time.time()
      # Utilize det-filter only in the training for a positive interval.
      if cfg.network.det_filter.step > 0. and cfg.optim.optimizer:
        if t % cfg.network.det_filter.step == 0:
          params, opt_state, step, num_det, sharded_key = det_filter.filtering(
             network_each_det, init_step, params, opt_state,
             data, sharded_key, cfg, step, num_det
          )

      sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
      data, params, opt_state, loss, aux_data, pmove = step(
          data=data,
          params=params,
          state=opt_state,
          key=subkeys,
          mcmc_width=mcmc_width)

      # due to pmean, loss, and pmove should be the same across
      # devices.
      loss = loss[0]
      # per batch variance isn't informative. Use weighted mean and variance
      # instead.
      weighted_stats = statistics.exponentialy_weighted_stats(
          alpha=0.1, observation=loss, previous_stats=weighted_stats)

      batch_var = aux_data.variance[0]
      pmove = pmove[0]
      num_outliers = aux_data.outlier_mask.size - jnp.sum(aux_data.outlier_mask)

      end_time = time.time()

      # Update MCMC move width
      if t > 0 and t % cfg.mcmc.adapt_frequency == 0:
        if np.mean(pmoves) > 0.55:
          mcmc_width *= 1.1
        if np.mean(pmoves) < 0.5:
          mcmc_width /= 1.1
        pmoves[:] = 0
      pmoves[t%cfg.mcmc.adapt_frequency] = pmove

      # Logging
      if t % cfg.log.stats_frequency == 0:

        logging.info(
            'Step %05d: %03.4f E_h, batch_var=%03.4f, exp. variance=%03.4f E_h^2, pmove=%0.2f, num_outliers=%d, num_det=%d, time=%.2f',
            t, loss, batch_var, weighted_stats.variance, pmove, num_outliers, num_det, end_time - init_time)
        writer.write(
            t,
            step=t,
            energy=np.asarray(loss),
            var=np.asarray(batch_var),
            ewmean=np.asarray(weighted_stats.mean),
            ewvar=np.asarray(weighted_stats.variance),
            pmove=np.asarray(pmove),
            num_outliers=np.asarray(num_outliers),
            num_det=np.asarray(num_det))

      # Checkpointing
      if should_save_ckpt(t, time_of_last_ckpt):
          # no checkpointing in inference mode
          if cfg.optim.optimizer != 'none':
            checkpoint.save(ckpt_save_path, t, data, params, opt_state, mcmc_width, sharded_key)
          time_of_last_ckpt = time.time()


def make_test_cfg(_cfg):
    cfg = _cfg.copy_and_resolve_references()
    # Just do 2 iterations of pretrain and training to see if OOM
    cfg.optim.iterations = 2
    cfg.pretrain.iterations = 2
    cfg.mcmc.steps = 10
    cfg.mcmc.burn_in = 10
    # Clear all the ckpt related path. No meaningful ckpt or log will be
    # loaded or written
    cfg.log.save_path = ""
    cfg.log.restore_path = ""

    # We don't do connection when testing
    cfg.multi_host = False
    return cfg


def find_el_partition_num(cfg, start, n):
    test_cfg = make_test_cfg(cfg)
    # Only try the divisor of n up to "n / 2", if all failed then fall back to 0
    for i in range(start, n // 2):
        if n % i != 0:
            continue

        try:
            test_cfg.optim.el_partition_num = i
            train(test_cfg)
        except Exception as e:
            print(f'failed partition {i} due to {e}')
        else:
            return i

    return 0


def get_params_initialization_key(deterministic):
  '''
  The key point here is to make sure different hosts uses the same RNG key
  to initialize network parameters.
  '''
  if deterministic:
    seed = 888
  else:

    # We make sure different hosts get the same seed.
    local_seed = time.time()
    float_seed = kfac_utils.compute_mean(jnp.ones(jax.local_device_count()) * local_seed)[0]
    seed = int(1e6 * float_seed)
  print(f'params initialization seed: {seed}')
  return jax.random.PRNGKey(seed)


def make_should_save_ckpt(cfg):
    '''
    A factory for the method determining if saving ckpt.
    '''

    def should_save_ckpt(iteration, time_of_last_ckpt):
          return  (
              time.time() - time_of_last_ckpt > cfg.log.save_frequency * 60 or
              iteration >= cfg.optim.iterations - 1 or
              (cfg.log.save_frequency_in_step > 0 and iteration % cfg.log.save_frequency_in_step == 0))

    return should_save_ckpt
