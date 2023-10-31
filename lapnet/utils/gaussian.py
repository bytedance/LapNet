# Copyright 2023 Bytedance Ltd. and/or its affiliate
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

from functools import lru_cache

import pyscf.pbc.gto

import jax.numpy as jnp
import haiku as hk
import jax
import numpy as np


@lru_cache(maxsize=16)
def get_cartesian_angulars(l):
    return [
        (lx, ly, l - lx - ly) for lx in range(l, -1, -1) for ly in range(l - lx, -1, -1)
    ]

def pow_int(xs, exps):
    xs_tile = jnp.tile(xs, [len(exps), 1])
    zs = xs_tile ** exps
    return zs


class GTOBasis(hk.Module):

    def __init__(self,
                 centers,
                 shells,
                 mol: pyscf.gto.Mole,
                 cart=False,
                 name=None):
        super(GTOBasis, self).__init__(name=name)
        self.centers = centers
        self.center_idxs, shells = zip(*shells)
        self.shells = shells
        self.normalization = jnp.diagonal(mol.intor('int1e_ovlp_cart')) ** 0.5
        self.cart = cart
        self.cart2sph = jnp.asarray(mol.cart2sph_coeff(normalized='sp'))


    def __len__(self):
        return sum(map(len, self.shells))

    def items(self):
        return zip(self.center_idxs, self.shells)

    @classmethod
    def from_pyscf(cls, mol, cart=False, name=None):
        centers = jnp.asarray(mol.atom_coords())
        shells = []
        for i in range(mol.nbas):
            l = mol.bas_angular(i)
            i_atom = mol.bas_atom(i)
            zetas = jnp.asarray(mol.bas_exp(i))
            coeff_sets = jnp.asarray(mol.bas_ctr_coeff(i).T)
            for coeffs in coeff_sets:
                shells.append((i_atom, GTOShell(l, coeffs, zetas)))
        return cls(centers, shells, mol, cart, name=name)

    def eval_ao(self, x):
        """

        :param x: electron coord with shape [3]
        :return: aos with shape [nao]
        """

        shells = [sh(x-self.centers[idx]) for idx, sh in self.items()]
        res = jnp.concatenate(shells, axis=-1) * self.normalization
        if not self.cart:
            res = jnp.dot(res, self.cart2sph)
        return res

    def eval_laps(self, x):
        shells = [sh.eval_laps(x - self.centers[idx]) for idx, sh in self.items()]
        res = jnp.concatenate(shells, axis=-1) * self.normalization
        if not self.cart:
            res = jnp.dot(res, self.cart2sph)
        return res

    def eval_grads(self, x):
        shells = [sh.eval_grads(x - self.centers[idx]) for idx, sh in self.items()]
        res = jnp.concatenate(shells, axis=-1) * self.normalization
        if not self.cart:
            res = jnp.dot(res, self.cart2sph)
        return res

    def __call__(self, x):
        return self.eval_ao(x)


from scipy.special import factorial2


class GTOShell(hk.Module):

    def __init__(self, l, coeffs, zetas, name=None):
        super(GTOShell, self).__init__(name=name)
        self.ls = np.asarray(get_cartesian_angulars(l))
        anorms = 1 / jnp.sqrt(factorial2(2 * self.ls - 1).prod(-1))
        self.anorms = jnp.asarray(anorms)
        rnorms = (2 * zetas / jnp.pi) ** (3 / 4) * (4 * zetas) ** (l / 2)
        coeffs = rnorms * coeffs
        self.coeffs = coeffs
        self.zetas = zetas

    def __len__(self):
        return len(self.ls)

    @property
    def l(self):
        return self.ls[0][0]

    def extra_repr(self):
        return f'l={self.l}, n_primitive={len(self.zetas)}'

    def eval_ao(self, rs):
        """

        :param rs:[3]
        :return: [nao]
        """
        rs_2 = jnp.sum(rs ** 2, axis=-1)
        angulars = pow_int(rs[None, :], self.ls).prod(axis=-1)
        exps = jnp.exp(-self.zetas * rs_2)
        radials = (self.coeffs * exps).sum(axis=-1)
        phis = self.anorms * angulars * radials
        return phis

    def eval_grads(self, rs):
        """

        :param rs:[3]
        :return: [4, nao]
        """
        rs_2 = jnp.sum(rs ** 2, axis=-1)
        angulars = pow_int(rs[None, :], self.ls).prod(axis=-1)
        exps = jnp.exp(-self.zetas * rs_2)
        radials = self.coeffs * exps
        # np
        prim = radials * (angulars * self.anorms)[:, None, None]
        # nao 1 np
        aos = jnp.sum(prim, axis=-1).reshape(-1)

        grads_plus_1 = -2 * self.zetas * (rs ** (jnp.ones(3)))[..., None] * prim
        grads_minus_1 = self.ls[..., None] * (rs ** (-jnp.ones(3)))[..., None] * prim
        grads = grads_plus_1 + grads_minus_1
        grads = jnp.sum(grads, axis=-1)
        aos_grads = jnp.concatenate([aos[..., None], grads], axis=-1)
        return aos_grads.T

    def eval_laps(self, rs):
        """

        :param rs:[3]
        :return: [2, nao]
        """

        rs_2 = jnp.sum(rs ** 2, axis=-1)
        angulars = pow_int(rs[None, :], self.ls).prod(axis=-1)
        exps = jnp.exp(-self.zetas * rs_2)
        radials = self.coeffs * exps
        # np

        prim = radials * (angulars * self.anorms)[:, None, None]
        # nao 1 np
        aos = jnp.sum(prim, axis=-1).reshape(-1)

        laps_plus_2 = 4 * self.zetas**2 * rs[:, None]**2 * prim
        #nao ndim np
        laps_plus_2 = jnp.sum(laps_plus_2, axis=1, keepdims=True)
        #nao 1 np
        laps_0 = -2 * self.zetas * prim * (3 + 2 * self.l)
        #nao 1 np
        ls_minus_2 = self.ls[:, None, :] - jnp.eye(3) * 2
        #nao ndim ndim
        mask = jnp.prod(ls_minus_2 >= 0, axis=-1)
        laps_minus_2 = jnp.where(mask[..., None],
                                 (self.ls**2 - self.ls)[..., None] * prim * (rs ** (-jnp.ones(3)*2))[:, None],
                                 0,)
        #nao ndim np
        laps_minus_2 = jnp.sum(laps_minus_2, axis=1, keepdims=True)
        laps = laps_minus_2 + laps_0 + laps_plus_2
        laps = jnp.sum(laps, axis=-1).reshape(-1)
        aos_laps = jnp.concatenate([aos[..., None], laps[..., None]], axis=-1)
        return aos_laps.T

    def __call__(self, rs):
        return self.eval_ao(rs)


def make_gto_basis(mol, cart=False, method_name='eval_ao'):
    def _gto_basis(x):
        gto_basis = GTOBasis.from_pyscf(mol, cart)
        result = getattr(gto_basis, method_name)(x)
        return result
    gto_basis_forward = hk.without_apply_rng(hk.transform(_gto_basis))
    return lambda x: gto_basis_forward.apply(None, x)


class JAX_SCF(hk.Module):
  def __init__(self,
               mf,
               mol,
               name=None):
    """
    Hartree Fock wave function class for QMC simulation

    :param mf: pyscf.gto.scf, mean field object
    """
    super(JAX_SCF, self).__init__(name=name)
    self.coeff_key = ("mo_coeff_alpha", "mo_coeff_beta")
    self.parameters = {}
    self.mol = mol
    self.atom_coords = mol.atom_coords()
    self.charges = mol.atom_charges()
    self.dim = 3
    self.mf = mf

    self.nelec = mol.nelec
    gto_basis = GTOBasis.from_pyscf(mol=mol, name='GTO_Basis')

    self.gto_basis = jax.jit(jax.vmap(gto_basis, in_axes=0, out_axes=0))
    self.gto_basis_grads = jax.jit(jax.vmap(gto_basis.eval_grads, in_axes=0, out_axes=0))
    self.gto_basis_laps = jax.jit(jax.vmap(gto_basis.eval_laps, in_axes=0, out_axes=0))
    self.init_scf()

  def init_scf(self):
    for s, key in enumerate(self.coeff_key):
      if len(self.mf.mo_occ.shape) == 2:
        self.parameters[key] = self.mf.mo_coeff[s][
                               :, np.asarray(self.mf.mo_occ[s] > 0.9)
                               ]
      else:
        minocc = (0.9, 1.1)[s]
        params = self.mf.mo_coeff[:, np.asarray(self.mf.mo_occ > minocc)]
        self.parameters[key] = jax.tree_map(lambda x: jnp.asarray(x),
                                            params)

  def eval_aos(self, coord):
    """

    :param coord: [ne * 3]
    :return: [ne, nao]
    """
    coord = coord.reshape(-1, 3)
    ao = self.gto_basis(coord)
    return ao

  def eval_aos_grad(self, coord):
    """

    :param coord:
    :return: [ne nao] [ne, 3, nao]
    """
    coord = coord.reshape([-1, 3])
    ao_grad = self.gto_basis_grads(coord)
    # ne 4 nao
    ao, grad = ao_grad[..., 0, :], ao_grad[..., 1:, :]

    return ao, grad

  def eval_aos_lap(self, coord):
    """

    :param coord:
    :return:[ne, nao] [ne, nao]
    """
    coord = coord.reshape([-1, 3])
    ao_lap = self.gto_basis_laps(coord)
    ao, lap = ao_lap[..., 0, :], ao_lap[..., 1, :]

    return ao, lap

  def eval_aos_grad_lap(self, coord):
    """

    :param coord:
    :return:[ne nao] [ne, 3, nao] [ne, nao]
    """
    coord = coord.reshape([-1, 3])
    ao_grad = self.gto_basis_grads(coord)
    lap = self.gto_basis_laps(coord)[..., [1], :]
    ao_grad_lap = jnp.concatenate([ao_grad, lap], axis=-2)
    ao, grad, lap = ao_grad_lap[..., 0, :], ao_grad_lap[..., 1:-1, :], ao_grad_lap[..., -1, :]
    return ao, grad, lap

  def eval_mos(self, aos, s):
    c = self.coeff_key[s]
    p = self.parameters[c]
    mo = aos.dot(p)
    return mo

  def eval_mats(self, coord):
    """

    :param coord:
    :return:list of [[nup, nup], [ndown, ndown]]
    """

    aos = self.eval_aos(coord)
    mos = []
    for s in [0, 1]:
      i0, i1 = s * self.nelec[0], self.nelec[0] + s * self.nelec[1]
      ne = self.nelec[s]
      mo = self.eval_mos(aos[i0:i1, :], s).reshape([ne, ne])
      mos.append(mo)
    return mos

  def eval_single_row(self, coord, e):
    """
    return a sinlge row of hf mats
    :param coord: [ne * 3]
    :param e: int
    :return:
    """
    coord_e = jax.lax.dynamic_slice(coord, (e * 3,), (3,))
    aos = self.eval_aos(coord_e)
    spin_index = e < self.nelec[0]
    mo = jnp.where(spin_index,
                   aos.dot(self.parameters["mo_coeff_alpha"]),
                   aos.dot(self.parameters["mo_coeff_beta"]))
    mo = mo.reshape(-1)
    return mo

  def eval_mats_grads(self, coord):
    """

    :param coord:
    :return: list of [[nup, nup], [ndown, ndown]],
    list of [[nup, 3, nup], [ndown, 3, ndown]]
    """
    aos, grads = self.eval_aos_grad(coord)
    mos = []
    mos_grads = []
    for s in [0, 1]:
      i0, i1 = s * self.nelec[0], self.nelec[0] + s * self.nelec[1]
      ne = self.nelec[s]
      mo = self.eval_mos(aos[i0:i1, :], s).reshape([ne, ne])
      mo_grad = self.eval_mos(grads[i0:i1, :], s).reshape([ne, 3, ne])
      mos.append(mo)
      mos_grads.append(mo_grad)
    return mos, mos_grads

  def eval_mats_laps(self, coord):
    """

    :param coord:
    :return: list of [[nup, nup], [ndown, ndown]]
    """
    aos, laps = self.eval_aos_lap(coord)
    mos = []
    mos_laps = []
    for s in [0, 1]:
      i0, i1 = s * self.nelec[0], self.nelec[0] + s * self.nelec[1]
      ne = self.nelec[s]
      mo = self.eval_mos(aos[i0:i1, :], s).reshape([ne, ne])
      mo_lap = self.eval_mos(laps[i0:i1, :], s).reshape([ne, ne])
      mos.append(mo)
      mos_laps.append(mo_lap)
    return mos, mos_laps

  def eval_mats_grads_laps(self, coord):
    """

    :param coord:
    :return:  list of [[nup, nup], [ndown, ndown]],
    list of [[nup, 3, nup], [ndown, 3, ndown]]
    """
    aos, grads, laps = self.eval_aos_grad_lap(coord)
    mos = []
    mos_grads = []
    mos_laps = []
    for s in [0, 1]:
      i0, i1 = s * self.nelec[0], self.nelec[0] + s * self.nelec[1]
      ne = self.nelec[s]
      mo = self.eval_mos(aos[i0:i1, :], s).reshape([ne, ne])
      mo_grad = self.eval_mos(grads[i0:i1, :], s).reshape([ne, 3, ne])
      mo_lap = self.eval_mos(laps[i0:i1, :], s).reshape([ne, ne])
      mos.append(mo)
      mos_grads.append(mo_grad)
      mos_laps.append(mo_lap)
    return mos, mos_grads, mos_laps

  def eval_grad(self, coord):
    """

    :param coord:
    :return: grads with shape [ne * 3]
    """
    mats, grads = self.eval_mats_grads(coord)
    invs = [jnp.linalg.inv(mat) for mat in mats]
    drift = [jnp.einsum('idj,ji->id', grad, inv) for inv, grad in zip(invs, grads)]
    drift = jnp.concatenate(drift, axis=0)
    drift = drift.reshape(-1)
    return drift

  def eval_grad_laplacian(self, coord):
    """

    :param coord:
    :return: grads with shape [ne * 3]
    local laplacian with shape [ne]
    """
    mats, grads, laps = self.eval_mats_grads_laps(coord)
    invs = [jnp.linalg.inv(mat) for mat in mats]
    grads = [jnp.einsum('idj,ji->id', grad, inv) for inv, grad in zip(invs, grads)]
    grads = jnp.concatenate(grads, axis=0)
    grads = grads.reshape(-1)

    laps = [jnp.einsum('ij,ji->i', lap, inv) for inv, lap in zip(invs, laps)]
    laps = jnp.concatenate(laps, axis=0)
    laps = laps.reshape(-1)
    return grads, laps

  def eval_laplacian(self, coord):
    """

    :param coord:
    :return: laplacian with shape [ne]
    """
    mats, laps = self.eval_mats_laps(coord)
    invs = [jnp.linalg.inv(mat) for mat in mats]
    laplacian = [jnp.einsum('ij,ji->i', lap, inv) for inv, lap in zip(invs, laps)]
    laplacian = jnp.concatenate(laplacian, axis=0)
    return laplacian

  def eval_kinetic(self, coord):
    """

    :param coord:
    :return: local kinetic energy with shape [ne]
    """
    kinetic = self.eval_laplacian(coord)
    kinetic = -0.5 * jnp.sum(kinetic)
    return kinetic

  def eval_phase_and_slogdet(self, x):
    mos = self.eval_mats(x)
    slogdets = [jnp.linalg.slogdet(mo) for mo in mos]
    phase, slogdet = list(map(lambda x, y: [x[0] * y[0], x[1] + y[1]], *zip(slogdets)))[0]
    return phase, slogdet

  def eval_logdet(self, x):
    mos = self.eval_mats(x)
    slogdets = [jnp.linalg.slogdet(mo) for mo in mos]
    phase, slogdet = list(map(lambda x, y: [x[0] * y[0], x[1] + y[1]], *zip(slogdets)))[0]
    return jnp.log(phase) + slogdet

  def eval_slogdet(self, x):
    mos = self.eval_mats(x)
    slogdets = [jnp.linalg.slogdet(mo) for mo in mos]
    _, slogdet = list(map(lambda x, y: [x[0] * y[0], x[1] + y[1]], *zip(slogdets)))[0]
    return slogdet

  def __call__(self, coord):
    return self.eval_slogdet(coord)

def make_jax_scf(mf,
                 mol,
                 method_name='eval_phase_and_slogdet'):
  """
  make molecule hf function
  :param mf:
  :param mol:
  :param method_name:
  :return:
  """

  if method_name not in dir(JAX_SCF):
    raise ValueError('Method name is not in class dir.')

  def _jax_scf(*x):

    jax_scf = JAX_SCF(mf=mf,
                      mol=mol, )
    return getattr(jax_scf, method_name)(*x)

  jax_scf_forward = hk.without_apply_rng(hk.transform(_jax_scf))
  return jax_scf_forward


if __name__ == '__main__':
    from pyscf import gto, scf
    import numpy as np
    from jax.config import config as jax_config
    jax_config.update("jax_enable_x64", True)
    mole = gto.Mole(atom='Li 0 0 0; Li 0 0 3',
                    basis='ccpvdz', spin=0, unit='bohr',
                    cart=False)
    mole.build()
    rhf = scf.RHF(mole)
    rhf.kernel()
    coords = np.random.uniform(0, 1, size=[500, 3])
    pyscf_cart_ao = mole.eval_ao('GTOval_cart', coords)
    pyscf_sph_ao = mole.eval_ao('GTOval_sph', coords)

    jax_gto_cart_basis = make_gto_basis(mole, cart=True, method_name='eval_ao')
    jax_gto_sph_basis = make_gto_basis(mole, cart=False, method_name='eval_ao')
    jax_gto_cart_basis = jax.vmap(jax_gto_cart_basis)
    jax_gto_sph_basis = jax.vmap(jax_gto_sph_basis)
    jax_cart_ao = jax_gto_cart_basis(coords)
    jax_sph_ao = jax_gto_sph_basis(coords)

    assert jnp.allclose(jax_sph_ao, pyscf_sph_ao)
    assert jnp.allclose(jax_cart_ao, pyscf_cart_ao)

    hf_ansatz = make_jax_scf(rhf, mole)
    hf_ansatz_mat = make_jax_scf(rhf, mole, method_name='eval_mats')
    test_coord = np.random.normal(size=[mole.nelectron*3])
    test_coord = jnp.asarray(test_coord)

    key = jax.random.PRNGKey(666)
    params = hf_ansatz.init(key, test_coord)
    jax_res = hf_ansatz.apply(params, test_coord)


