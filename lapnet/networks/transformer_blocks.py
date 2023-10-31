# Copyright 2022 The Flax Authors.
# Copyright 2023 Bytedance Ltd. and/or its affiliate
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Sequence, Any, Tuple, Union, Optional, Callable
import attr

from jax import lax
import jax
from flax import linen as nn
from flax.linen.module import Module, compact
from flax.linen.initializers import lecun_normal, zeros

from lapjax import LapTuple
import lapjax.numpy as jnp
from lapjax.nn import softmax
from lapjax.lapsrc.laputils import lap_counter
from lapjax import InputInfo, SparsInfo
import lapnet.networks.network_blocks as network_blocks

PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?
Array = Any
PrecisionLike = Union[None, str, lax.Precision, Tuple[str, str],
                      Tuple[lax.Precision, lax.Precision]]

default_kernel_init = lecun_normal()

# For those functions defined in the flax, we have to replace the 
# original jax.numpy with lapjax.numpy to use forward laplacian.
# We copy them to this file and thus the jnp now refers to lapjax.numpy. 
# The corresponding LICENSE has been added in the begin of this file. 

def scaled_dot_product(q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray
      ) -> Sequence[jnp.ndarray]:
  """Calculate the head values and attention given transfomer q,k,v.

  This implementation is from `https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial6/Transformers_and_MHAttention.html`.

  Args:
      q (jnp.ndarray): the q tensors with shape [Head, N_elec, Dims]
      k (jnp.ndarray): the k tensors with shape [Head, N_elec, Dims]
      v (jnp.ndarray): the v tensors with shape [Head, N_elec, Dims]

  """
  d_k = q.shape[-1]
  attn_logits = jnp.matmul(q, jnp.swapaxes(k, -2, -1))
  attn_logits = attn_logits / jnp.sqrt(d_k)
  attention = softmax(attn_logits, axis=-1)
  values = jnp.matmul(attention, v)
  return values, attention

def attention_sparse_dot_product(q,k,v):
  '''
  A sparse version of scaled dot product attention, used in LapNet.
  '''
  num = lap_counter([q,k,v])
  if num == 0:
    return scaled_dot_product(q,k,v)
  else:
    # LapJAX does not support tensors with sparsity accross different
    # axises, e.g. the tensor like A_{ij}\delta_{jk} + B_{ik}\delta_{ij}. 
    # Here we have to manully define the LapTuple propogation rule.
    q:LapTuple = q
    k:LapTuple = k
    v:LapTuple = v.set_dense(force=True)
    num_head, nelec, d_k = q.shape
    k = jnp.swapaxes(k, -2, -1)
    logits_value = q.value @ k.value  # shape = H, N, N
    logits_grad1 = q.grad @ k.value  # shape = 3, H, N, N
    logits_grad2 = q.value @ k.grad  # shape = 3, H, N, N

    # push all diag grad into grad1 to avoid laptuple bug
    idx = jax.numpy.arange(nelec)
    diag_grad = logits_grad2[...,idx,idx]

    logits_grad1 = logits_grad1.at[...,idx,idx].add(diag_grad)
    logits_grad2 = logits_grad2.at[...,idx,idx].add(-diag_grad)
    logits_grad = jax.numpy.concatenate((logits_grad1,logits_grad2), axis=0)

    I_N = jax.numpy.eye(nelec)

    logits_lap = q.lap @ k.value + q.value @ k.lap  # shape = H, N, N
    cross_lap = 2 * jnp.sum(q.grad * jnp.swapaxes(k.grad, -2, -1), axis=(0,3))  # only i==j has none-zero lap term
    logits_lap = logits_lap.at[:, idx, idx].add(cross_lap)

    # create logits tuple to utilize lapjax
    spars = SparsInfo(inputinfo=InputInfo(logits_grad.shape[0]), is_input=True)
    spars.set_dense(logits_grad[None],force=True)
    logits = LapTuple(logits_value, logits_grad, logits_lap, spars)

    logits = logits / jnp.sqrt(d_k)
    max_value = jax.numpy.max(logits.value, axis=-1, keepdims=True)
    logits = logits - max_value
    exp_logits:LapTuple = jnp.exp(logits)

    # # \sum_j exp_{ij}

    # grad1 refer to i, grad2 refer to j
    exp_logits_grad1, exp_logits_grad2 = jax.numpy.split(exp_logits.grad, 2, axis=0)  # shape = (3, H, N_i, N_j), N_i=N_j

    # grad1 can be sumover through standard summation of lapjax
    spars = SparsInfo(inputinfo=InputInfo(exp_logits_grad1.shape[0]), is_input=True)
    spars.set_dense(exp_logits_grad1[None],force=True)
    exp_logits_for_sum = LapTuple(exp_logits.value, exp_logits_grad1, exp_logits.lap, spars)
    sum_exp = jnp.sum(exp_logits_for_sum, axis=-1)

    # map to dense grad
    sum_exp_grad = sum_exp.grad[None, ...] * I_N[:, None, None, :] + exp_logits_grad2.transpose(3,0,1,2)
    sum_exp.grad = sum_exp_grad.reshape((-1,)+sum_exp.value.shape)
    sum_exp.spars = v.spars

    # \sum_j exp_{ij}v_j
    sum_exp_v_value = exp_logits.value @ v.value

    v_grad_new = v.grad.reshape((nelec,-1) + v.value.shape)  # shape = (N,3,H,N,D)
    sum_exp_v_grad = exp_logits.value @ v_grad_new  # shape = (N, 3, H, N, D)
    sum_exp_v_grad1 = exp_logits_grad1 @ v.value  # shape = (3, H, N, D), refer to i
    sum_exp_v_grad = sum_exp_v_grad + sum_exp_v_grad1[None,...] * I_N[:,None,None,:,None]  # shape = (N, 3, H, N, D)
    sum_exp_v_grad2 = exp_logits_grad2[...,None] * v.value[None,:,None,:,:]  # shape = 3, H, N, N, D
    sum_exp_v_grad2 = sum_exp_v_grad2.transpose(3,0,1,2,4)  # shape = N, 3, H, N, D
    sum_exp_v_grad = (sum_exp_v_grad + sum_exp_v_grad2).reshape((-1,) + sum_exp_v_value.shape)

    sum_exp_v_lap = exp_logits.lap @ v.value + exp_logits.value @ v.lap
    lap1 = jnp.sum(exp_logits_grad1[...,None] * v_grad_new.transpose(1,2,0,3,4), axis=(0,3))
    lap2 = jnp.sum(exp_logits_grad2[...,None] * v_grad_new[idx,...,idx,:].transpose(1,2,0,3)[:,:,None,...], axis=(0,3))
    sum_exp_v_lap = sum_exp_v_lap + 2*lap1 + 2*lap2

    sum_exp_v = LapTuple(sum_exp_v_value, sum_exp_v_grad, sum_exp_v_lap, v.spars)
    values = sum_exp_v / sum_exp[...,None]

    # we do not use attention map after QKV product and the LapTuple for 
    # attention map have not been computed in current function.
    # So we return None as a placeholder. May be refined in the future
    return values, None

class Dense(Module):
  """A linear transformation applied over the last dimension of the input.

  Attributes:
    features: the number of output features.
    use_bias: whether to add a bias to the output (default: True).
    dtype: the dtype of the computation (default: infer from input and params).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.
    kernel_init: initializer function for the weight matrix.
    bias_init: initializer function for the bias.
  """
  features: int
  use_bias: bool = True
  dtype: Optional[Dtype] = None
  param_dtype: Dtype = jnp.float32
  precision: PrecisionLike = None
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = zeros

  @compact
  def __call__(self, inputs: Array) -> Array:
    """Applies a linear transformation to the inputs along the last dimension.

    Args:
      inputs: The nd-array to be transformed.

    Returns:
      The transformed input.
    """
    kernel = self.param('kernel',
                        self.kernel_init,
                        (jnp.shape(inputs)[-1], self.features),)
                        #self.param_dtype)
    if self.use_bias:
      bias = self.param('bias', self.bias_init, (self.features,),)
                        #self.param_dtype)
    else:
      bias = None
    y = network_blocks.linear_layer(inputs, kernel, bias)  # use linear_layer to avoid orphan in kfac
    return y


class MultiheadAttention(nn.Module):
  """Multi-head Attention Layer. 

  The function returns `W_o * concat_h(SelfAttention(h,Wq,Wk,Wv))`.

  This implementation is from `https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial6/Transformers_and_MHAttention.html`.

  Example:
    rng = jax.random.PRNGKey(42)
    x = jax.random.normal(rng, (16, 128))
    mh_attn = MultiheadAttention(output_dim=128, num_heads=4)
    
    params = mh_attn.init(rng, x)['params']
    out, attn = mh_attn.apply({'params': params}, x)

    print('Out', out.shape, 'Attention', attn.shape)

    ```
      Out (16, 128) Attention (4, 16, 16)
    ```

  Args:
      output_dim (int): Output dimension. Specifically, it is the output dimension after concatenating all heads. Each head has attention dimension output_dim // h
      num_heads (int):  Number of parallel heads (h).

  Params:
    self.init()['params'] belongs to `flax.core.frozen_dict.FrozenDict`.
    It has key `qkj_proj` and `o_proj`, each with matrixs `kernel` and `bias`.
    This initialized parameters should be passed to apply().

  Returns:
      When the instance of `MultiheadAttention` is called, 
      the projected value and the attention tensors are returned.
  """

  # This is the hyper-parameters to be specified
  # then the class is firstly instantiated.
  output_dim : int  # Output dimension
  num_heads : int  # Number of parallel heads (h)

  def setup(self):
    # Stack all weight matrices 1...h and W^Q, W^K, W^V together for efficiency
    # Note that in many implementations you see "bias=False" which is optional
    self.qkv_proj = Dense(3*self.output_dim,
                              kernel_init=nn.initializers.xavier_uniform(),  # Weights with Xavier uniform init
                              bias_init=nn.initializers.zeros  # Bias init with zeros
                            )
    self.o_proj = Dense(self.output_dim,
                            kernel_init=nn.initializers.xavier_uniform(),
                            bias_init=nn.initializers.zeros)

  def __call__(self, x: jnp.ndarray) -> Sequence[jnp.ndarray]:
    n_elec = x.shape[0]
    qkv = self.qkv_proj(x)

    # Separate Q, K, V from linear output
    qkv = qkv.reshape(n_elec, self.num_heads, -1)
    qkv = qkv.transpose(1, 0, 2)  # [Head, N_elec, Dims]
    q, k, v = jnp.array_split(qkv, 3, axis=-1)

    # Determine value outputs
    values, attention = scaled_dot_product(q, k, v)
    values = values.transpose(1, 0, 2)  # [N_elec, Head, Dims]
    values = values.reshape(n_elec, self.output_dim)
    o = self.o_proj(values)

    return o, attention


class MultiheadCrossAttention(nn.Module):
  """
  This module is adopted from MultiheadAttention class. It is used for cross attention in LapNet.
  The dot product function in this module leverages the sparsity in the LapNet.
  WARN: if you want to use this crossattention module to other architecture with the sparsity is different from LapNet,
  you need to replace the `attention_sparse_dot_product` function with `attention_dot_product`.
  """

  # This is the hyper-parameters to be specified
  # then the class is firstly instantiated.
  output_dim : int  # Output dimension
  num_heads : int  # Number of parallel heads (h)

  def setup(self):
    # Stack all weight matrices 1...h and W^Q, W^K, W^V together for efficiency
    # Note that in many implementations you see "bias=False" which is optional
    self.qk_proj = Dense(2 * self.output_dim,
                          kernel_init=nn.initializers.xavier_uniform(),  # Weights with Xavier uniform init
                          bias_init=nn.initializers.zeros  # Bias init with zeros
                          )
    self.v_proj = Dense(self.output_dim,
                          kernel_init=nn.initializers.xavier_uniform(),  # Weights with Xavier uniform init
                          bias_init=nn.initializers.zeros  # Bias init with zeros
                          )
    self.o_proj = Dense(self.output_dim,
                            kernel_init=nn.initializers.xavier_uniform(),
                            bias_init=nn.initializers.zeros)

  def __call__(self, h: Tuple[jnp.ndarray]) -> Sequence[jnp.ndarray]:
    hs, hd = h
    n_elec = hs.shape[0]
    qk = self.qk_proj(hs)

    # Separate Q, K from linear output
    q, k = jnp.array_split(qk, 2, axis=-1)
    v = self.v_proj(hd)

    trans = lambda x: x.reshape(n_elec, self.num_heads, -1).transpose(1, 0, 2)
    q, k, v = trans(q), trans(k), trans(v)

    # Determine value outputs

    values, _ = attention_sparse_dot_product(q, k, v)

    values = values.transpose(1, 0, 2)  # [N_elec, Head, Dims]
    values = values.reshape(n_elec, self.output_dim)
    values = self.o_proj(values)

    return values, None

class LayerNorm(nn.Module):
  """Layer normalization (https://arxiv.org/abs/1607.06450).

  LayerNorm normalizes the activations of the layer for each given example in a
  batch independently, rather than across a batch like Batch Normalization.
  i.e. applies a transformation that maintains the mean activation within
  each example close to 0 and the activation standard deviation close to 1.

  Attributes:
    epsilon: A small float added to variance to avoid dividing by zero.
    dtype: the dtype of the result (default: infer from input and params).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    use_bias:  If True, bias (beta) is added.
    use_scale: If True, multiply by scale (gamma). When the next layer is linear
      (also e.g. nn.relu), this can be disabled since the scaling will be done
      by the next layer.
    bias_init: Initializer for bias, by default, zero.
    scale_init: Initializer for scale, by default, one.
    reduction_axes: Axes for computing normalization statistics.
    feature_axes: Feature axes for learned bias and scaling.
    axis_name: the axis name used to combine batch statistics from multiple
      devices. See `jax.pmap` for a description of axis names (default: None).
      This is only needed if the model is subdivided across devices, i.e. the
      array being normalized is sharded across devices within a pmap.
    axis_index_groups: groups of axis indices within that named axis
      representing subsets of devices to reduce over (default: None). For
      example, `[[0, 1], [2, 3]]` would independently batch-normalize over
      the examples on the first two and last two devices. See `jax.lax.psum`
      for more details.
  """
  epsilon: float = 1e-6
  dtype: Optional[Dtype] = None
  use_bias: bool = True
  use_scale: bool = True
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.zeros
  scale_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.ones
  reduction_axes: Union[int, Any] = -1
  feature_axes: Union[int, Any] = -1
  axis_name: Optional[str] = None
  axis_index_groups: Any = None

  @compact
  def __call__(self, x):
    """Applies layer normalization on the input.

    Args:
      x: the inputs

    Returns:
      Normalized inputs (the same shape as inputs).
    """

    mean = jnp.mean(x, axis=self.reduction_axes, keepdims=True)
    mean2 = jnp.mean(x**2, axis=self.reduction_axes, keepdims=True)
    # mean2 - _abs_sq(mean) is not guaranteed to be non-negative due
    # to floating point round-off errors.
    var = mean2 - mean**2
    y = x - mean
    y = y / jnp.sqrt(var + self.epsilon)
    scale = self.param('scale',
                        self.scale_init,
                        (jnp.shape(x)[-1],))
    bias = self.param('bias', self.bias_init, (jnp.shape(x)[-1],),)
    return y*scale + bias

class LayerNormBlock(nn.Module):
  """LayerNorm Block, with nn.Module as base class.
  This ensures the jax compling suits with flax. 

  """
  use_layernorm: bool

  def setup(self):
    self.norm = LayerNorm()

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    return self.norm(x) if self.use_layernorm else x

@attr.s(auto_attribs=True)
class CrossAttentionLayer:
  attention: MultiheadCrossAttention
  layernorm1: LayerNormBlock
  layernorm2: LayerNormBlock
  layernorm3: LayerNormBlock

@attr.s(auto_attribs=True)
class TransformerLayer:
  attention: MultiheadAttention
  layernorm1: LayerNormBlock
  layernorm2: LayerNormBlock
