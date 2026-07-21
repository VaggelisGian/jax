# Copyright 2025 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test TPU-specific uses of Pallas memory space APIs."""

import functools
import json
import re
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax._src import core as jax_core
from jax._src import test_util as jtu
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np


jax.config.parse_flags_with_absl()
P = jax.sharding.PartitionSpec
partial = functools.partial


def _json_object_with_fields(fields):
  return (
      r'\{'
      + ''.join(
          '(?=[^{}]*' + _json_object_with_a_field(name, value) + ')'
          for name, value in fields.items()
      )
      + r'[^{}]*\}'
  )


def _json_object_with_a_field(field_name, value):
  return (
      re.escape(json.dumps(field_name, separators=(',', ':')))
      + r'\s*:\s*'
      + _json_value_pattern(value)
  )


def _json_value_pattern(value):
  if isinstance(value, dict):
    return _json_object_with_fields(value)
  if isinstance(value, list):
    return (
        r'\['
        + ','.join(_json_value_pattern(entry) for entry in value)
        + r'\]'
    )
  return re.escape(json.dumps(value, separators=(',', ':')))


class TPUPallasCallMemorySpaceTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if not jtu.is_device_tpu_at_least(5):
      self.skipTest('Needs a newer TPU')

  @parameterized.parameters(
      (pltpu.VMEM, 1),
      (pltpu.SMEM, 4),
      (pltpu.HBM, 0),
      (pl.ANY, None),
  )
  def test_basic_input_memory_space_constraint(self, memory_space, color):
    def kernel(x_ref, y_ref):
      pltpu.sync_copy(x_ref, y_ref)

    def g(x):
      return pl.pallas_call(
          kernel,
          out_shape=x,
          in_specs=[pl.BlockSpec(memory_space=memory_space)],
          out_specs=pl.BlockSpec(memory_space=pl.ANY),
      )(x)

    @jax.jit
    def f(x):
      x = pltpu.with_memory_space_constraint(x, memory_space=memory_space)
      if color is not None:
        self.assertEqual(jax.typeof(x).memory_space, memory_space)
      x = g(x)
      return x

    x = jnp.ones((8, 128), dtype=jnp.float32)
    y = f(x)
    np.testing.assert_array_equal(y, x)
    lowered = jax.jit(f).lower(x)
    lowered.compile()
    hlo = lowered.compiler_ir(dialect='hlo').as_hlo_text()
    if color is None or memory_space == pltpu.SMEM:
      self.assertNotIn('input_memory_space_colors', hlo)
    else:
      self.assertRegex(
          hlo,
          _json_object_with_a_field(
              'input_memory_space_colors',
              [{
                  'color': color,
                  'operand_index': 0,
              }],
          ),
      )

  @parameterized.parameters(
      (pltpu.VMEM, 1),
      (pltpu.SMEM, 4),
      (pltpu.HBM, 0),
      (pl.ANY, None),
      (pl.HOST, 5),
  )
  def test_basic_output_memory_space_constraint(self, memory_space, color):
    if color is None:
      out_shape_ctor = jax.ShapeDtypeStruct
    else:
      out_shape_ctor = lambda shape, dtype: jax_core.ShapedArray(
          shape, dtype, memory_space=memory_space
      )

    def kernel(x_ref, y_ref):
      pltpu.sync_copy(x_ref, y_ref)

    def g(x):
      return pl.pallas_call(
          kernel,
          out_shape=out_shape_ctor(x.shape, x.dtype),
          in_specs=[pl.BlockSpec(memory_space=pl.ANY)],
          out_specs=pl.BlockSpec(memory_space=memory_space),
      )(x)

    if memory_space == pl.HOST:
      if jax.device_count() > 1:
        self.skipTest('Test only works with a single device.')
      out_sharding = jax.sharding.NamedSharding(
          jax.sharding.Mesh(jax.devices(), 'x'),
          jax.sharding.PartitionSpec(),
          memory_kind='pinned_host',
      )
    else:
      out_sharding = None

    @jax.jit(out_shardings=out_sharding)
    def f(x):
      x = g(x)
      return x

    x = jnp.ones((8, 128), dtype=jnp.float32)
    y = f(x)
    np.testing.assert_array_equal(y, x)
    lowered = jax.jit(f, out_shardings=out_sharding).lower(x)
    lowered.compile()
    hlo = lowered.compiler_ir(dialect='hlo').as_hlo_text()
    if color is None:
      self.assertNotIn('output_memory_space_colors', hlo)
    else:
      self.assertRegex(
          hlo,
          _json_object_with_a_field(
              'output_memory_space_colors',
              [{'color': color}],
          ),
      )

  def test_tuple_output_memory_space_constraint(self):
    memory_space = pltpu.VMEM
    color = 1

    def kernel(x_ref, y_ref, z_ref):
      pltpu.sync_copy(x_ref, y_ref)
      pltpu.sync_copy(x_ref, z_ref)

    def g(x):
      return pl.pallas_call(
          kernel,
          out_shape=(
              pltpu.VMEM(x.shape, x.dtype),
              pltpu.VMEM(x.shape, x.dtype),
          ),
          in_specs=[pl.BlockSpec(memory_space=pl.ANY)],
          out_specs=(
              pl.BlockSpec(memory_space=memory_space),
              pl.BlockSpec(memory_space=memory_space),
          ),
      )(x)

    @jax.jit
    def f(x):
      y, z = g(x)
      return y, z

    x = jnp.ones((8, 128), dtype=jnp.float32)
    y, z = f(x)
    np.testing.assert_array_equal(y, x)
    np.testing.assert_array_equal(z, x)
    lowered = jax.jit(f).lower(x)
    lowered.compile()
    hlo = lowered.compiler_ir(dialect='hlo').as_hlo_text()
    self.assertRegex(
        hlo,
        _json_object_with_a_field(
            'output_memory_space_colors',
            [
                {'color': color, 'shape_index': [0]},
                {'color': color, 'shape_index': [1]},
            ],
        ),
    )


# TODO(slebedev): Inline once ``pl.core_map`` is gone.
def _make_run_kernel(**kwargs):
  return lambda fn: pl.kernel(**kwargs)(fn)()


class TPUCoreMapMemorySpaceTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if not jtu.is_device_tpu_at_least(5):
      self.skipTest('Needs a newer TPU')

  @parameterized.parameters(
      (pltpu.VMEM, 1),
      (pltpu.SMEM, 4),
      (pltpu.HBM, 0),
      (pl.ANY, None),
  )
  def test_basic_ref_memory_space_constraint(
      self, memory_space, color
  ):
    @jax.jit
    def f(x):
      x_ref = jax.new_ref(x, memory_space=memory_space)
      y_ref = jax.new_ref(pl.empty_like(x), memory_space=memory_space)

      self.assertEqual(jax.typeof(x_ref).memory_space, memory_space)
      self.assertEqual(jax.typeof(y_ref).memory_space, memory_space)

      run_kernel = _make_run_kernel(
          mesh=pltpu.TensorCoreMesh(axis_name='core'),
      )

      @run_kernel
      def _():
        if jax.typeof(x_ref).memory_space is pltpu.VMEM:
          y_ref[...] = x_ref[...]
        else:
          pltpu.sync_copy(x_ref, y_ref)

      return y_ref[...]

    x = jnp.arange(1024, dtype=jnp.float32).reshape((8, 128))
    num_cores = jax.devices()[0].num_cores
    # TODO(slebedev): Make sure mpmd_map also fails here.
    if num_cores > 1 and memory_space is pltpu.VMEM:
      with self.assertRaisesRegex(
          NotImplementedError,
          'TensorCoreMesh does not support VMEM inputs/outputs when there are'
          ' >1 cores. Use HBM or ANY instead.',
      ):
        f.lower(x).compile()
      return
    lowered = f.lower(x)
    compiled = lowered.compile()
    hlo = lowered.compiler_ir(dialect='hlo').as_hlo_text()
    if color is None or memory_space == pltpu.SMEM:
      self.assertNotIn('input_memory_space_colors', hlo)
    else:
      self.assertRegex(
          hlo,
          _json_object_with_a_field(
              'input_memory_space_colors',
              [
                  {
                      'color': color,
                      'operand_index': 0,
                  },
                  {
                      'color': color,
                      'operand_index': 1,
                  },
              ],
          ),
      )
    y = compiled(x)
    np.testing.assert_array_equal(y, x)

  def test_smem_copy(self):
    mesh = pltpu.TensorCoreMesh(axis_name='core')
    if mesh.num_cores > 1:
      self.skipTest('Only one core is supported for this test.')

    run_kernel = _make_run_kernel(mesh=mesh)

    @jax.jit
    def f():
      y_ref = pl.empty_ref_like(pltpu.SMEM((8,), jnp.int32))

      @run_kernel
      def _():
        for i in range(y_ref.shape[0]):
          y_ref[i] = i

      @run_kernel
      def _():
        for i in range(y_ref.shape[0]):
          y_ref[i] = y_ref[i] + 1

      return y_ref[...]

    np.testing.assert_array_equal(f(), np.arange(8) + 1)

  def test_smem_async_copy(self):
    mesh = pltpu.TensorCoreMesh(axis_name='core')
    if mesh.num_cores > 1:
      self.skipTest('Only one core is supported for this test.')

    run_kernel = _make_run_kernel(mesh=mesh)

    @jax.jit
    def f():
      y_ref = pl.empty_ref_like(pltpu.SMEM((8,), jnp.int32))

      @run_kernel
      def _():
        for i in range(y_ref.shape[0]):
          y_ref[i] = i

      @run_kernel
      def _():
        for i in range(y_ref.shape[0]):
          y_ref[i] = y_ref[i] + 1

      y_out_ref = pl.empty_ref_like(pltpu.HBM((8,), jnp.int32))

      sem = pl.empty_ref_like(pltpu.SemaphoreType.DMA(()))

      @run_kernel
      def _():
        pltpu.make_async_copy(y_ref, y_out_ref, sem).start()

      @run_kernel
      def _():
        pltpu.make_async_copy(y_ref, y_out_ref, sem).wait()

      return y_out_ref[...]

    np.testing.assert_array_equal(f(), np.arange(8) + 1)

  def test_smem_async_copy_megacore(self):
    mesh = pltpu.TensorCoreMesh(axis_name='core')
    num_cores = mesh.num_cores
    if num_cores == 1:
      self.skipTest('Only megacore is supported for this test.')

    run_kernel = _make_run_kernel(mesh=mesh)
    n = 256

    @jax.jit
    def f():
      y_ref = pl.empty_ref_like(pltpu.SMEM((1, n), jnp.int32))

      @run_kernel
      def _():
        core_i = jax.lax.axis_index('core')
        for i in range(n):
          y_ref[0, i] = i + core_i * n

      @run_kernel
      def _():
        for i in range(n):
          y_ref[0, i] = y_ref[0, i] + 1

      y_out_ref = pl.empty_ref_like(pltpu.HBM((num_cores, 1, n), jnp.int32))

      sem = pl.empty_ref_like(pltpu.SemaphoreType.DMA(()))

      @run_kernel
      def _():
        core_i = jax.lax.axis_index('core')
        pltpu.make_async_copy(y_ref, y_out_ref.at[core_i, ...], sem).start()

      @run_kernel
      def _():
        core_i = jax.lax.axis_index('core')
        pltpu.make_async_copy(y_ref, y_out_ref.at[core_i, ...], sem).wait()

      return y_out_ref[...]

    np.testing.assert_array_equal(
        f(), np.arange(num_cores * n).reshape((num_cores, 1, n)) + 1
    )

  def test_annotate_slicing_and_indexing_in_kernel(self):

    def kernel(x_ref, y_ref, scratch_ref):
      ann_x = pltpu.annotate(
          x_ref, pltpu.LLOAttr(no_store=True, no_bank_conflict=True)
      )
      ann_scratch = pltpu.annotate(
          scratch_ref,
          pltpu.LLOAttr(no_bank_conflict=True, no_hazard_no_deps=True),
      )
      ann_y = pltpu.annotate(y_ref, pltpu.LLOAttr(no_bank_conflict=True))
      # Slice and index reads/writes on annotated references inside kernel body
      ann_scratch[0, :] = ann_x[0, :]
      ann_scratch[1:, :] = ann_x[1:, :]
      ann_y[0, :] = ann_scratch[0, :]
      ann_y[1:, :] = ann_scratch[1:, :]

    x = jnp.arange(128, dtype=jnp.float32).reshape((8, 16))

    # 1. Verify that interpret mode tracing succeeds
    y_interpret = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)],
        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
        scratch_shapes=[pltpu.VMEM((8, 16), jnp.float32)],
        interpret=True,
    )(x)
    np.testing.assert_allclose(y_interpret, x)

    # 2. Run on TPU
    if jtu.is_device_tpu_at_least(5):

      @jax.jit
      def f(x_arg):
        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(x_arg.shape, x_arg.dtype),
            in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)],
            out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
            scratch_shapes=[pltpu.VMEM((8, 16), jnp.float32)],
        )(x_arg)

      y_lowered = f(x)
      np.testing.assert_allclose(y_lowered, x)
      lowered = f.lower(x)
      hlo_text = lowered.compiler_ir(dialect='hlo').as_hlo_text()
      self.assertIn('tpu_custom_call', hlo_text)

  def test_annotate_reference_in_run_scoped(self):

    def kernel(x_ref, y_ref):

      def body(tmp_ref):
        ann_tmp = pltpu.annotate(
            tmp_ref,
            pltpu.LLOAttr(no_bank_conflict=True, no_hazard_no_deps=True),
        )
        ann_x = pltpu.annotate(x_ref, pltpu.LLOAttr(no_store=True))
        ann_y = pltpu.annotate(y_ref, pltpu.LLOAttr(no_bank_conflict=True))
        ann_tmp[...] = ann_x[...]
        ann_y[...] = ann_tmp[...]

      pl.run_scoped(body, pltpu.VMEM((8, 16), jnp.float32))

    x = jnp.arange(128, dtype=jnp.float32).reshape((8, 16))

    # 1. Verify interpret mode execution
    y_interpret = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)],
        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
        interpret=True,
    )(x)
    np.testing.assert_allclose(y_interpret, x)

    # 2. Verify compilation
    if jtu.is_device_tpu_at_least(5):

      @jax.jit
      def f(x_arg):
        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(x_arg.shape, x_arg.dtype),
            in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)],
            out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
        )(x_arg)

      y_lowered = f(x)
      np.testing.assert_allclose(y_lowered, x)
      lowered = f.lower(x)
      hlo_text = lowered.compiler_ir(dialect='hlo').as_hlo_text()
      self.assertIn('tpu_custom_call', hlo_text)

  def test_annotate_lowering_multiple_or_sequential_attributes(self):

    def kernel(x_ref, y_ref):
      ann1 = pltpu.annotate(x_ref, pltpu.LLOAttr(no_store=True))
      ann2 = pltpu.annotate(ann1, pltpu.LLOAttr(no_bank_conflict=True))
      y_ref[...] = ann2[...]

    x = jnp.arange(128, dtype=jnp.float32).reshape((8, 16))
    y_interpret = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)],
        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
        interpret=True,
    )(x)
    np.testing.assert_allclose(y_interpret, x)

    if jtu.is_device_tpu_at_least(5):

      @jax.jit
      def f(x_arg):
        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(x_arg.shape, x_arg.dtype),
            in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)],
            out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
        )(x_arg)

      y_lowered = f(x)
      np.testing.assert_allclose(y_lowered, x)
      lowered = f.lower(x)
      hlo_text = lowered.compiler_ir(dialect='hlo').as_hlo_text()
      self.assertIn('tpu_custom_call', hlo_text)


class LLOAnnotationInvariantsTest(jtu.JaxTestCase):

  def test_annotate_on_subview_slice(self):

    def kernel(x_ref, y_ref):
      ann_slice = pltpu.annotate(x_ref.at[0], pltpu.LLOAttr(no_store=True))
      y_ref[0] = ann_slice[...]
      y_ref[1:] = x_ref.at[1:][...]

    x = jnp.arange(128, dtype=jnp.float32).reshape((8, 16))
    y_interpret = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)],
        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
        interpret=True,
    )(x)
    np.testing.assert_allclose(y_interpret, x)

    if jtu.is_device_tpu_at_least(5):

      @jax.jit
      def f(x_arg):
        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(x_arg.shape, x_arg.dtype),
            in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)],
            out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
        )(x_arg)

      y_lowered = f(x)
      np.testing.assert_allclose(y_lowered, x)
      lowered = f.lower(x)
      hlo_text = lowered.compiler_ir(dialect='hlo').as_hlo_text()
      self.assertIn('tpu_custom_call', hlo_text)

  def test_annotate_inside_region_scope(self):
    """Verifies that annotating a MemRef inside a region scope works."""

    def kernel(x_ref, y_ref):

      def loop_body(i, _):
        ann_x = pltpu.annotate(x_ref, pltpu.LLOAttr(no_store=True))
        y_ref[i] = ann_x[i]

      jax.lax.fori_loop(0, 8, loop_body, None, unroll=True)

    x = jnp.arange(128, dtype=jnp.float32).reshape((8, 16))
    y_interpret = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)],
        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
        interpret=True,
    )(x)
    np.testing.assert_allclose(y_interpret, x)

    if jtu.is_device_tpu_at_least(5):

      @jax.jit
      def f(x_arg):
        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(x_arg.shape, x_arg.dtype),
            in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)],
            out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
        )(x_arg)

      y_lowered = f(x)
      np.testing.assert_allclose(y_lowered, x)
      lowered = f.lower(x)
      hlo_text = lowered.compiler_ir(dialect='hlo').as_hlo_text()
      self.assertIn('tpu_custom_call', hlo_text)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
