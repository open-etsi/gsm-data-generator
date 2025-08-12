# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Test last-stage of codegen VM.

Restrictions: all shape lowered, explicit allocation.
"""
import numpy as np
import pytest
import gsmDataGen
import gsmDataGen.testing
from gsmDataGen import relax
from gsmDataGen.relax.testing.runtime_builtin import MakeShapeCode, MatchShapeCode
from gsmDataGen.relax.testing.vm import check_saved_func
from gsmDataGen.script import ir as I
from gsmDataGen.script import relax as R
from gsmDataGen.script import tir as T

EXEC_MODE = ["bytecode", "compiled"]


def codegen(mod, target, exec_mode="bytecode"):
    builder = relax.ExecBuilder()
    tir_mod = relax.vm_build._vmcodegen(builder, mod, exec_mode=exec_mode)
    return relax.vm_build._vmlink(builder, target, tir_mod)


@pytest.mark.parametrize("exec_mode", EXEC_MODE)
def test_vm_copy(exec_mode):
    @gsmDataGen.script.ir_module
    class TestVMMove:
        @R.function(pure=False)
        def foo(x: R.Tensor((3, 4), "float32")):
            R.func_attr({"global_symbol": "foo"})
            z = R.call_packed("vm.builtin.copy", x, sinfo_args=(R.Tensor((3, 4), dtype="float32")))
            return z

    mod = TestVMMove
    target = gsmDataGen.target.Target("llvm", host="llvm")
    ex = codegen(mod, target, exec_mode)
    inp = gsmDataGen.nd.array(np.random.rand(3, 4).astype(np.float32))
    vm = relax.VirtualMachine(ex, gsmDataGen.cpu())
    res = check_saved_func(vm, "foo", inp)
    gsmDataGen.testing.assert_allclose(res.numpy(), inp.numpy(), rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize("exec_mode", EXEC_MODE)
def test_vm_to_device(exec_mode):
    @gsmDataGen.script.ir_module
    class TestVMToDevice:
        @R.function(pure=False)
        def foo(x: R.Tensor((3, 4), "float32")):
            R.func_attr({"global_symbol": "foo"})
            # Copy x to the first cpu: device_type=1 and device_id=0.
            z = R.call_packed(
                "vm.builtin.to_device", x, 1, 0, sinfo_args=(R.Tensor((3, 4), dtype="float32"))
            )
            return z

    mod = TestVMToDevice
    target = gsmDataGen.target.Target("llvm", host="llvm")
    ex = codegen(mod, target, exec_mode)
    inp = gsmDataGen.nd.array(np.random.rand(3, 4).astype(np.float32))
    vm = relax.VirtualMachine(ex, gsmDataGen.cpu())
    res = check_saved_func(vm, "foo", inp)
    gsmDataGen.testing.assert_allclose(res.numpy(), inp.numpy(), rtol=1e-7, atol=1e-7)
    # check the resulting tensor is on cpu:0
    assert res.device == gsmDataGen.cpu(0)
    assert res.device.device_type == 1
    assert res.device.device_id == 0


@pytest.mark.parametrize("exec_mode", EXEC_MODE)
def test_if_cond_const(exec_mode):
    @gsmDataGen.script.ir_module
    class TestVMIfCondConst:
        @R.function
        def main(x: R.Tensor(ndim=2, dtype="float32")) -> R.Tensor(ndim=2, dtype="float32"):
            R.func_attr({"global_symbol": "main"})
            if relax.const(True, dtype="bool"):
                ret = x
            else:
                ret = x
            return ret

    mod = TestVMIfCondConst
    target = gsmDataGen.target.Target("llvm", host="llvm")
    ex = codegen(mod, target, exec_mode)
    vm = relax.VirtualMachine(ex, gsmDataGen.cpu())
    inp = gsmDataGen.nd.array(np.random.rand(3, 4))
    res = vm["main"](inp)
    gsmDataGen.testing.assert_allclose(res.numpy(), inp.numpy())


@pytest.mark.parametrize("exec_mode", EXEC_MODE)
def test_vm_exec_serialize_export_library(exec_mode):
    @gsmDataGen.script.ir_module
    class TestVMMove:
        @R.function(pure=False)
        def foo(x: R.Tensor((3, 4), "float32")):
            R.func_attr({"global_symbol": "foo"})
            z = R.call_packed("vm.builtin.copy", x, sinfo_args=(R.Tensor((3, 4), dtype="float32")))
            return z

    mod = TestVMMove
    target = gsmDataGen.target.Target("llvm", host="llvm")
    ex = codegen(mod, target)
    from gsmDataGen.contrib import utils

    temp_dir = utils.tempdir()
    path_exec = temp_dir.relpath("exec.so")
    ex.export_library(path_exec)

    loaded_exec = gsmDataGen.runtime.load_module(path_exec)
    assert ex.as_text() == loaded_exec["as_text"]()


@pytest.mark.parametrize("exec_mode", EXEC_MODE)
def test_if_cond(exec_mode):
    @gsmDataGen.script.ir_module
    class TestVMCompileIf:
        @R.function(pure=False)
        def ife(cond: R.Tensor((), "bool"), x: R.Tensor((3, 4), "float32")) -> R.Tensor:
            R.func_attr({"global_symbol": "ife"})
            if cond:
                w = R.call_packed("test.vm.add", x, x, sinfo_args=(R.Tensor))
            else:
                w = R.call_packed("test.vm.mul", x, x, sinfo_args=(R.Tensor))
            return w

    mod = TestVMCompileIf
    target = gsmDataGen.target.Target("llvm", host="llvm")
    ex = codegen(mod, target, exec_mode)
    vm = relax.VirtualMachine(ex, gsmDataGen.cpu())
    inp = gsmDataGen.nd.array(np.random.rand(3, 4))
    res = vm["ife"](gsmDataGen.nd.array(1), inp)
    gsmDataGen.testing.assert_allclose(res.numpy(), inp.numpy() + inp.numpy(), rtol=1e-7, atol=1e-7)
    res = vm["ife"](gsmDataGen.nd.array(True), inp)
    gsmDataGen.testing.assert_allclose(res.numpy(), inp.numpy() + inp.numpy(), rtol=1e-7, atol=1e-7)
    res = vm["ife"](gsmDataGen.nd.array(0), inp)
    gsmDataGen.testing.assert_allclose(res.numpy(), inp.numpy() * inp.numpy(), rtol=1e-7, atol=1e-7)
    res = vm["ife"](gsmDataGen.nd.array(False), inp)
    gsmDataGen.testing.assert_allclose(res.numpy(), inp.numpy() * inp.numpy(), rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize("exec_mode", EXEC_MODE)
def test_vm_return_const_tuple(exec_mode):
    @gsmDataGen.script.ir_module
    class ReturnConstTuple:
        @R.function
        def main(x: R.Tensor(ndim=2, dtype="float32")):
            R.func_attr({"global_symbol": "main"})
            y = R.const([1, 2])
            z = (y, R.const([3, 4]), x)
            return z

    mod = ReturnConstTuple
    target = gsmDataGen.target.Target("llvm", host="llvm")
    ex = codegen(mod, target, exec_mode)
    vm = relax.VirtualMachine(ex, gsmDataGen.cpu())
    inp = gsmDataGen.nd.array(np.random.rand(2, 3))
    res0, res1, res2 = vm["main"](inp)
    gsmDataGen.testing.assert_allclose(res0.numpy(), np.array([1, 2]))
    gsmDataGen.testing.assert_allclose(res1.numpy(), np.array([3, 4]))
    gsmDataGen.testing.assert_allclose(res2.numpy(), inp.numpy())


@pytest.mark.parametrize("exec_mode", EXEC_MODE)
def test_vm_const_as_call_arg(exec_mode):
    @gsmDataGen.script.ir_module
    class TestVMConstAsCallArg:
        @R.function(pure=False)
        def main(x: R.Tensor(ndim=2, dtype="float32")):
            R.func_attr({"global_symbol": "main"})
            a = R.call_packed(
                "test.vm.add",
                relax.const([1, 2]),
                relax.const([3, 4]),
                sinfo_args=(R.Tensor(ndim=2, dtype="float32")),
            )
            b = R.call_packed(
                "test.vm.add",
                a,
                x,
                sinfo_args=(R.Tensor(ndim=2, dtype="float32")),
            )
            return b

    mod = TestVMConstAsCallArg
    target = gsmDataGen.target.Target("llvm", host="llvm")
    ex = codegen(mod, target, exec_mode)
    vm = relax.VirtualMachine(ex, gsmDataGen.cpu())
    inp = gsmDataGen.nd.array(np.random.rand(1, 2))
    res = vm["main"](inp)
    gsmDataGen.testing.assert_allclose(res.numpy(), np.array([4, 6]) + inp.numpy())


@pytest.mark.parametrize("exec_mode", EXEC_MODE)
def test_shape_check_builtin(exec_mode):
    MS = MatchShapeCode
    MK = MakeShapeCode
    # slot assignment:
    # 0: n, 1: m
    sindex = {"n": 0, "m": 1}

    @gsmDataGen.script.ir_module
    class TestVMShapeCheck:
        @R.function(pure=False)
        def main(x: R.Tensor(["n", "m"], "float32")) -> R.Shape(ndim=3):
            R.func_attr({"global_symbol": "main"})
            n = T.int64()
            k = T.int64()
            shape_heap = R.call_builtin_with_ctx(
                "vm.builtin.alloc_shape_heap",
                [R.prim_value(3)],
                sinfo_args=[R.Tensor(ndim=1, dtype="int64")],
            )
            _ = R.call_packed(
                "vm.builtin.check_tensor_info", x, 2, R.dtype("float32"), "", sinfo_args=[R.Tuple()]
            )
            _ = R.call_packed(
                "vm.builtin.match_shape",
                x,
                shape_heap,
                2,
                MS.STORE_TO_HEAP,
                sindex["n"],
                MS.STORE_TO_HEAP,
                sindex["m"],
                "",
                sinfo_args=[R.Tuple()],
            )
            # construct shape value for return
            s = R.call_packed(
                "vm.builtin.make_shape",
                shape_heap,
                3,
                MK.LOAD_SHAPE,
                sindex["m"],
                MK.LOAD_SHAPE,
                sindex["n"],
                MK.USE_IMM,
                2,
                sinfo_args=[R.Shape(ndim=3)],
            )
            return s

    mod = TestVMShapeCheck
    target = gsmDataGen.target.Target("llvm", host="llvm")
    ex = codegen(mod, target, exec_mode)
    vm = relax.VirtualMachine(ex, gsmDataGen.cpu())
    x = gsmDataGen.nd.array(np.zeros((1, 2)).astype("float32"))
    res = vm["main"](x)
    assert res == gsmDataGen.runtime.container.ShapeTuple([2, 1, 2])

    # wrong input type
    with pytest.raises(TypeError):
        vm["main"]([])

    # wrong ndim
    with pytest.raises(ValueError, match=r".*ndim.*"):
        vm["main"](gsmDataGen.nd.array(np.zeros(1).astype("float32")))

    # wrong dtype
    with pytest.raises(ValueError, match=r".*dtype.*"):
        vm["main"](gsmDataGen.nd.array(np.zeros((1, 2)).astype("int32")))


@pytest.mark.parametrize("exec_mode", EXEC_MODE)
def test_prim_value(exec_mode):
    @gsmDataGen.script.ir_module
    class TestVMPrimValue:
        @R.function
        def main():
            R.func_attr({"global_symbol": "main"})
            ret = R.prim_value(T.int64(1))
            return ret

    mod = TestVMPrimValue
    target = gsmDataGen.target.Target("llvm", host="llvm")
    ex = codegen(mod, target, exec_mode)
    vm = relax.VirtualMachine(ex, gsmDataGen.cpu())
    res = vm["main"]()
    assert res == 1


@pytest.mark.parametrize("exec_mode", EXEC_MODE)
def test_string_imm(exec_mode):
    @gsmDataGen.script.ir_module
    class TestVMStringImm:
        @R.function
        def main():
            R.func_attr({"global_symbol": "main"})
            ret = R.str("hello")
            return ret

    mod = TestVMStringImm
    target = gsmDataGen.target.Target("llvm", host="llvm")
    ex = codegen(mod, target, exec_mode)
    vm = relax.VirtualMachine(ex, gsmDataGen.cpu())
    res = vm["main"]()
    assert res == "hello"


@pytest.mark.parametrize("exec_mode", EXEC_MODE)
def test_datatype_imm(exec_mode):
    @gsmDataGen.script.ir_module
    class TestDataTypeImm:
        @R.function
        def main():
            R.func_attr({"global_symbol": "main"})
            ret = R.dtype("float32")
            return ret

    mod = TestDataTypeImm
    target = gsmDataGen.target.Target("llvm", host="llvm")
    ex = codegen(mod, target, exec_mode)
    vm = relax.VirtualMachine(ex, gsmDataGen.cpu())
    res = vm["main"]()
    assert res == "float32"


@pytest.mark.parametrize("exec_mode", EXEC_MODE)
def test_vm_builtin_reshape(exec_mode):
    @gsmDataGen.script.ir_module
    class TestVMBuiltinReshape:
        @R.function(pure=False)
        def main(x: R.Tensor((3, 4), "float32")):
            R.func_attr({"global_symbol": "main"})
            y = R.call_packed(
                "vm.builtin.reshape", x, R.shape([6, 2]), sinfo_args=R.Tensor((6, 2), "float32")
            )
            return y

    mod = TestVMBuiltinReshape
    target = gsmDataGen.target.Target("llvm", host="llvm")
    ex = codegen(mod, target, exec_mode)
    dev = gsmDataGen.cpu()
    vm = relax.VirtualMachine(ex, dev)

    input_np = np.random.rand(3, 4).astype("float32")
    input = gsmDataGen.nd.array(input_np, dev)
    res = vm["main"](input)
    expected = input_np.reshape(6, 2)
    gsmDataGen.testing.assert_allclose(res.numpy(), expected, rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize("exec_mode", EXEC_MODE)
def test_vm_kill_object(exec_mode):
    @I.ir_module
    class TestKillObject:
        @T.prim_func
        def full(T_full: T.Buffer((T.int64(4),), "float32")):
            T.func_attr({"global_symbol": "full", "tir.noalias": True})
            for ax0 in range(T.int64(4)):
                with T.block("T_full"):
                    v_ax0 = T.axis.spatial(T.int64(4), ax0)
                    T.reads()
                    T.writes(T_full[v_ax0])
                    T_full[v_ax0] = T.float32(0)

        @T.prim_func
        def full1(T_full: T.Buffer((T.int64(4),), "float32")):
            T.func_attr({"global_symbol": "full1", "tir.noalias": True})
            for ax0 in range(T.int64(4)):
                with T.block("T_full"):
                    v_ax0 = T.axis.spatial(T.int64(4), ax0)
                    T.reads()
                    T.writes(T_full[v_ax0])
                    T_full[v_ax0] = T.float32(1)

        # PrimFuncs called directly are treated as impure
        @R.function(pure=False)
        def main() -> R.Tensor((4,), dtype="float32"):
            R.func_attr({"global_symbol": "main"})
            cls = TestKillObject
            storage: R.Object = R.vm.alloc_storage(R.shape([16]), R.prim_value(0), R.dtype("uint8"))
            alloc: R.Tensor((4,), dtype="float32") = R.vm.alloc_tensor(
                storage, R.prim_value(0), R.shape([4]), R.dtype("float32")
            )
            _: R.Tuple = cls.full(alloc)
            __1: R.Tuple = R.vm.kill_object(alloc)
            x: R.Tensor((4,), dtype="float32") = alloc
            alloc1: R.Tensor((4,), dtype="float32") = R.vm.alloc_tensor(
                storage, R.prim_value(0), R.shape([4]), R.dtype("float32")
            )
            _1: R.Tuple = cls.full(alloc1)
            _1_1: R.Tuple = R.vm.kill_object(alloc1)
            y: R.Tensor((4,), dtype="float32") = alloc1
            storage_1: R.Object = R.vm.alloc_storage(
                R.shape([16]), R.prim_value(0), R.dtype("uint8")
            )
            alloc2: R.Tensor((4,), dtype="float32") = R.vm.alloc_tensor(
                storage_1, R.prim_value(0), R.shape([4]), R.dtype("float32")
            )
            _2: R.Tuple = cls.full1(alloc2)
            z: R.Tensor((4,), dtype="float32") = alloc2
            _2_1: R.Tuple = R.vm.kill_object(storage)
            return z

    mod = TestKillObject
    target = gsmDataGen.target.Target("llvm", host="llvm")
    ex = codegen(mod, target, exec_mode)
    dev = gsmDataGen.cpu()
    vm = relax.VirtualMachine(ex, dev)

    res = vm["main"]()
    gsmDataGen.testing.assert_allclose(res.numpy(), np.ones((4,), "float32"))


@pytest.mark.parametrize("exec_mode", EXEC_MODE)
def test_preserve_trivial_bindings(exec_mode):
    @I.ir_module
    class mod:
        @R.function(pure=False)
        def main():
            callback = R.ExternFunc("test.vm.check_if_defined")

            storage = R.vm.alloc_storage(R.shape([16]), R.prim_value(0), R.dtype("uint8"))
            alloc = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape([4]), R.dtype("float32"))
            storage_alias = storage
            alloc_alias = alloc

            storage_before = callback(storage)
            alloc_before = callback(alloc)
            storage_alias_before = callback(storage_alias)
            alloc_alias_before = callback(alloc_alias)

            _ = R.vm.kill_object(storage)
            _ = R.vm.kill_object(alloc)

            storage_after = callback(storage)
            alloc_after = callback(alloc)
            storage_alias_after = callback(storage_alias)
            alloc_alias_after = callback(alloc_alias)

            return (
                storage_before,
                alloc_before,
                storage_alias_before,
                alloc_alias_before,
                storage_after,
                alloc_after,
                storage_alias_after,
                alloc_alias_after,
            )

    target = gsmDataGen.target.Target("llvm", host="llvm")
    ex = codegen(mod, target, exec_mode)
    dev = gsmDataGen.cpu()
    vm = relax.VirtualMachine(ex, dev)

    result_list = vm["main"]()

    # Making a dictionary of expected results is purely to improve
    # readability of test failures.  This is equivalent to asserting
    # on each element of the result array, but lets pytest give us a
    # diff of the dictionaries in case of failure.
    expected_results = {
        "storage_before": True,
        "alloc_before": True,
        "storage_alias_before": True,
        "alloc_alias_before": True,
        "storage_after": False,
        "alloc_after": False,
        "storage_alias_after": True,
        "alloc_alias_after": True,
    }

    observed_results = {
        name: bool(tir_bool) for name, tir_bool in zip(expected_results.keys(), result_list)
    }

    assert observed_results == expected_results


if __name__ == "__main__":
    gsmDataGen.testing.main()
