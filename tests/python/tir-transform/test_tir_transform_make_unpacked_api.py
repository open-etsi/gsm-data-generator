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
import pytest

import gsmDataGen
from gsmDataGen import te, tir
from gsmDataGen.script import tir as T, ir as I
import numpy


@pytest.fixture
def mod_without_attrs():
    ib = gsmDataGen.tir.ir_builder.create()
    A = gsmDataGen.tir.decl_buffer(name="A", shape=[1])
    stmt = ib.get()
    return gsmDataGen.IRModule.from_expr(gsmDataGen.tir.PrimFunc([A], stmt))


@pytest.fixture
def mod(mod_without_attrs):
    mod = gsmDataGen.tir.transform.Apply(lambda f: f.with_attr("target", gsmDataGen.target.Target("llvm")))(
        mod_without_attrs
    )
    mod = gsmDataGen.tir.transform.Apply(lambda f: f.with_attr("global_symbol", "main"))(mod)

    return mod


def test_noop_if_not_global_symbol(mod_without_attrs):
    target = gsmDataGen.target.Target("llvm", host="llvm")
    before = gsmDataGen.tir.transform.Apply(lambda f: f.with_attr("target", target))(mod_without_attrs)
    after = gsmDataGen.tir.transform.MakeUnpackedAPI()(before)
    gsmDataGen.ir.assert_structural_equal(before, after)


def test_fails_if_no_target(mod_without_attrs):
    mod = gsmDataGen.tir.transform.Apply(lambda f: f.with_attr("global_symbol", "main"))(mod_without_attrs)
    with pytest.raises(
        gsmDataGen.TVMError,
        match="MakeUnpackedAPI required the function to be annotated with tvm::attr::kTarget",
    ):
        f = gsmDataGen.tir.transform.MakeUnpackedAPI()(mod)["main"]


@gsmDataGen.testing.parametrize_targets("c", "llvm", "cuda")
def test_device_setup(mod, target, dev):
    target = gsmDataGen.target.Target(target, host="llvm")
    mod = gsmDataGen.tir.transform.Apply(lambda f: f.with_attr("target", target))(mod)
    f = gsmDataGen.tir.transform.MakeUnpackedAPI()(mod)["main"]
    assert len(f.params) == 1
    assert f.params[0].name == "A"
    assert f.body.node == "default"
    assert f.body.attr_key == "device_id"
    assert f.body.value == 0
    assert f.body.body.node == "default"
    assert f.body.body.attr_key == "device_type"
    assert f.body.body.value == dev.device_type


def test_no_buffers_no_device_setup():
    ib = gsmDataGen.tir.ir_builder.create()
    A = ib.pointer("float32", name="A")
    stmt = ib.get()
    mod = gsmDataGen.IRModule.from_expr(gsmDataGen.tir.PrimFunc([A], stmt))
    mod = gsmDataGen.tir.transform.Apply(lambda f: f.with_attr("target", gsmDataGen.target.Target("llvm")))(mod)
    mod = gsmDataGen.tir.transform.Apply(lambda f: f.with_attr("global_symbol", "main"))(mod)

    f = gsmDataGen.tir.transform.MakeUnpackedAPI()(mod)["main"]
    assert len(f.params) == 1
    assert f.params[0].name == "A"


def test_argument_mapping(mod):
    f = gsmDataGen.tir.transform.MakeUnpackedAPI()(mod)["main"]
    assert len(f.params) == 1
    assert f.params[0].name == "A"


def test_argument_mapping_multiple():
    ib = gsmDataGen.tir.ir_builder.create()
    A = gsmDataGen.tir.decl_buffer(name="A", shape=[1])
    B = gsmDataGen.tir.decl_buffer(name="B", shape=[1])

    stmt = ib.get()
    mod = gsmDataGen.IRModule.from_expr(gsmDataGen.tir.PrimFunc([A, B], stmt))
    mod = gsmDataGen.tir.transform.Apply(lambda f: f.with_attr("target", gsmDataGen.target.Target("llvm")))(mod)
    mod = gsmDataGen.tir.transform.Apply(lambda f: f.with_attr("global_symbol", "main"))(mod)

    f = gsmDataGen.tir.transform.MakeUnpackedAPI()(mod)["main"]
    assert len(f.params) == 2
    assert f.params[0].name == "A"
    assert f.params[1].name == "B"


def test_argument_mapping_multiple_matching():
    ib = gsmDataGen.tir.ir_builder.create()
    A = gsmDataGen.tir.decl_buffer(name="A", shape=[1])
    B = gsmDataGen.tir.decl_buffer(name="B", shape=[1])
    stmt = ib.get()
    mod = gsmDataGen.IRModule.from_expr(gsmDataGen.tir.PrimFunc([A, A], stmt))
    mod = gsmDataGen.tir.transform.Apply(lambda f: f.with_attr("target", gsmDataGen.target.Target("llvm")))(mod)
    mod = gsmDataGen.tir.transform.Apply(lambda f: f.with_attr("global_symbol", "main"))(mod)

    f = gsmDataGen.tir.transform.MakeUnpackedAPI()(mod)["main"]
    assert len(f.params) == 2
    assert f.params[0].name == "A"
    assert f.params[1].name == "A"


def test_body():
    ib = gsmDataGen.tir.ir_builder.create()
    A = gsmDataGen.tir.decl_buffer(name="A", shape=[1])
    B = gsmDataGen.tir.decl_buffer(name="B", shape=[1])
    C = ib.buffer_ptr(A)

    stmt = ib.get()
    mod = gsmDataGen.IRModule.from_expr(gsmDataGen.tir.PrimFunc([A, B, C], stmt))
    mod = gsmDataGen.tir.transform.Apply(lambda f: f.with_attr("target", gsmDataGen.target.Target("llvm")))(mod)
    mod = gsmDataGen.tir.transform.Apply(lambda f: f.with_attr("global_symbol", "main"))(mod)
    f = gsmDataGen.tir.transform.MakeUnpackedAPI()(mod)["main"]
    assert len(f.params) == 3
    assert f.params[0].name == "A"
    assert f.params[1].name == "B"
    assert f.params[2].name == "A"


class TestTargetHostRemoved(gsmDataGen.testing.CompareBeforeAfter):
    """After MakeUnpackedAPI, host-side target should be the host

    MakeUnpackedAPI is the last transform that requires both the device
    and the host.  After MakeUnpackedAPI, the target attribute should
    only contain the host-side target.
    """

    transform = gsmDataGen.tir.transform.MakeUnpackedAPI()

    def before(self):
        @I.ir_module
        class mod:
            @T.prim_func
            def main(A: T.Buffer(1, "float32")):
                T.func_attr({"global_symbol": "main", "target": T.target("cuda", host="llvm")})
                mod.subroutine(A.data)

            @T.prim_func(private=True)
            def subroutine(A_data: T.handle("float32")):
                T.func_attr({"target": T.target("cuda")})
                T.evaluate(A_data)

        return mod

    def expected(self):
        @I.ir_module
        class mod:
            @T.prim_func
            def main(A_data: T.handle("float32")) -> T.int32:
                T.func_attr({"global_symbol": "main", "target": T.target("llvm")})
                T.attr("default", "device_id", 0)
                T.attr("default", "device_type", 2)
                mod.subroutine(A_data)
                T.ret(T.int32(0))

            @T.prim_func(private=True)
            def subroutine(A_data: T.handle("float32")):
                T.func_attr({"target": T.target("cuda")})
                T.evaluate(A_data)

        return mod


class TestInternalSubroutineCall(gsmDataGen.testing.CompareBeforeAfter):
    """Internal subroutines do not require modification

    A subroutine without the "global_symbol" attribute is an internal
    subroutine, and is not directly exposed to a user of the generated
    `runtime.Module`.
    """

    transform = gsmDataGen.tir.transform.MakeUnpackedAPI()

    def before(self):
        @I.ir_module
        class mod:
            @T.prim_func
            def main(A: T.Buffer(1, "float32")):
                T.func_attr({"global_symbol": "main", "target": T.target("llvm", host="llvm")})
                mod.subroutine(A.data)

            @T.prim_func(private=True)
            def subroutine(A_data: T.handle("float32")):
                T.func_attr({"target": T.target("llvm")})
                T.evaluate(A_data)

        return mod

    def expected(self):
        @I.ir_module
        class mod:
            @T.prim_func
            def main(A_data: T.handle("float32")) -> T.int32:
                T.func_attr({"global_symbol": "main", "target": T.target("llvm")})
                T.attr("default", "device_id", 0)
                T.attr("default", "device_type", 1)
                mod.subroutine(A_data)
                T.ret(T.int32(0))

            @T.prim_func(private=True)
            def subroutine(A_data: T.handle("float32")):
                T.func_attr({"target": T.target("llvm")})
                T.evaluate(A_data)

        return mod


class TestSubroutineCallToExternallyVisibleSubroutine(gsmDataGen.testing.CompareBeforeAfter):
    """Externally-visible subroutines should be updated

    Subroutines that are exposed externally should be updated by
    MakeUnpackedAPI.
    """

    transform = gsmDataGen.tir.transform.MakeUnpackedAPI()

    def before(self):
        @I.ir_module
        class mod:
            @T.prim_func
            def main(A: T.Buffer(1, "float32")):
                T.func_attr({"global_symbol": "main", "target": T.target("llvm", host="llvm")})
                mod.subroutine(A.data)

            @T.prim_func
            def subroutine(A_data: T.handle("float32")):
                T.func_attr(
                    {"global_symbol": "subroutine", "target": T.target("llvm", host="llvm")}
                )
                T.evaluate(A_data)

        return mod

    def expected(self):
        @I.ir_module
        class mod:
            @T.prim_func
            def main(A_data: T.handle("float32")) -> T.int32:
                T.func_attr({"global_symbol": "main", "target": T.target("llvm")})
                T.attr("default", "device_id", 0)
                T.attr("default", "device_type", 1)
                mod.subroutine(A_data)
                T.ret(T.int32(0))

            @T.prim_func
            def subroutine(A_data: T.handle("float32")) -> T.int32:
                T.func_attr({"global_symbol": "subroutine", "target": T.target("llvm")})
                T.evaluate(A_data)
                T.ret(T.int32(0))

        return mod


class TestCallExternallyVisibleSubroutineWithDLTensor(gsmDataGen.testing.CompareBeforeAfter):
    """Callsites of externally-visible subroutines may require updates

    The MakeUnpackedAPI transform lowers all buffers into a data
    pointer to a primitive type.  If a subroutine call is currently
    passing a DLTensor produced by `T.tvm_make_stack_array` into the
    subroutine, the callsite should be updated to instead pass the
    data pointer directly.
    """

    transform = gsmDataGen.tir.transform.MakeUnpackedAPI()

    def before(self):
        @I.ir_module
        class mod:
            @T.prim_func
            def main(A: T.Buffer(1, "float32")):
                T.func_attr({"global_symbol": "main", "target": T.target("llvm", host="llvm")})
                mod.subroutine(
                    T.tvm_stack_make_array(
                        A.data,
                        T.tvm_stack_make_shape(1, dtype="handle"),
                        T.reinterpret(T.uint64(0), dtype="handle"),
                        T.uint32(1),
                        T.Cast("float32", 0),
                        0,
                        dtype="handle",
                    )
                )

            @T.prim_func
            def subroutine(A: T.Buffer(1, "float32")):
                T.func_attr(
                    {"global_symbol": "subroutine", "target": T.target("llvm", host="llvm")}
                )
                T.evaluate(A.data)

        return mod

    def expected(self):
        @I.ir_module
        class mod:
            @T.prim_func
            def main(A_data: T.handle("float32")) -> T.int32:
                T.func_attr({"global_symbol": "main", "target": T.target("llvm")})
                T.attr("default", "device_id", 0)
                T.attr("default", "device_type", 1)
                mod.subroutine(A_data)
                T.ret(T.int32(0))

            @T.prim_func
            def subroutine(A_data: T.handle("float32")) -> T.int32:
                T.func_attr({"global_symbol": "subroutine", "target": T.target("llvm")})
                T.attr("default", "device_id", 0)
                T.attr("default", "device_type", 1)
                T.evaluate(A_data)
                T.ret(T.int32(0))

        return mod


if __name__ == "__main__":
    gsmDataGen.testing.main()
