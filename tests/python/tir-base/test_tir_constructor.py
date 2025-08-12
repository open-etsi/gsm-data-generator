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
import gsm_data_generator
from gsm_data_generator import te


def test_expr_constructor():
    x = gsm_data_generator.tir.Var("xx", "float32")
    assert isinstance(x, gsm_data_generator.tir.Var)
    assert x.name == "xx"

    x = gsm_data_generator.tir.Reduce(None, [1], [gsm_data_generator.tir.IterVar((0, 1), "x", 2)], None, 0)
    assert isinstance(x, gsm_data_generator.tir.Reduce)
    assert x.combiner == None
    assert x.value_index == 0

    x = gsm_data_generator.tir.FloatImm("float32", 1.0)
    assert isinstance(x, gsm_data_generator.tir.FloatImm)
    assert x.value == 1.0
    assert x.dtype == "float32"

    x = gsm_data_generator.tir.IntImm("int64", 2)
    assert isinstance(x, gsm_data_generator.tir.IntImm)
    assert x.value == 2
    assert x.dtype == "int64"

    x = gsm_data_generator.tir.StringImm("xyza")
    assert isinstance(x, gsm_data_generator.tir.StringImm)
    assert x.value == "xyza"

    x = gsm_data_generator.tir.Cast("float32", gsm_data_generator.tir.IntImm("uint32", 1))
    assert isinstance(x, gsm_data_generator.tir.Cast)
    assert x.dtype == "float32"
    assert x.value.value == 1

    a = gsm_data_generator.tir.const(1.0, dtype="float32")
    b = te.var("x", dtype="float32")

    for cls in [
        gsm_data_generator.tir.Add,
        gsm_data_generator.tir.Sub,
        gsm_data_generator.tir.Mul,
        gsm_data_generator.tir.Div,
        gsm_data_generator.tir.Mod,
        gsm_data_generator.tir.Min,
        gsm_data_generator.tir.Max,
        gsm_data_generator.tir.LT,
        gsm_data_generator.tir.LE,
        gsm_data_generator.tir.GT,
        gsm_data_generator.tir.GE,
    ]:
        x = cls(a, b)
        assert isinstance(x, cls)
        assert x.a == a
        assert x.b.same_as(b)

    a = gsm_data_generator.runtime.convert(te.var("x") > 1)
    b = gsm_data_generator.runtime.convert(te.var("x") == 1)

    for cls in [gsm_data_generator.tir.And, gsm_data_generator.tir.Or]:
        x = cls(a, b)
        assert isinstance(x, cls)
        assert x.a == a
        assert x.b.same_as(b)

    x = gsm_data_generator.tir.Not(a)
    assert isinstance(x, gsm_data_generator.tir.Not)
    assert x.a == a

    x = gsm_data_generator.tir.Select(a, a, b)
    assert isinstance(x, gsm_data_generator.tir.Select)
    assert x.true_value == a
    assert x.false_value == b
    assert x.condition == a

    buffer_var = gsm_data_generator.tir.Var("buf", gsm_data_generator.ir.PointerType(gsm_data_generator.ir.PrimType("float32")))
    buffer = gsm_data_generator.tir.decl_buffer([16], "float32", data=buffer_var)
    x = gsm_data_generator.tir.BufferLoad(buffer, [1])
    assert isinstance(x, gsm_data_generator.tir.BufferLoad)
    assert x.dtype == "float32"
    assert x.buffer == buffer
    assert x.buffer.data == buffer_var
    assert list(x.indices) == [1]

    x = gsm_data_generator.tir.Ramp(1, 2, 10)
    assert isinstance(x, gsm_data_generator.tir.Ramp)
    assert x.base.value == 1
    assert x.stride.value == 2
    assert x.lanes == 10

    x = gsm_data_generator.tir.Broadcast(a, 10)
    assert isinstance(x, gsm_data_generator.tir.Broadcast)
    assert x.value == a
    assert x.lanes == 10

    x = gsm_data_generator.tir.Shuffle([a], [0])
    assert isinstance(x, gsm_data_generator.tir.Shuffle)
    assert x.vectors[0] == a
    assert x.indices[0].value == 0

    x = gsm_data_generator.tir.Call("float32", "tir.call_extern", [gsm_data_generator.tir.StringImm("xyz"), a])
    assert isinstance(x, gsm_data_generator.tir.Call)
    assert x.dtype == "float32"
    assert x.op.name == "tir.call_extern"
    assert x.args[1] == a

    v = te.var("aa")
    x = gsm_data_generator.tir.Let(v, 1, v)
    assert x.var == v
    assert x.value.value == 1
    assert x.body == v


def test_stmt_constructor():
    v = te.var("aa")
    nop = gsm_data_generator.tir.Evaluate(1)
    x = gsm_data_generator.tir.LetStmt(v, 1, gsm_data_generator.tir.Evaluate(1))
    assert isinstance(x, gsm_data_generator.tir.LetStmt)
    assert x.var == v
    assert x.value.value == 1
    assert isinstance(x.body, gsm_data_generator.tir.Evaluate)

    x = gsm_data_generator.tir.AttrStmt(v == 1, "xx", 1, gsm_data_generator.tir.Evaluate(1))
    assert isinstance(x, gsm_data_generator.tir.AttrStmt)
    assert x.value.value == 1

    x = gsm_data_generator.tir.AssertStmt(gsm_data_generator.tir.const(1, "uint1"), gsm_data_generator.runtime.convert("hellow"), nop)
    assert isinstance(x, gsm_data_generator.tir.AssertStmt)
    assert x.body == nop

    x = gsm_data_generator.tir.For(te.var("x"), 0, 10, gsm_data_generator.tir.ForKind.SERIAL, nop)
    assert isinstance(x, gsm_data_generator.tir.For)
    assert x.min.value == 0
    assert x.extent.value == 10
    assert x.body == nop

    buffer_var = gsm_data_generator.tir.Var("buf", gsm_data_generator.ir.PointerType(gsm_data_generator.ir.PrimType("uint1")))
    buffer = gsm_data_generator.tir.decl_buffer([16], "uint1", data=buffer_var)
    x = gsm_data_generator.tir.BufferStore(buffer, gsm_data_generator.tir.IntImm("bool", 1), [10])
    assert isinstance(x, gsm_data_generator.tir.BufferStore)
    assert x.buffer == buffer
    assert x.buffer.data == buffer_var
    assert list(x.indices) == [10]
    assert x.value.value == 1

    buffer_var = gsm_data_generator.tir.Var("buf", gsm_data_generator.ir.PointerType(gsm_data_generator.ir.PrimType("float32")))
    x = gsm_data_generator.tir.Allocate(buffer_var, "float32", [10], gsm_data_generator.tir.const(1, "uint1"), nop)
    assert isinstance(x, gsm_data_generator.tir.Allocate)
    assert x.dtype == "float32"
    assert x.buffer_var == buffer_var
    assert x.body == nop

    storage_scope = "global.texture"
    buffer_var = gsm_data_generator.tir.Var("buf", gsm_data_generator.ir.PointerType(gsm_data_generator.ir.PrimType("float32"), storage_scope))
    x = gsm_data_generator.tir.Allocate(buffer_var, "float32", [10], gsm_data_generator.tir.const(1, "uint1"), nop)
    assert isinstance(x, gsm_data_generator.tir.Allocate)
    assert x.dtype == "float32"
    assert x.buffer_var == buffer_var
    assert x.buffer_var.type_annotation.storage_scope == storage_scope
    assert x.body == nop

    x = gsm_data_generator.tir.AttrStmt(buffer_var, "xyz", 1, nop)
    assert isinstance(x, gsm_data_generator.tir.AttrStmt)
    assert x.node == buffer_var
    assert x.attr_key == "xyz"
    assert x.body == nop

    x = gsm_data_generator.tir.IfThenElse(gsm_data_generator.tir.const(1, "uint1"), gsm_data_generator.tir.Evaluate(11), nop)
    assert isinstance(x, gsm_data_generator.tir.IfThenElse)
    assert x.then_case.value.value == 11
    assert x.else_case == nop


def test_float_constructor_requires_float_dtype():
    with pytest.raises(gsm_data_generator.TVMError):
        gsm_data_generator.tir.FloatImm("int32", 1.0)


if __name__ == "__main__":
    gsm_data_generator.testing.main()
