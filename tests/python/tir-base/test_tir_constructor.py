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
from gsmDataGen import te


def test_expr_constructor():
    x = gsmDataGen.tir.Var("xx", "float32")
    assert isinstance(x, gsmDataGen.tir.Var)
    assert x.name == "xx"

    x = gsmDataGen.tir.Reduce(None, [1], [gsmDataGen.tir.IterVar((0, 1), "x", 2)], None, 0)
    assert isinstance(x, gsmDataGen.tir.Reduce)
    assert x.combiner == None
    assert x.value_index == 0

    x = gsmDataGen.tir.FloatImm("float32", 1.0)
    assert isinstance(x, gsmDataGen.tir.FloatImm)
    assert x.value == 1.0
    assert x.dtype == "float32"

    x = gsmDataGen.tir.IntImm("int64", 2)
    assert isinstance(x, gsmDataGen.tir.IntImm)
    assert x.value == 2
    assert x.dtype == "int64"

    x = gsmDataGen.tir.StringImm("xyza")
    assert isinstance(x, gsmDataGen.tir.StringImm)
    assert x.value == "xyza"

    x = gsmDataGen.tir.Cast("float32", gsmDataGen.tir.IntImm("uint32", 1))
    assert isinstance(x, gsmDataGen.tir.Cast)
    assert x.dtype == "float32"
    assert x.value.value == 1

    a = gsmDataGen.tir.const(1.0, dtype="float32")
    b = te.var("x", dtype="float32")

    for cls in [
        gsmDataGen.tir.Add,
        gsmDataGen.tir.Sub,
        gsmDataGen.tir.Mul,
        gsmDataGen.tir.Div,
        gsmDataGen.tir.Mod,
        gsmDataGen.tir.Min,
        gsmDataGen.tir.Max,
        gsmDataGen.tir.LT,
        gsmDataGen.tir.LE,
        gsmDataGen.tir.GT,
        gsmDataGen.tir.GE,
    ]:
        x = cls(a, b)
        assert isinstance(x, cls)
        assert x.a == a
        assert x.b.same_as(b)

    a = gsmDataGen.runtime.convert(te.var("x") > 1)
    b = gsmDataGen.runtime.convert(te.var("x") == 1)

    for cls in [gsmDataGen.tir.And, gsmDataGen.tir.Or]:
        x = cls(a, b)
        assert isinstance(x, cls)
        assert x.a == a
        assert x.b.same_as(b)

    x = gsmDataGen.tir.Not(a)
    assert isinstance(x, gsmDataGen.tir.Not)
    assert x.a == a

    x = gsmDataGen.tir.Select(a, a, b)
    assert isinstance(x, gsmDataGen.tir.Select)
    assert x.true_value == a
    assert x.false_value == b
    assert x.condition == a

    buffer_var = gsmDataGen.tir.Var("buf", gsmDataGen.ir.PointerType(gsmDataGen.ir.PrimType("float32")))
    buffer = gsmDataGen.tir.decl_buffer([16], "float32", data=buffer_var)
    x = gsmDataGen.tir.BufferLoad(buffer, [1])
    assert isinstance(x, gsmDataGen.tir.BufferLoad)
    assert x.dtype == "float32"
    assert x.buffer == buffer
    assert x.buffer.data == buffer_var
    assert list(x.indices) == [1]

    x = gsmDataGen.tir.Ramp(1, 2, 10)
    assert isinstance(x, gsmDataGen.tir.Ramp)
    assert x.base.value == 1
    assert x.stride.value == 2
    assert x.lanes == 10

    x = gsmDataGen.tir.Broadcast(a, 10)
    assert isinstance(x, gsmDataGen.tir.Broadcast)
    assert x.value == a
    assert x.lanes == 10

    x = gsmDataGen.tir.Shuffle([a], [0])
    assert isinstance(x, gsmDataGen.tir.Shuffle)
    assert x.vectors[0] == a
    assert x.indices[0].value == 0

    x = gsmDataGen.tir.Call("float32", "tir.call_extern", [gsmDataGen.tir.StringImm("xyz"), a])
    assert isinstance(x, gsmDataGen.tir.Call)
    assert x.dtype == "float32"
    assert x.op.name == "tir.call_extern"
    assert x.args[1] == a

    v = te.var("aa")
    x = gsmDataGen.tir.Let(v, 1, v)
    assert x.var == v
    assert x.value.value == 1
    assert x.body == v


def test_stmt_constructor():
    v = te.var("aa")
    nop = gsmDataGen.tir.Evaluate(1)
    x = gsmDataGen.tir.LetStmt(v, 1, gsmDataGen.tir.Evaluate(1))
    assert isinstance(x, gsmDataGen.tir.LetStmt)
    assert x.var == v
    assert x.value.value == 1
    assert isinstance(x.body, gsmDataGen.tir.Evaluate)

    x = gsmDataGen.tir.AttrStmt(v == 1, "xx", 1, gsmDataGen.tir.Evaluate(1))
    assert isinstance(x, gsmDataGen.tir.AttrStmt)
    assert x.value.value == 1

    x = gsmDataGen.tir.AssertStmt(gsmDataGen.tir.const(1, "uint1"), gsmDataGen.runtime.convert("hellow"), nop)
    assert isinstance(x, gsmDataGen.tir.AssertStmt)
    assert x.body == nop

    x = gsmDataGen.tir.For(te.var("x"), 0, 10, gsmDataGen.tir.ForKind.SERIAL, nop)
    assert isinstance(x, gsmDataGen.tir.For)
    assert x.min.value == 0
    assert x.extent.value == 10
    assert x.body == nop

    buffer_var = gsmDataGen.tir.Var("buf", gsmDataGen.ir.PointerType(gsmDataGen.ir.PrimType("uint1")))
    buffer = gsmDataGen.tir.decl_buffer([16], "uint1", data=buffer_var)
    x = gsmDataGen.tir.BufferStore(buffer, gsmDataGen.tir.IntImm("bool", 1), [10])
    assert isinstance(x, gsmDataGen.tir.BufferStore)
    assert x.buffer == buffer
    assert x.buffer.data == buffer_var
    assert list(x.indices) == [10]
    assert x.value.value == 1

    buffer_var = gsmDataGen.tir.Var("buf", gsmDataGen.ir.PointerType(gsmDataGen.ir.PrimType("float32")))
    x = gsmDataGen.tir.Allocate(buffer_var, "float32", [10], gsmDataGen.tir.const(1, "uint1"), nop)
    assert isinstance(x, gsmDataGen.tir.Allocate)
    assert x.dtype == "float32"
    assert x.buffer_var == buffer_var
    assert x.body == nop

    storage_scope = "global.texture"
    buffer_var = gsmDataGen.tir.Var("buf", gsmDataGen.ir.PointerType(gsmDataGen.ir.PrimType("float32"), storage_scope))
    x = gsmDataGen.tir.Allocate(buffer_var, "float32", [10], gsmDataGen.tir.const(1, "uint1"), nop)
    assert isinstance(x, gsmDataGen.tir.Allocate)
    assert x.dtype == "float32"
    assert x.buffer_var == buffer_var
    assert x.buffer_var.type_annotation.storage_scope == storage_scope
    assert x.body == nop

    x = gsmDataGen.tir.AttrStmt(buffer_var, "xyz", 1, nop)
    assert isinstance(x, gsmDataGen.tir.AttrStmt)
    assert x.node == buffer_var
    assert x.attr_key == "xyz"
    assert x.body == nop

    x = gsmDataGen.tir.IfThenElse(gsmDataGen.tir.const(1, "uint1"), gsmDataGen.tir.Evaluate(11), nop)
    assert isinstance(x, gsmDataGen.tir.IfThenElse)
    assert x.then_case.value.value == 11
    assert x.else_case == nop


def test_float_constructor_requires_float_dtype():
    with pytest.raises(gsmDataGen.TVMError):
        gsmDataGen.tir.FloatImm("int32", 1.0)


if __name__ == "__main__":
    gsmDataGen.testing.main()
