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
import gsm_data_generator
from gsm_data_generator import te
from gsm_data_generator.contrib import utils
import numpy as np
import gsm_data_generator.testing
from gsm_data_generator import tir


@gsm_data_generator.testing.requires_gpu
def test_large_uint_imm():
    value = (1 << 63) + 123
    other = gsm_data_generator.tir.const(3, "uint64")
    n = 12
    num_thread = 2

    A = te.compute((n,), lambda *i: gsm_data_generator.tir.const(value, "uint64") + other, name="A")

    # Convert to TIR and create schedule
    mod = te.create_prim_func([A])
    sch = tir.Schedule(mod)

    # Get block and loop
    block = sch.get_block("A")
    loop = sch.get_loops(block)[0]

    # Split and bind
    xo, xi = sch.split(loop, factors=[None, num_thread])
    sch.bind(xi, "threadIdx.x")
    sch.bind(xo, "blockIdx.x")

    def check_target(device):
        if not gsm_data_generator.testing.device_enabled(device):
            return
        dev = gsm_data_generator.device(device, 0)
        f = gsm_data_generator.compile(sch.mod, target=device)
        # launch the kernel.
        a = gsm_data_generator.nd.empty((n,), dtype=A.dtype, device=dev)
        f(a)
        assert a.numpy()[0] == value + 3

    check_target("cuda")
    check_target("vulkan -from_device=0")


@gsm_data_generator.testing.requires_gpu
def test_add_pipeline():
    n = te.size_var("n")
    A = te.placeholder((n,), name="A")
    B = te.placeholder((), name="B")
    C = te.compute(A.shape, lambda *i: A(*i) + B(), name="C")
    D = te.compute(A.shape, lambda *i: C(*i) + 1, name="D")

    # Convert to TIR and create schedule
    mod = te.create_prim_func([A, B, D])
    sch = tir.Schedule(mod)

    # Get blocks and loops
    c_block = sch.get_block("C")
    d_block = sch.get_block("D")
    c_loop = sch.get_loops(c_block)[0]
    d_loop = sch.get_loops(d_block)[0]

    # GPU schedule have to split by gridIdx and threadIdx
    num_thread = 256

    # Schedule C
    c_xo, c_xi = sch.split(c_loop, factors=[None, num_thread])
    sch.bind(c_xi, "threadIdx.x")
    sch.bind(c_xo, "blockIdx.x")

    # Schedule D
    d_xo, d_xi = sch.split(d_loop, factors=[None, num_thread])
    sch.bind(d_xi, "threadIdx.x")
    sch.bind(d_xo, "blockIdx.x")

    def check_target(device, host):
        if not gsm_data_generator.testing.device_enabled(device) or not gsm_data_generator.testing.device_enabled(host):
            return
        dev = gsm_data_generator.device(device, 0)
        target = gsm_data_generator.target.Target(device, host)
        mhost = gsm_data_generator.tir.build(sch.mod, target=target)
        f = mhost.entry_func
        # launch the kernel.
        n = 1027
        a = gsm_data_generator.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
        b = gsm_data_generator.nd.array(np.random.uniform(size=()).astype(B.dtype), dev)
        d = gsm_data_generator.nd.array(np.zeros(n, dtype=D.dtype), dev)
        f(a, b, d)
        gsm_data_generator.testing.assert_allclose(d.numpy(), a.numpy() + b.numpy() + 1)

    check_target("cuda", host="llvm")
    check_target("nvptx", host="llvm")
    check_target("vulkan", host="llvm")
    check_target("rocm", host="llvm")


if __name__ == "__main__":
    test_large_uint_imm()
    test_add_pipeline()
