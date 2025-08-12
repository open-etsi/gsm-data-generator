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
"""Relax hexagon test."""

import numpy as np
import pytest
import gsm_data_generator.testing
from gsm_data_generator import relax, runtime
from gsm_data_generator.relax.frontend import onnx
from gsm_data_generator.relax.testing import relay_translator
from gsm_data_generator.contrib.hexagon.session import Session


def get_onnx_mobilenet():
    """Download and import mobilenet model with ONNX"""
    import onnx  # pylint: disable=import-outside-toplevel

    # pylint: disable=line-too-long
    model_url = "https://github.com/onnx/models/raw/131c99da401c757207a40189385410e238ed0934/vision/classification/mobilenet/model/mobilenetv2-7.onnx"
    model_path = gsm_data_generator.contrib.download.download_testdata(
        model_url, "mobilenetv2-7.onnx", module="onnx"
    )
    return onnx.load(model_path)


@pytest.mark.skip("takes too long (~20min)")
@gsm_data_generator.testing.requires_hexagon
def test_mobilenet_onnx(hexagon_session: Session):
    """Test MobileNetV2 ONNX model"""
    onnx_model = get_onnx_mobilenet()
    data_np = np.random.rand(1, 3, 224, 224).astype("float32")
    shape_dict = {"input": data_np.shape}
    relay_mod, _ = relay.frontend.from_onnx(onnx_model, shape_dict, freeze_params=True)

    target_hexagon = gsm_data_generator.target.hexagon("v68")
    target = gsm_data_generator.target.Target(target_hexagon, host=target_hexagon)
    relax_mod = onnx.from_onnx(onnx_model, shape_dict, freeze_params=True)
    relax_mod = relay_translator.from_relay(relay_mod["main"], target_hexagon)

    # Compile and run on Hexagon.
    exe = gsm_data_generator.compile(relax_mod, target)
    dev = hexagon_session.device

    vm_mod = hexagon_session.get_executor_from_factory(exe)
    vm_rt = relax.VirtualMachine(vm_mod, dev)
    data = gsm_data_generator.nd.array(data_np, dev)
    vm_rt.set_input("main", data)
    vm_rt.invoke_stateful("main")
    hexagon_res = vm_rt.get_outputs("main")

    # Compile and run on LLVM for comparison.
    relax_mod = relay_translator.from_relay(relay_mod["main"], "llvm")
    exe = gsm_data_generator.compile(relax_mod, "llvm")
    dev = gsm_data_generator.cpu()
    vm_rt = relax.VirtualMachine(exe, dev)
    data = gsm_data_generator.nd.array(data_np, dev)
    llvm_res = vm_rt["main"](data)
    gsm_data_generator.testing.assert_allclose(hexagon_res.numpy(), llvm_res.numpy(), rtol=1e-3)


@pytest.mark.skip("takes too long (~20min)")
@gsm_data_generator.testing.requires_hexagon
def test_mobilenet(hexagon_session: Session):
    """Test MobileNet workload"""
    relay_mod, params = testing.mobilenet.get_workload(batch_size=1, dtype="float32")
    data_np = np.random.rand(1, 3, 224, 224).astype("float32")

    target_hexagon = gsm_data_generator.target.hexagon("v68")
    target = gsm_data_generator.target.Target(target_hexagon, host=target_hexagon)

    # translate the relay mobilenet and bind params
    relax_mod = relay_translator.from_relay(relay_mod["main"], target, params)

    # Compile and run on Hexagon.
    exe = gsm_data_generator.compile(relax_mod, target)
    dev = hexagon_session.device

    vm_mod = hexagon_session.get_executor_from_factory(exe)
    vm_rt = relax.VirtualMachine(vm_mod, dev)
    data = gsm_data_generator.nd.array(data_np, dev)
    vm_rt.set_input("main", data)
    vm_rt.invoke_stateful("main")
    hexagon_res = vm_rt.get_outputs("main")

    # Compile and run on LLVM for comparison.
    relax_mod = relay_translator.from_relay(relay_mod["main"], "llvm", params)
    exe = gsm_data_generator.compile(relax_mod, "llvm")
    dev = gsm_data_generator.cpu()
    vm_rt = relax.VirtualMachine(exe, dev)
    data = gsm_data_generator.nd.array(data_np, dev)
    llvm_res = vm_rt["main"](data)
    gsm_data_generator.testing.assert_allclose(hexagon_res.numpy(), llvm_res.numpy(), rtol=1e-3)


if __name__ == "__main__":
    gsm_data_generator.testing.main()
