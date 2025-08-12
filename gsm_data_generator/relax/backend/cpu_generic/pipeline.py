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
"""The Relax CPU backend compilation pipeline and other passes."""
import gsm_data_generator
from gsm_data_generator import relax


def library_dispatch_passes(target: gsm_data_generator.target.Target):  # pylint: disable=unused-argument
    """The default library dispatch passes for CPU backend."""
    return []


def legalize_passes(target: gsm_data_generator.target.Target):  # pylint: disable=unused-argument
    """The default legalization passes for CPU backend."""
    return [
        gsm_data_generator.relax.transform.LegalizeOps(),
        gsm_data_generator.relax.transform.AnnotateTIROpPattern(),
        gsm_data_generator.relax.transform.FoldConstant(),
        gsm_data_generator.relax.transform.FuseOps(),
        gsm_data_generator.relax.transform.FuseTIR(),
    ]


def dataflow_lower_passes(target: gsm_data_generator.target.Target):  # pylint: disable=unused-argument
    """The default dataflow lowering passes for CPU backend."""
    return [
        relax.transform.RewriteDataflowReshape(),
        relax.transform.ToNonDataflow(),
        relax.transform.RemovePurityChecking(),
        relax.transform.CallTIRRewrite(),
    ]


def finalize_passes(target: gsm_data_generator.target.Target):  # pylint: disable=unused-argument
    """The default finalization passes for CPU backend."""
    return [
        relax.transform.StaticPlanBlockMemory(),
        relax.transform.LowerAllocTensor(),
        relax.transform.KillAfterLastUse(),
        relax.transform.LowerRuntimeBuiltin(),
        relax.transform.ComputePrimValue(),
        relax.transform.VMShapeLower(),
        relax.transform.AttachGlobalSymbol(),
    ]


def get_default_pipeline(target: gsm_data_generator.target.Target):
    """Return the default compilation pipeline for CPU."""

    @gsm_data_generator.transform.module_pass(opt_level=0)
    def _pipeline(mod: gsm_data_generator.ir.IRModule, _ctx: gsm_data_generator.transform.PassContext):
        with target:
            seq = gsm_data_generator.transform.Sequential(
                library_dispatch_passes(target)
                + legalize_passes(target)
                + dataflow_lower_passes(target)
                + finalize_passes(target)
            )
            mod = seq(mod)
        return mod

    return _pipeline
