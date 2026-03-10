# Copyright 2023-2026 SGLang Team
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
# ==============================================================================
"""Parallelism and distributed serving argument group."""

from __future__ import annotations

import argparse
import dataclasses
from typing import Optional


@dataclasses.dataclass
class ParallelArgs:
    """Arguments for parallelism, distributed topology, and device placement."""

    # Device
    device: Optional[str] = None

    # Tensor / pipeline / data / expert parallelism
    tp_size: int = 1
    pp_size: int = 1
    pp_max_micro_batch_size: Optional[int] = None
    pp_async_batch_depth: int = 0
    dp_size: int = 1
    ep_size: int = 1
    attn_cp_size: int = 1
    moe_dp_size: int = 1

    # Multi-node distributed serving
    nnodes: int = 1
    node_rank: int = 0
    dist_init_addr: Optional[str] = None

    # GPU placement
    base_gpu_id: int = 0
    gpu_id_step: int = 1
    nccl_port: Optional[int] = None

    # Distributed timeout
    dist_timeout: Optional[int] = None

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        group = parser.add_argument_group("Parallelism and distributed")
        group.add_argument(
            "--device",
            type=str,
            default=ParallelArgs.device,
            help="The device to use ('cuda', 'xpu', 'hpu', 'npu', 'cpu'). Defaults to auto-detection if not specified.",
        )
        group.add_argument(
            "--tensor-parallel-size",
            "--tp-size",
            type=int,
            default=ParallelArgs.tp_size,
            help="The tensor parallelism size.",
        )
        group.add_argument(
            "--pipeline-parallel-size",
            "--pp-size",
            type=int,
            default=ParallelArgs.pp_size,
            help="The pipeline parallelism size.",
        )
        group.add_argument(
            "--pp-max-micro-batch-size",
            type=int,
            default=ParallelArgs.pp_max_micro_batch_size,
            help="The maximum micro batch size in pipeline parallelism.",
        )
        group.add_argument(
            "--pp-async-batch-depth",
            type=int,
            default=ParallelArgs.pp_async_batch_depth,
            help="The async batch depth of pipeline parallelism.",
        )
        group.add_argument(
            "--data-parallel-size",
            "--dp-size",
            type=int,
            default=ParallelArgs.dp_size,
            help="The data parallelism size.",
        )
        group.add_argument(
            "--expert-parallel-size",
            "--ep-size",
            "--ep",
            type=int,
            default=ParallelArgs.ep_size,
            help="The expert parallelism size.",
        )
        group.add_argument(
            "--attention-context-parallel-size",
            "--attn-cp-size",
            type=int,
            default=ParallelArgs.attn_cp_size,
            help="The attention context parallelism size.",
        )
        group.add_argument(
            "--moe-data-parallel-size",
            "--moe-dp-size",
            type=int,
            default=ParallelArgs.moe_dp_size,
            help="The moe data parallelism size.",
        )
        group.add_argument(
            "--nnodes",
            type=int,
            default=ParallelArgs.nnodes,
            help="The number of nodes.",
        )
        group.add_argument(
            "--node-rank",
            type=int,
            default=ParallelArgs.node_rank,
            help="The node rank.",
        )
        group.add_argument(
            "--dist-init-addr",
            "--nccl-init-addr",
            type=str,
            help="The host address for initializing distributed backend (e.g., `192.168.0.2:25000`).",
        )
        group.add_argument(
            "--base-gpu-id",
            type=int,
            default=ParallelArgs.base_gpu_id,
            help="The base GPU ID to start allocating GPUs from. Useful when running multiple instances on the same machine.",
        )
        group.add_argument(
            "--gpu-id-step",
            type=int,
            default=ParallelArgs.gpu_id_step,
            help="The delta between consecutive GPU IDs that are used. For example, setting it to 2 will use GPU 0,2,4,...",
        )
        group.add_argument(
            "--nccl-port",
            type=int,
            default=ParallelArgs.nccl_port,
            help="The port for NCCL initialization.",
        )
        group.add_argument(
            "--dist-timeout",
            type=int,
            default=ParallelArgs.dist_timeout,
            help="Set timeout for torch.distributed initialization.",
        )
