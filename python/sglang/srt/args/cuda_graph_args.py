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
"""CUDA graph and torch.compile argument group."""

from __future__ import annotations

import argparse
import dataclasses
from typing import List, Optional


@dataclasses.dataclass
class CudaGraphArgs:
    """Arguments for CUDA graph capture, piecewise CUDA graph, and torch.compile."""

    # CUDA graph
    cuda_graph_max_bs: Optional[int] = None
    cuda_graph_bs: Optional[List[int]] = None
    disable_cuda_graph: bool = False
    disable_cuda_graph_padding: bool = False
    enable_profile_cuda_graph: bool = False
    enable_cudagraph_gc: bool = False

    # Piecewise CUDA graph
    disable_piecewise_cuda_graph: bool = False
    enforce_piecewise_cuda_graph: bool = False
    piecewise_cuda_graph_max_tokens: Optional[int] = None
    piecewise_cuda_graph_tokens: Optional[List[int]] = None
    piecewise_cuda_graph_compiler: str = "eager"

    # Torch compile
    enable_torch_compile: bool = False
    enable_torch_compile_debug_mode: bool = False
    torch_compile_max_bs: int = 32
    torchao_config: str = ""

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        from sglang.srt.server_args import DeprecatedAction

        group = parser.add_argument_group("CUDA graph and torch.compile")
        group.add_argument(
            "--cuda-graph-max-bs",
            type=int,
            default=CudaGraphArgs.cuda_graph_max_bs,
            help="Set the maximum batch size for cuda graph. It will extend the cuda graph capture batch size to this value.",
        )
        group.add_argument(
            "--cuda-graph-bs",
            type=int,
            nargs="+",
            help="Set the list of batch sizes for cuda graph.",
        )
        group.add_argument(
            "--disable-cuda-graph",
            action="store_true",
            help="Disable cuda graph.",
        )
        group.add_argument(
            "--disable-cuda-graph-padding",
            action="store_true",
            help="Disable cuda graph when padding is needed. Still uses cuda graph when padding is not needed.",
        )
        group.add_argument(
            "--enable-profile-cuda-graph",
            action="store_true",
            help="Enable profiling of cuda graph capture.",
        )
        group.add_argument(
            "--enable-cudagraph-gc",
            action="store_true",
            help="Enable garbage collection during CUDA graph capture. If disabled (default), GC is frozen during capture to speed up the process.",
        )
        group.add_argument(
            "--disable-piecewise-cuda-graph",
            action="store_true",
            help="Disable piecewise cuda graph for extend/prefill.",
        )
        group.add_argument(
            "--enable-piecewise-cuda-graph",
            action=DeprecatedAction,
            help="Deprecated: Piecewise cuda graph is enabled by default. Use --enforce-piecewise-cuda-graph to skip auto-disable conditions.",
        )
        group.add_argument(
            "--enforce-piecewise-cuda-graph",
            action="store_true",
            help="Enforce piecewise cuda graph, skipping all auto-disable conditions. Used for testing.",
        )
        group.add_argument(
            "--piecewise-cuda-graph-tokens",
            type=int,
            nargs="+",
            help="Set the list of token lengths for piecewise cuda graph capture.",
        )
        group.add_argument(
            "--piecewise-cuda-graph-compiler",
            type=str,
            default=CudaGraphArgs.piecewise_cuda_graph_compiler,
            help="Set the compiler for piecewise cuda graph. Choices are: eager, inductor.",
            choices=["eager", "inductor"],
        )
        group.add_argument(
            "--piecewise-cuda-graph-max-tokens",
            type=int,
            default=CudaGraphArgs.piecewise_cuda_graph_max_tokens,
            help="Set the maximum tokens when using piecewise cuda graph.",
        )
        group.add_argument(
            "--enable-torch-compile",
            action="store_true",
            help="Optimize the model with torch.compile. Experimental feature.",
        )
        group.add_argument(
            "--enable-torch-compile-debug-mode",
            action="store_true",
            help="Enable debug mode for torch compile",
        )
        group.add_argument(
            "--torch-compile-max-bs",
            type=int,
            default=CudaGraphArgs.torch_compile_max_bs,
            help="Set the maximum batch size when using torch compile.",
        )
        group.add_argument(
            "--torchao-config",
            type=str,
            default=CudaGraphArgs.torchao_config,
            help="Optimize the model with torchao. Experimental feature. Current choices are: int8dq, int8wo, int4wo-<group_size>, fp8wo, fp8dq-per_tensor, fp8dq-per_row",
        )
