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
"""Model and tokenizer argument group."""

from __future__ import annotations

import argparse
import dataclasses
from typing import Optional


@dataclasses.dataclass
class ModelArgs:
    """Arguments for model and tokenizer configuration."""

    # Model and tokenizer
    model_path: str = ""
    tokenizer_path: Optional[str] = None
    tokenizer_mode: str = "auto"
    tokenizer_worker_num: int = 1
    skip_tokenizer_init: bool = False
    load_format: str = "auto"
    model_loader_extra_config: str = "{}"
    trust_remote_code: bool = False
    context_length: Optional[int] = None
    is_embedding: bool = False
    enable_multimodal: Optional[bool] = None
    revision: Optional[str] = None
    model_impl: str = "auto"

    # Model download and loading (from runtime section)
    download_dir: Optional[str] = None
    model_checksum: Optional[str] = None

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        # Avoid circular import at module level
        from sglang.srt.server_args import LOAD_FORMAT_CHOICES

        group = parser.add_argument_group("Model and tokenizer")
        group.add_argument(
            "--model-path",
            "--model",
            type=str,
            help="The path of the model weights. This can be a local folder or a Hugging Face repo ID.",
            required=True,
        )
        group.add_argument(
            "--tokenizer-path",
            type=str,
            default=ModelArgs.tokenizer_path,
            help="The path of the tokenizer.",
        )
        group.add_argument(
            "--tokenizer-mode",
            type=str,
            default=ModelArgs.tokenizer_mode,
            choices=["auto", "slow"],
            help="Tokenizer mode. 'auto' will use the fast "
            "tokenizer if available, and 'slow' will "
            "always use the slow tokenizer.",
        )
        group.add_argument(
            "--tokenizer-worker-num",
            type=int,
            default=ModelArgs.tokenizer_worker_num,
            help="The worker num of the tokenizer manager.",
        )
        group.add_argument(
            "--skip-tokenizer-init",
            action="store_true",
            help="If set, skip init tokenizer and pass input_ids in generate request.",
        )
        group.add_argument(
            "--load-format",
            type=str,
            default=ModelArgs.load_format,
            choices=LOAD_FORMAT_CHOICES,
            help="The format of the model weights to load. "
            '"auto" will try to load the weights in the safetensors format '
            "and fall back to the pytorch bin format if safetensors format "
            "is not available. "
            '"pt" will load the weights in the pytorch bin format. '
            '"safetensors" will load the weights in the safetensors format. '
            '"npcache" will load the weights in pytorch format and store '
            "a numpy cache to speed up the loading. "
            '"dummy" will initialize the weights with random values, '
            "which is mainly for profiling."
            '"gguf" will load the weights in the gguf format. '
            '"bitsandbytes" will load the weights using bitsandbytes '
            "quantization."
            '"layered" loads weights layer by layer so that one can quantize a '
            "layer before loading another to make the peak memory envelope "
            "smaller.",
        )
        group.add_argument(
            "--model-loader-extra-config",
            type=str,
            help="Extra config for model loader. "
            "This will be passed to the model loader corresponding to the chosen load_format.",
            default=ModelArgs.model_loader_extra_config,
        )
        group.add_argument(
            "--trust-remote-code",
            action="store_true",
            help="Whether or not to allow for custom models defined on the Hub in their own modeling files.",
        )
        group.add_argument(
            "--context-length",
            type=int,
            default=ModelArgs.context_length,
            help="The model's maximum context length. Defaults to None (will use the value from the model's config.json instead).",
        )
        group.add_argument(
            "--is-embedding",
            action="store_true",
            help="Whether to use a CausalLM as an embedding model.",
        )
        group.add_argument(
            "--enable-multimodal",
            default=ModelArgs.enable_multimodal,
            action="store_true",
            help="Enable the multimodal functionality for the served model. If the model being served is not multimodal, nothing will happen",
        )
        group.add_argument(
            "--revision",
            type=str,
            default=None,
            help="The specific model version to use. It can be a branch "
            "name, a tag name, or a commit id. If unspecified, will use "
            "the default version.",
        )
        group.add_argument(
            "--model-impl",
            type=str,
            default=ModelArgs.model_impl,
            help="Which implementation of the model to use.\n\n"
            '- "auto" will try to use the SGLang implementation if available, and fall back to the transformers implementation.\n'
            '- "transformers" will use the transformers implementation, supporting all features (e.g. LoRA, quantization) '
            "of transformers models.\n",
        )
        group.add_argument(
            "--download-dir",
            type=str,
            default=ModelArgs.download_dir,
            help="Model download directory for huggingface.",
        )
        group.add_argument(
            "--model-checksum",
            type=str,
            nargs="?",
            const="",
            default=None,
            help="Model file integrity verification. If provided without value, uses model-path as HF repo ID. Otherwise, provide checksums JSON file path or HuggingFace repo ID.",
        )
