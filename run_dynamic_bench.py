import torch
import time
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.server_args import ServerArgs

def run_dynamic_test(drop_mode):
    server_args = ServerArgs(
        model_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        enable_torch_compile=True,
        drop_guard=drop_mode,
        load_format="dummy"
    )
    
    model_config = ModelConfig.from_server_args(server_args)
    model_runner = ModelRunner(
        model_config=model_config,
        mem_fraction_static=server_args.mem_fraction_static,
        gpu_id=0,
        tp_rank=0,
        tp_size=server_args.tp_size,
        moe_ep_rank=0,
        moe_ep_size=server_args.ep_size,
        pp_rank=0,
        pp_size=1,
        nccl_port=3000,
        server_args=server_args,
    )
    
    batch_sizes = [1, 8, 16, 32]
    
    print(f"\n -- Starting Test for Mode : {drop_mode}-------")
    
    for bs in batch_sizes:
        input_ids = torch.ones((bs, 128), dtype=torch.int64, device="cuda")
        
        start_time = time.perf_counter()
        with torch.no_grad():
            model_runner.forward_decode(input_ids)
        end_time = time.perf_counter()
        latency = (end_time-start_time)*1000
        
        print(f"Batch Size: {bs:2d} | Latency: {latency:8.2f} ms")

if __name__=="__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "none"
    run_dynamic_test(mode)