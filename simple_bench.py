import torch
import torch._dynamo
import time

torch._dynamo.config.automatic_dynamic_shapes = False

def run_bench(drop_guard:bool):
    print(f"\n ------Drop Guard : {drop_guard}--------")
    torch._dynamo.reset()
    
    model = torch.nn.Sequential(torch.nn.Linear(1024, 1024), torch.nn.ReLU()).cuda()
    
    options = {}
    if drop_guard:
        options["guard_filter_fn"] = lambda gs: [getattr(g, "guard_type", None) == "SHAPE_ENV" for g in gs]
    
    compiled_model = torch.compile(model, dynamic=drop_guard, options=options)
    
    for size in [32, 512, 1024]:
        x = torch.randn(1, size, 1024).cuda()
        if drop_guard:
            torch._dynamo.mark_dynamic(x, 1)
        
        torch._dynamo.utils.counters.clear()
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        compiled_model(x)
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        new_compile = torch._dynamo.utils.counters.get("frames", {}).get("total", 0) > 0
        
        print(f"Size : {size:<4} | Time : {(end-start)*1000:7.2f} ms | Recompiled : {new_compile}")


run_bench(drop_guard=False)
run_bench(drop_guard=True)