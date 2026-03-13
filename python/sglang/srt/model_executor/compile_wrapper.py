import torch

class CompileWrapper:
    def __init__(self, model: torch.nn.Module, drop_guard: str = "none"):
        self.model = model
        self.drop_guard =  drop_guard
        print(f"########################compilewrapper############################# {self.drop_guard} ######")
    
        def _guard_filter(guards):
            if self.drop_guard == "none":
                return guards
            
            if self.drop_guard == "all":
                return [False for _ in guards]

            # handle guard drops for dynamic shapes with SHAPE_ENV
            if self.drop_guard == "shape":
                return [getattr(g, "guard_type", None) == "SHAPE_ENV" for g in guards]
            return guards
    
        self.compiled_model = torch.compile(
            model,
            dynamic=True,
            options={"guard_filter_fn": _guard_filter}
        )
    
    def forward(self, *args, **kwargs):
        return self.compiled_model(*args, **kwargs)
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

