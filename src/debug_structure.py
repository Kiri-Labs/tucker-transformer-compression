"""Debug model structure"""
from transformers import AutoModel
import sys

model = AutoModel.from_pretrained("distilgpt2")
print("DistilGPT2 Module Structure:")
print("=" * 60)
for name, module in model.named_modules():
    module_type = type(module).__name__
    if hasattr(module, 'weight'):
        shape = module.weight.shape if module.weight is not None else None
        print(f"{name}: {module_type} {shape}")
    else:
        print(f"{name}: {module_type}")
print("=" * 60)
