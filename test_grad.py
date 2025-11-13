
import torch
from models.fire_vit import build_fire_vit
import yaml

with open('configs/base_config.yaml') as f:
    config = yaml.safe_load(f)

model = build_fire_vit(config)
print('Model initialized successfully!')

# Check initial bbox predictions
dummy_input = torch.randn(1, 3, 640, 640)
with torch.no_grad():
    preds = model(dummy_input)
    bbox = preds[0]['bbox_pred']
    print(f'Initial bbox predictions: min={bbox.min():.4f}, max={bbox.max():.4f}, mean={bbox.mean():.4f}')
    print(f'Expected: small values around 0.1-10')