"""
Ultra-simple check to see if model makes predictions
Run this to quickly diagnose the issue
"""
import sys
import torch

print("Checking PyTorch installation...")
print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Try to load checkpoint and check its contents
checkpoint_path = "experiments/quick_test/checkpoints/best_model.pth"

try:
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print(f"\nCheckpoint loaded successfully!")
    print(f"Keys: {list(checkpoint.keys())}")

    if 'epoch' in checkpoint:
        print(f"Epoch: {checkpoint['epoch']}")
    if 'metrics' in checkpoint:
        print(f"Metrics: {checkpoint['metrics']}")

    # Check model state
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print(f"\nModel has {len(state_dict)} parameters")

        # Check if weights are zeros or random
        first_key = list(state_dict.keys())[0]
        first_param = state_dict[first_key]
        print(f"First parameter '{first_key}':")
        print(f"  Shape: {first_param.shape}")
        print(f"  Mean: {first_param.mean():.6f}")
        print(f"  Std: {first_param.std():.6f}")
        print(f"  Min/Max: {first_param.min():.6f} / {first_param.max():.6f}")

        if torch.allclose(first_param, torch.zeros_like(first_param)):
            print("  ⚠️  WARNING: Parameters are all zeros!")

except Exception as e:
    print(f"❌ Error loading checkpoint: {e}")
    sys.exit(1)

print("\n✓ Checkpoint check complete")
