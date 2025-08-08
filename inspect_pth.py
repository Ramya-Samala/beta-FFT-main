import torch

# Path to your .pth file
pth_path = 'best_results/ACDC_5/unet_best.pth'

# Load the checkpoint
checkpoint = torch.load(pth_path, map_location='cpu')

print("Top-level keys:", checkpoint.keys())

# Try common keys for model weights
if 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
elif 'net' in checkpoint:
    state_dict = checkpoint['net']
elif isinstance(checkpoint, dict):
    state_dict = checkpoint
else:
    state_dict = None

if state_dict:
    print("\nModel parameters:")
    for k, v in state_dict.items():
        print(f"{k}: {v.shape}")
else:
    print("No model state_dict found in this checkpoint.")