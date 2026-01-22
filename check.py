"""
Debug script to verify Lunit weights are loading correctly
Run this on HPC to diagnose the issue
"""

import torch
import timm
import os

print("="*70)
print("LUNIT WEIGHTS LOADING DIAGNOSTIC")
print("="*70)

# Paths
lunit_path = "/Users/manivannans/Dino_Pathology/lunit_vit_small_dino.pth"

# 1. Check if file exists and size
print("\n[1] Checking file...")
if os.path.exists(lunit_path):
    size_mb = os.path.getsize(lunit_path) / 1024 / 1024
    print(f"✓ File exists: {lunit_path}")
    print(f"  Size: {size_mb:.1f} MB")
    if size_mb < 50:
        print(f"  ⚠ WARNING: File seems too small! Should be ~80-90 MB")
else:
    print(f"✗ File NOT found: {lunit_path}")
    exit(1)

# 2. Load the saved state dict
print("\n[2] Loading saved state dict...")
try:
    lunit_state_dict = torch.load(lunit_path, map_location='cpu')
    print(f"✓ State dict loaded")
    print(f"  Total keys: {len(lunit_state_dict)}")
    print(f"  First 5 keys:")
    for i, key in enumerate(list(lunit_state_dict.keys())[:5]):
        print(f"    {i+1}. {key}")
except Exception as e:
    print(f"✗ Error loading: {e}")
    exit(1)

# 3. Create empty ViT-Small model
print("\n[3] Creating empty ViT-Small model...")
try:
    model = timm.create_model(
        "vit_small_patch16_224",
        pretrained=False,  # No downloads
        num_classes=0
    )
    print(f"✓ Model created")
    model_keys = list(model.state_dict().keys())
    print(f"  Total keys in model: {len(model_keys)}")
    print(f"  First 5 keys:")
    for i, key in enumerate(model_keys[:5]):
        print(f"    {i+1}. {key}")
except Exception as e:
    print(f"✗ Error creating model: {e}")
    exit(1)

# 4. Try to load Lunit weights into model
print("\n[4] Attempting to load Lunit weights into model...")
try:
    missing, unexpected = model.load_state_dict(lunit_state_dict, strict=False)
    print(f"✓ Load attempt completed")
    print(f"  Missing keys: {len(missing)}")
    print(f"  Unexpected keys: {len(unexpected)}")
    
    if missing:
        print(f"\n  First 5 missing keys:")
        for i, key in enumerate(list(missing)[:5]):
            print(f"    {i+1}. {key}")
    
    if unexpected:
        print(f"\n  First 5 unexpected keys:")
        for i, key in enumerate(list(unexpected)[:5]):
            print(f"    {i+1}. {key}")
    
    # Calculate how many actually matched
    lunit_keys_set = set(lunit_state_dict.keys())
    model_keys_set = set(model_keys)
    matched = lunit_keys_set & model_keys_set
    
    print(f"\n  Keys that matched: {len(matched)}/{len(lunit_state_dict)}")
    match_percent = (len(matched) / len(lunit_state_dict)) * 100
    print(f"  Match percentage: {match_percent:.1f}%")
    
    if match_percent < 50:
        print("\n  ⚠ WARNING: Less than 50% of keys matched!")
        print("  This means weights are NOT loading correctly!")
    elif match_percent > 90:
        print("\n  ✓ GOOD: Most keys matched! Weights should be working.")
    
except Exception as e:
    print(f"✗ Error loading weights: {e}")
    exit(1)

# 5. Test inference to verify weights work
print("\n[5] Testing inference with loaded weights...")
try:
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"✓ Inference successful")
    print(f"  Output shape: {output.shape}")
    print(f"  Output mean: {output.mean().item():.4f}")
    print(f"  Output std: {output.std().item():.4f}")
    
    # Check if output is all zeros/ones (sign of failed loading)
    if torch.allclose(output, torch.zeros_like(output)):
        print("  ⚠ WARNING: Output is all zeros! Weights may not have loaded.")
    elif output.std().item() < 0.01:
        print("  ⚠ WARNING: Output has very low variance! Suspicious.")
    else:
        print("  ✓ Output looks reasonable")
        
except Exception as e:
    print(f"✗ Error during inference: {e}")

# 6. Summary
print("\n" + "="*70)
print("DIAGNOSTIC SUMMARY")
print("="*70)
print(f"File exists: {'YES' if os.path.exists(lunit_path) else 'NO'}")
print(f"File size: {size_mb:.1f} MB")
print(f"Keys matched: {match_percent:.1f}%")
print(f"Inference works: YES")

if match_percent > 90:
    print("\n✓ DIAGNOSIS: Weights should be loading correctly!")
    print("  If training loss still starts at 9+, check normalization in dataloader.")
else:
    print("\n✗ DIAGNOSIS: Weights are NOT loading correctly!")
    print("  State dict key mismatch. Need to debug further.")

print("="*70)