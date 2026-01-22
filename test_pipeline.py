"""
Test script to verify DINO pipeline works correctly (GLOBAL CROPS ONLY).
Run this BEFORE starting full training to catch issues early.
"""

import torch
from PIL import Image
import numpy as np
import sys
import shutil
from pathlib import Path

def test_dataloader():
    """Test that data loading works correctly."""
    print("="*60)
    print("TEST 1: Data Loading")
    print("="*60)
    
    try:
        from dino_dataloader import get_dino_dataloader
        
        # Create a dummy dataset structure for testing
        test_dir = Path("test_data")
        test_dir.mkdir(exist_ok=True)
        
        
        for i in range(10):
            img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            img.save(test_dir / f"image_{i}.jpg")
        
        print(f"✓ Created {len(list(test_dir.glob('*.jpg')))} test images")
        
        # Create dataloader
        train_loader = get_dino_dataloader(
            data_dir=str(test_dir),
            batch_size=4,
            num_workers=0,  # Use 0 for testing to avoid multiprocessing issues
            size=224
        )
        
        print(f"✓ DataLoader created successfully")
        
        # Test loading one batch
        batch = next(iter(train_loader))
        
        print(f"✓ Batch loaded successfully")
        print(f"  - Number of views: {len(batch)}")
        print(f"  - Global view 1 shape: {batch[0].shape}")
        print(f"  - Global view 2 shape: {batch[1].shape}")
        
        # Verify shapes (GLOBAL ONLY - 2 views)
        assert len(batch) == 2, f"Expected 2 views (2 global crops only), got {len(batch)}"
        assert batch[0].shape == (4, 3, 224, 224), f"Wrong global view 1 shape: {batch[0].shape}"
        assert batch[1].shape == (4, 3, 224, 224), f"Wrong global view 2 shape: {batch[1].shape}"
        
        print("✓ All shapes correct!")
        
        # Cleanup
        shutil.rmtree(test_dir)
        
        print("\n✅ TEST 1 PASSED: Data loading works!\n")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 1 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_model():
    """Test that model forward pass works."""
    print("="*60)
    print("TEST 2: Model Forward Pass")
    print("="*60)
    
    try:
        from model import build_dino_model
        
        # Build model
        student, teacher = build_dino_model(arch='vit_small', out_dim=1024)
        
        print(f"✓ Models built successfully")
        print(f"  - Student params: {sum(p.numel() for p in student.parameters())/1e6:.2f}M")
        print(f"  - Teacher params: {sum(p.numel() for p in teacher.parameters())/1e6:.2f}M")
        
        # Create dummy batch (simulating dataloader output - GLOBAL ONLY)
        batch = [
            torch.randn(2, 3, 224, 224),  # Global view 1
            torch.randn(2, 3, 224, 224),  # Global view 2
        ]
        
        print(f"✓ Created dummy batch with 2 global views")
        
        # Forward pass
        with torch.no_grad():
            student_out = student(batch)
            teacher_out = teacher(batch)
        
        print(f"✓ Forward pass successful")
        print(f"  - Student outputs: {len(student_out)} tensors")
        print(f"  - Student output shapes: {[s.shape for s in student_out]}")
        print(f"  - Teacher outputs: {len(teacher_out)} tensors")
        print(f"  - Teacher output shapes: {[t.shape for t in teacher_out]}")
        
        # Verify outputs (GLOBAL ONLY - 2 outputs each)
        assert len(student_out) == 2, f"Expected 2 student outputs, got {len(student_out)}"
        assert len(teacher_out) == 2, f"Expected 2 teacher outputs, got {len(teacher_out)}"
        assert student_out[0].shape == (2, 1024), f"Wrong output shape: {student_out[0].shape}"
        
        print("✓ Output structure correct!")
        
        print("\n✅ TEST 2 PASSED: Model forward pass works!\n")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 2 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_loss():
    """Test that loss computation works."""
    print("="*60)
    print("TEST 3: Loss Computation")
    print("="*60)
    
    try:
        from train import DinoLoss
        
        # Create loss (NO n_local_crops for global-only)
        dino_loss = DinoLoss(
            out_dim=1024,
            warmup_teacher_temp_epochs=10,
        )
        
        print(f"✓ Loss module created")
        
        # Create dummy outputs (GLOBAL ONLY)
        batch_size = 2
        out_dim = 1024
        
        # Student: 2 global views
        student_output = [
            torch.randn(batch_size, out_dim),  # Global view 1
            torch.randn(batch_size, out_dim),  # Global view 2
        ]
        
        # Teacher: 2 global views
        teacher_output = [
            torch.randn(batch_size, out_dim),  # Global view 1
            torch.randn(batch_size, out_dim),  # Global view 2
        ]
        
        print(f"✓ Created dummy outputs")
        
        # Compute loss
        loss = dino_loss(student_output, teacher_output, epoch=5)
        
        print(f"✓ Loss computed successfully")
        print(f"  - Loss value: {loss.item():.4f}")
        print(f"  - Loss is finite: {torch.isfinite(loss).item()}")
        
        # Verify loss
        assert torch.isfinite(loss), "Loss is not finite!"
        assert loss.item() > 0, "Loss should be positive!"
        
        print("✓ Loss value valid!")
        
        print("\n✅ TEST 3 PASSED: Loss computation works!\n")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 3 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """Test full integration with real data flow."""
    print("="*60)
    print("TEST 4: Full Integration")
    print("="*60)
    
    try:
        from dino_dataloader import get_dino_dataloader
        from model import build_dino_model
        from train import DinoLoss
        
        # Create test data
        test_dir = Path("test_data")
        test_dir.mkdir(exist_ok=True)
        
        from PIL import Image
        import numpy as np
        for i in range(10):
            img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            img.save(test_dir / f"image_{i}.jpg")
        
        print(f"✓ Created test dataset")
        
        # Create dataloader
        train_loader = get_dino_dataloader(
            data_dir=str(test_dir),
            batch_size=2,
            num_workers=0,
        )
        
        # Create model
        student, teacher = build_dino_model( out_dim=1024)
        
        # Create loss (NO n_local_crops)
        dino_loss = DinoLoss(out_dim=1024)
        
        # Create optimizer
        optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)
        
        print(f"✓ Created all components")
        
        # Run one training step
        batch = next(iter(train_loader))
        
        # Forward
        student_output = student(batch)
        with torch.no_grad():
            teacher_output = teacher(batch)
        
        # Compute loss
        loss = dino_loss(student_output, teacher_output, epoch=0)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"✓ Training step completed")
        print(f"  - Loss: {loss.item():.4f}")

        shutil.rmtree(test_dir)
        
        print("\n✅ TEST 4 PASSED: Full integration works!\n")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 4 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("DINO PIPELINE TESTING (GLOBAL CROPS ONLY)")
    print("="*60 + "\n")
    
    results = []
    
    # Run tests
    results.append(("Data Loading", test_dataloader()))
    results.append(("Model Forward", test_model()))
    results.append(("Loss Computation", test_loss()))
    results.append(("Full Integration", test_integration()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:.<40} {status}")
    
    all_passed = all(r[1] for r in results)
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! Pipeline is ready for training.")
        print("\nYou can now run: python train.py")
    else:
        print("\n⚠️  SOME TESTS FAILED! Please fix issues before training.")
    
    print("="*60 + "\n")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)