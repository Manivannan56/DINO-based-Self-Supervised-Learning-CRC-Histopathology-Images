import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path
import random


class HistologyDataset(Dataset):
    """
    Dataset for loading histology images for DINO self-supervised learning.
    No labels needed - just loads images.
    """
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir: Path to dataset (e.g., 'NCT-CRC-HE-100K')
            transform: DINO multi-crop transforms
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # Collect all image paths (supports nested folders)
        self.image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']:
            self.image_paths.extend(list(self.root_dir.rglob(ext)))
        
        print(f"Found {len(self.image_paths)} images in {root_dir}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            # DINO returns multiple crops of same image
            crops = self.transform(image)
            return crops
        
        return image


class DINOMultiCropAugmentation:
    """
    DINO's FULL multi-crop data augmentation.
    Creates 2 global views (224x224) + 8 local views (96x96).
    This is the complete DINO strategy used in papers.
    """
    def __init__(
        self,
        global_crops_scale=(0.4, 1.0),
        local_crops_scale=(0.05, 0.4),
        global_crops_number=2,
        local_crops_number=8,
        size=224,
        local_size=96,
    ):
        """
        Args:
            global_crops_scale: Scale range for global crops (0.4-1.0 of original)
            local_crops_scale: Scale range for local crops (0.05-0.4 of original)
            global_crops_number: Number of global crops (default: 2)
            local_crops_number: Number of local crops (default: 8)
            size: Global crop size (default: 224)
            local_size: Local crop size (default: 96)
        """
        
        # Color jittering for histology (adjusted for H&E staining)
        color_jitter = transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.1,
            hue=0.05
        )
        
        # Pathology-specific normalization (Lunit statistics)
        normalize = transforms.Normalize(
            mean=[0.70322989, 0.53606487, 0.66096631],
            std=[0.21716536, 0.26081574, 0.20723464]
        )
        
        # GLOBAL CROP 1 - Standard augmentation
        self.global_1 = transforms.Compose([
            transforms.RandomResizedCrop(
                size, 
                scale=global_crops_scale,
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.5),
            transforms.ToTensor(),
            normalize,
        ])
        
        # GLOBAL CROP 2 - With additional solarization
        self.global_2 = transforms.Compose([
            transforms.RandomResizedCrop(
                size,
                scale=global_crops_scale,
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.1),
            transforms.RandomSolarize(threshold=128, p=0.2),
            transforms.ToTensor(),
            normalize,
        ])
        
        # LOCAL CROPS - Smaller crops to capture cell-level features
        # In pathology, 96x96 crops at 20x magnification capture individual cells
        self.local = transforms.Compose([
            transforms.RandomResizedCrop(
                local_size,
                scale=local_crops_scale,
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.5),
            transforms.ToTensor(),
            normalize,
        ])
        
        self.global_crops_number = global_crops_number
        self.local_crops_number = local_crops_number
        
        print(f"Multi-crop strategy initialized:")
        print(f"  Global crops: {global_crops_number} x {size}x{size} (scale {global_crops_scale})")
        print(f"  Local crops: {local_crops_number} x {local_size}x{local_size} (scale {local_crops_scale})")
        print(f"  Total views per image: {global_crops_number + local_crops_number}")
    
    def __call__(self, image):
        """
        Returns list of crops: [global1, global2, local1, ..., local8]
        
        Teacher will only see global crops.
        Student will see all crops.
        """
        crops = []
        
        # Generate global crops
        crops.append(self.global_1(image))
        crops.append(self.global_2(image))
        
        # Generate local crops
        for _ in range(self.local_crops_number):
            crops.append(self.local(image))
        
        return crops  # Returns list of 10 tensors (2 global + 8 local)


def collate_fn(batch):
    """
    Custom collate function for DINO multi-crop.
    
    batch: List of lists, where each inner list contains [global1, global2, local1, ..., local8]
    
    Returns:
        List of batched tensors: [global1_batch, global2_batch, local1_batch, ..., local8_batch]
    """
    n_crops = len(batch[0])  # Should be 10 (2 global + 8 local)
    
    # Transpose and stack
    batched_crops = []
    for i in range(n_crops):
        crop_batch = torch.stack([sample[i] for sample in batch])
        batched_crops.append(crop_batch)
    
    return batched_crops


def get_dino_dataloader(
    data_dir,
    batch_size=64,
    num_workers=4,
    size=224,
    local_size=96,
    use_multicrop=True
):
    """
    Create DataLoader for DINO training.
    
    Args:
        data_dir: Path to image directory
        batch_size: Batch size (typically 64-256)
        num_workers: Number of data loading workers
        size: Global crop size
        local_size: Local crop size
        use_multicrop: If True, uses 2 global + 8 local crops. If False, only 2 global crops.
    
    Returns:
        DataLoader
    """
    if use_multicrop:
        print("Using FULL multi-crop strategy (2 global + 8 local)")
        transform = DINOMultiCropAugmentation(
            size=size,
            local_size=local_size,
            global_crops_number=2,
            local_crops_number=8
        )
    else:
        print("Using global crops only (2 views)")
        from dino_dataloader import DINODataAugmentation as GlobalOnly
        transform = GlobalOnly(size=size)
    
    dataset = HistologyDataset(data_dir, transform=transform)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    
    return dataloader


# Example usage and testing
if __name__ == "__main__":
    
    # Test multi-crop dataloader
    print("="*60)
    print("Testing Multi-Crop DataLoader")
    print("="*60)
    
    train_loader = get_dino_dataloader(
        data_dir="/home/senthilkumar.m/Dino_Pathology/data/NCT-CRC-HE-100K/dataset",
        batch_size=64,
        num_workers=4,
        use_multicrop=True  # Set False for global-only
    )
    
    # Test loading one batch
    print("\nLoading test batch...")
    for batch in train_loader:
        print(f"\n✓ Batch loaded successfully!")
        print(f"  Total views: {len(batch)}")
        print(f"  Global crop 1: {batch[0].shape}")  # [64, 3, 224, 224]
        print(f"  Global crop 2: {batch[1].shape}")  # [64, 3, 224, 224]
        print(f"  Local crop 1: {batch[2].shape}")   # [64, 3, 96, 96]
        print(f"  Local crop 8: {batch[9].shape}")   # [64, 3, 96, 96]
        
        # Memory estimate
        total_tensors = sum(b.numel() for b in batch)
        memory_mb = total_tensors * 4 / 1024 / 1024  # 4 bytes per float32
        print(f"\n  Estimated batch memory: {memory_mb:.1f} MB")
        
        break
    
    print("\n" + "="*60)
    print("Multi-crop dataloader ready for training!")
    print("="*60)