"""
Comprehensive analysis of histology dataset and DINO augmentations.
Run this BEFORE training to understand your data.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def analyze_dataset_structure(data_dir):
    """Analyze the structure and distribution of the dataset."""
    print("="*70)
    print("DATASET STRUCTURE ANALYSIS")
    print("="*70)
    
    data_path = Path(data_dir)
    
    # Count images per class
    class_counts = {}
    all_images = []
    
    for class_dir in sorted(data_path.iterdir()):
        if class_dir.is_dir():
            images = list(class_dir.glob('*.tif')) + list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpg'))
            class_counts[class_dir.name] = len(images)
            all_images.extend(images)
    
    print(f"\n✓ Found {len(class_counts)} classes")
    print(f"✓ Total images: {len(all_images)}")
    print(f"\nClass distribution:")
    print("-" * 40)
    for cls, count in sorted(class_counts.items()):
        bar = "█" * int(count / 100)
        print(f"{cls:10s} | {count:6d} images | {bar}")
    
    # Check for imbalance
    counts = list(class_counts.values())
    max_count, min_count = max(counts), min(counts)
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    print(f"\n{'Imbalance ratio:':<20} {imbalance_ratio:.2f}x")
    if imbalance_ratio > 2:
        print("⚠️  Warning: Significant class imbalance detected")
    else:
        print("✓ Classes are reasonably balanced")
    
    return all_images, class_counts


def analyze_image_properties(image_paths, n_samples=1000):
    """Analyze basic image properties."""
    print("\n" + "="*70)
    print("IMAGE PROPERTIES ANALYSIS")
    print("="*70)
    
    # Sample images
    sample_paths = np.random.choice(image_paths, min(n_samples, len(image_paths)), replace=False)
    
    sizes = []
    aspect_ratios = []
    mean_colors = []
    std_colors = []
    
    print(f"\nAnalyzing {len(sample_paths)} random images...")
    
    for img_path in tqdm(sample_paths, desc="Processing"):
        try:
            img = Image.open(img_path).convert('RGB')
            w, h = img.size
            sizes.append((w, h))
            aspect_ratios.append(w / h)
            
            # Color statistics
            img_array = np.array(img) / 255.0
            mean_colors.append(img_array.mean(axis=(0, 1)))
            std_colors.append(img_array.std(axis=(0, 1)))
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
    
    # Statistics
    sizes = np.array(sizes)
    aspect_ratios = np.array(aspect_ratios)
    mean_colors = np.array(mean_colors)
    std_colors = np.array(std_colors)
    
    print(f"\n{'Image dimensions:':<25}")
    unique_sizes = np.unique(sizes, axis=0)
    for size in unique_sizes[:10]:  # Show first 10 unique sizes
        count = np.sum((sizes == size).all(axis=1))
        print(f"  {size[0]}x{size[1]}: {count} images")
    
    print(f"\n{'Aspect ratio:':<25} {aspect_ratios.mean():.3f} ± {aspect_ratios.std():.3f}")
    
    print(f"\n{'Mean pixel values (RGB):':<25}")
    print(f"  R: {mean_colors[:, 0].mean():.3f} ± {mean_colors[:, 0].std():.3f}")
    print(f"  G: {mean_colors[:, 1].mean():.3f} ± {mean_colors[:, 1].std():.3f}")
    print(f"  B: {mean_colors[:, 2].mean():.3f} ± {mean_colors[:, 2].std():.3f}")
    
    print(f"\n{'Std pixel values (RGB):':<25}")
    print(f"  R: {std_colors[:, 0].mean():.3f} ± {std_colors[:, 0].std():.3f}")
    print(f"  G: {std_colors[:, 1].mean():.3f} ± {std_colors[:, 1].std():.3f}")
    print(f"  B: {std_colors[:, 2].mean():.3f} ± {std_colors[:, 2].std():.3f}")
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Color distribution
    axes[0].hist(mean_colors[:, 0], bins=50, alpha=0.5, label='Red', color='red')
    axes[0].hist(mean_colors[:, 1], bins=50, alpha=0.5, label='Green', color='green')
    axes[0].hist(mean_colors[:, 2], bins=50, alpha=0.5, label='Blue', color='blue')
    axes[0].set_xlabel('Mean Pixel Value')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Color Distribution')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Aspect ratios
    axes[1].hist(aspect_ratios, bins=50, color='purple', alpha=0.7)
    axes[1].set_xlabel('Aspect Ratio (W/H)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Aspect Ratio Distribution')
    axes[1].grid(alpha=0.3)
    
    # Image sizes
    axes[2].scatter(sizes[:, 0], sizes[:, 1], alpha=0.3, s=10)
    axes[2].set_xlabel('Width (pixels)')
    axes[2].set_ylabel('Height (pixels)')
    axes[2].set_title('Image Dimensions')
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dataset_statistics.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved visualization: dataset_statistics.png")
    plt.close()
    
    return mean_colors, std_colors


def visualize_sample_images(image_paths, class_counts, n_per_class=3):
    """Visualize sample images from each class."""
    print("\n" + "="*70)
    print("SAMPLE IMAGES VISUALIZATION")
    print("="*70)
    
    data_dir = Path(image_paths[0]).parent.parent
    classes = sorted(list(class_counts.keys()))
    
    n_classes = len(classes)
    fig, axes = plt.subplots(n_classes, n_per_class, figsize=(n_per_class * 3, n_classes * 3))
    
    if n_classes == 1:
        axes = axes.reshape(1, -1)
    
    for i, cls in enumerate(classes):
        class_dir = data_dir / cls
        class_images = list(class_dir.glob('*.tif'))[:n_per_class]
        
        for j, img_path in enumerate(class_images):
            img = Image.open(img_path).convert('RGB')
            axes[i, j].imshow(img)
            axes[i, j].axis('off')
            if j == 0:
                axes[i, j].set_title(f'{cls} (n={class_counts[cls]})', fontsize=10, fontweight='bold')
            else:
                axes[i, j].set_title('')
    
    plt.suptitle('Sample Images per Class', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('sample_images_per_class.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved visualization: sample_images_per_class.png")
    plt.close()


def visualize_augmentations(image_path, n_augmentations=8):
    """Visualize DINO augmentations on a single image."""
    print("\n" + "="*70)
    print("AUGMENTATION VISUALIZATION")
    print("="*70)
    
    from dino_dataloader import DINODataAugmentation
    
    # Load original image
    original_img = Image.open(image_path).convert('RGB')
    
    # Create augmentation
    aug = DINODataAugmentation(size=224)
    
    # Generate multiple augmented views
    views = []
    for _ in range(n_augmentations):
        augmented = aug(original_img)  # Returns [view1, view2]
        views.extend(augmented)
    
    # Denormalize for visualization
    def denormalize(tensor):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = tensor * std + mean
        tensor = torch.clamp(tensor, 0, 1)
        return tensor.permute(1, 2, 0).numpy()
    
    # Visualize
    n_rows = (n_augmentations + 1) // 2
    fig, axes = plt.subplots(n_rows + 1, 4, figsize=(16, 4 * (n_rows + 1)))
    
    # Original image
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title('Original Image', fontweight='bold', fontsize=12)
    axes[0, 0].axis('off')
    for j in range(1, 4):
        axes[0, j].axis('off')
    
    # Augmented views
    for idx, view in enumerate(views[:n_augmentations * 2]):
        row = (idx // 4) + 1
        col = idx % 4
        
        img_denorm = denormalize(view)
        axes[row, col].imshow(img_denorm)
        axes[row, col].set_title(f'Augmented View {idx + 1}', fontsize=10)
        axes[row, col].axis('off')
    
    # Hide unused subplots
    for idx in range(len(views), n_rows * 4):
        row = (idx // 4) + 1
        col = idx % 4
        if row < len(axes):
            axes[row, col].axis('off')
    
    plt.suptitle('DINO Augmentations (Global Crops)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('augmentation_examples.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved visualization: augmentation_examples.png")
    plt.close()


def compare_augmentation_strength(image_path):
    """Compare different augmentation strengths."""
    print("\n" + "="*70)
    print("AUGMENTATION STRENGTH COMPARISON")
    print("="*70)
    
    from torchvision import transforms
    from PIL import Image
    
    original_img = Image.open(image_path).convert('RGB')
    
    # Different augmentation strategies
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    augmentations = {
        'Minimal': transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            normalize,
        ]),
        'Light': transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.2, 0.2, 0.1, 0.05),
            transforms.ToTensor(),
            normalize,
        ]),
        'DINO (Ours)': transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.4, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            normalize,
        ]),
        'Heavy': transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(0.6, 0.6, 0.3, 0.2),
            transforms.RandomGrayscale(p=0.3),
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            transforms.RandomSolarize(threshold=128, p=0.3),
            transforms.ToTensor(),
            normalize,
        ]),
    }
    
    def denormalize(tensor):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = tensor * std + mean
        tensor = torch.clamp(tensor, 0, 1)
        return tensor.permute(1, 2, 0).numpy()
    
    fig, axes = plt.subplots(len(augmentations), 4, figsize=(16, 4 * len(augmentations)))
    
    for row, (name, aug) in enumerate(augmentations.items()):
        for col in range(4):
            augmented = aug(original_img)
            img_denorm = denormalize(augmented)
            axes[row, col].imshow(img_denorm)
            if col == 0:
                axes[row, col].set_ylabel(name, fontsize=12, fontweight='bold', rotation=0, 
                                          ha='right', va='center', labelpad=40)
            axes[row, col].axis('off')
    
    plt.suptitle('Augmentation Strength Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('augmentation_strength_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved visualization: augmentation_strength_comparison.png")
    plt.close()


def check_data_loading_speed(data_dir, batch_size=64, num_workers_list=[0, 2, 4, 8]):
    """Benchmark data loading speed with different num_workers."""
    print("\n" + "="*70)
    print("DATA LOADING SPEED BENCHMARK")
    print("="*70)
    
    from dino_dataloader import get_dino_dataloader
    import time
    
    results = {}
    
    for num_workers in num_workers_list:
        print(f"\nTesting with num_workers={num_workers}...")
        
        try:
            loader = get_dino_dataloader(
                data_dir=data_dir,
                batch_size=batch_size,
                num_workers=num_workers,
            )
            
            # Time loading first 10 batches
            start_time = time.time()
            for i, batch in enumerate(loader):
                if i >= 10:
                    break
            elapsed = time.time() - start_time
            
            batches_per_sec = 10 / elapsed
            images_per_sec = batches_per_sec * batch_size
            
            results[num_workers] = images_per_sec
            print(f"  ✓ {images_per_sec:.1f} images/sec ({batches_per_sec:.2f} batches/sec)")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results[num_workers] = 0
    
    # Visualize
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(results)), list(results.values()), color='steelblue', alpha=0.8)
    plt.xticks(range(len(results)), [f'{k} workers' for k in results.keys()])
    plt.ylabel('Images/second')
    plt.title('Data Loading Speed vs num_workers')
    plt.grid(axis='y', alpha=0.3)
    
    # Add values on bars
    for i, (k, v) in enumerate(results.items()):
        plt.text(i, v + 5, f'{v:.0f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('data_loading_benchmark.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved visualization: data_loading_benchmark.png")
    plt.close()
    
    # Recommendation
    best_workers = max(results, key=results.get)
    print(f"\n💡 Recommendation: Use num_workers={best_workers} for best performance")


def main():
    """Run complete dataset analysis."""
    
    # Configuration
    DATA_DIR = "data/NCT-CRC-HE-100K"
    
    print("\n" + "="*70)
    print("HISTOLOGY DATASET & AUGMENTATION ANALYSIS")
    print("="*70)
    print(f"\nDataset directory: {DATA_DIR}\n")
    
    # 1. Dataset structure
    image_paths, class_counts = analyze_dataset_structure(DATA_DIR)
    
    # 2. Image properties
    mean_colors, std_colors = analyze_image_properties(image_paths, n_samples=1000)
    
    # 3. Sample images per class
    visualize_sample_images(image_paths, class_counts, n_per_class=4)
    
    # 4. Augmentation visualization
    sample_image = np.random.choice(image_paths)
    visualize_augmentations(sample_image, n_augmentations=8)
    
    # 5. Augmentation strength comparison
    compare_augmentation_strength(sample_image)
    
    # 6. Data loading benchmark
    check_data_loading_speed(DATA_DIR, batch_size=64, num_workers_list=[0, 2, 4, 8])
    
    # Summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  1. dataset_statistics.png")
    print("  2. sample_images_per_class.png")
    print("  3. augmentation_examples.png")
    print("  4. augmentation_strength_comparison.png")
    print("  5. data_loading_benchmark.png")
    print("\n💡 Review these visualizations before starting training!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()