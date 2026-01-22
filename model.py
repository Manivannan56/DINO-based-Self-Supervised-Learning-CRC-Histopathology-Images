import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import os

def build_dino_model(
        arch="vit_small",
        out_dim=16384,
        pretrained=True,
        lunit_weights_path="/home/senthilkumar.m/Dino_Pathology/lunit_vit_small_dino.pth"
):
    """
    Build DINO models with Lunit pathology pretrained initialization (loaded from local file)
    """
    
    class DinoHead(nn.Module):
        def __init__(self, in_dim, out_dim):
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, 2048),
                nn.GELU(),
                nn.Linear(2048, 256),
            )
            self.last_layer = nn.utils.weight_norm(
                nn.Linear(256, out_dim, bias=False)
            )
            self.last_layer.weight_g.data.fill_(1.0)
            self.last_layer.weight_g.requires_grad = False
        
        def forward(self, x):
            x = self.mlp(x)
            x = F.normalize(x, dim=-1)
            x = self.last_layer(x)
            return x
    

    class MultiCropViT(nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone = backbone
            self.head = head
        
        def forward(self, crops):
            outputs = []
            for crop in crops:
                features = self.backbone(crop)
                out = self.head(features)
                outputs.append(out)
            return outputs
    
    # Create empty ViT-Small backbones
    if arch == "vit_small":
        student_backbone = timm.create_model(
            "vit_small_patch16_224",
            pretrained=False,  # Don't download ImageNet weights
            num_classes=0
        )
        teacher_backbone = timm.create_model(
            "vit_small_patch16_224",
            pretrained=False,
            num_classes=0
        )
    elif arch == "vit_base":
        print("⚠ Warning: Lunit only provides ViT-Small, using ViT-Small instead")
        student_backbone = timm.create_model(
            "vit_small_patch16_224",
            pretrained=False,
            num_classes=0
        )
        teacher_backbone = timm.create_model(
            "vit_small_patch16_224",
            pretrained=False,
            num_classes=0
        )
    
    # Load Lunit pretrained weights from local file
    if pretrained and os.path.exists(lunit_weights_path):
        print("="*60)
        print("Loading Lunit DINO Pretrained Weights (Local File)")
        print("="*60)
        print(f"Path: {lunit_weights_path}")
        print(f"Source: TCGA pathology dataset (19M patches)")
        print(f"Training: 200 epochs, DINO SSL with multi-crop")
        
        lunit_state_dict = torch.load(lunit_weights_path, map_location='cpu')
        
        # Load into student
        missing, unexpected = student_backbone.load_state_dict(lunit_state_dict, strict=False)
        print(f"✓ Student backbone loaded")
        if missing:
            print(f"  Missing keys: {len(missing)} (projection head - OK)")
        
        # Load into teacher
        teacher_backbone.load_state_dict(lunit_state_dict, strict=False)
        print(f"✓ Teacher backbone loaded")
        
    elif pretrained:
        print(f"⚠ WARNING: Lunit weights not found at {lunit_weights_path}")
        print(f"  Using RANDOM initialization instead!")
        print(f"  Download weights first or training will be unstable!")
    else:
        print("Using random initialization (pretrained=False)")

    embed_dim = student_backbone.num_features  # 384 for ViT-Small
    
    # Build student with random projection head
    student = MultiCropViT(
        student_backbone,
        DinoHead(embed_dim, out_dim)
    )

    # Build teacher with same structure
    teacher = MultiCropViT(
        teacher_backbone,
        DinoHead(embed_dim, out_dim)
    )

    # Initialize teacher with student weights
    teacher.load_state_dict(student.state_dict())
    
    # Freeze teacher
    for p in teacher.parameters():
        p.requires_grad = False
    
    print("\n" + "="*60)
    print("DINO Models Initialized")
    print("="*60)
    print(f"Backbone: {'Lunit pretrained (19M pathology)' if (pretrained and os.path.exists(lunit_weights_path)) else 'Random'}")
    print(f"Projection head: Random init (adapts to CRC-100K)")
    print(f"Embed dim: {embed_dim}")
    print(f"Output dim: {out_dim}")
    print(f"Strategy: Continued SSL - Domain adaptation to CRC")
    print("="*60 + "\n")
    
    return student, teacher