import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import timm
import math
from tqdm import tqdm
import os



class DinoLoss(nn.Module):

    def __init__(
            self,
            out_dim,
            warmup_teacher_temp=0.07,
            teacher_temp=0.04,
            warmup_teacher_temp_epochs=5,
            student_temp=0.1,
    ):
        
        super().__init__()
        self.student_temp=student_temp
        self.teacher_temp=teacher_temp
        self.warmup_teacher_temp_epochs=warmup_teacher_temp_epochs
        self.warmup_teacher_temp=warmup_teacher_temp

        self.register_buffer("center",torch.zeros(1,out_dim))

    
    def forward(self,student_output,teacher_output,epoch):
        
        # Args:
        #Student Output:[global 1, global 2]
        #Teacher Output:[global 1, global 2]

        student_output=torch.cat([s/ self.student_temp for s in student_output])
        
        # Calculate temperature for this epoch
        temp=self.teacher_temp
        if epoch<self.warmup_teacher_temp_epochs:
            temp=self.warmup_teacher_temp + (epoch/self.warmup_teacher_temp_epochs)*(self.teacher_temp-self.warmup_teacher_temp)

        # FIXED: Update center on RAW teacher logits (before scaling)
        teacher_output_raw = torch.cat(teacher_output)
        self.update_center(teacher_output_raw)
        
        # FIXED: Apply centering and temperature scaling
        teacher_output = (teacher_output_raw - self.center) / temp
        
        # FIXED: Clamp to prevent extreme values
        teacher_output = teacher_output.clamp(min=-10, max=10)
        
        student_out=F.log_softmax(student_output,dim=-1)
        teacher_out=F.softmax(teacher_output,dim=-1)
        
        loss=0
        batch_size = student_output.shape[0] // 2
        loss+=torch.sum(
            -teacher_out[batch_size:]*student_out[:batch_size],dim=-1
        ).mean()

        loss+=torch.sum(
            -teacher_out[:batch_size]* student_out[batch_size:],dim=-1

        ).mean()

        loss=loss/2
        return loss
    
    def update_center(self,teacher_output):
        batch_center = teacher_output.detach().mean(dim=0, keepdim=True)
        self.center=0.9* self.center+ batch_center*0.1

def cosine_scheduler(base_value,final_value,epochs,niters_per_ep,warmup_epochs=0):
    
    warmup_schedule=torch.linspace(0,base_value,warmup_epochs*niters_per_ep) if warmup_epochs>0 else torch.tensor([])
    iters=torch.arange(epochs* niters_per_ep- warmup_epochs*niters_per_ep)
    schedule=final_value+0.5*(base_value-final_value)*(1+torch.cos(math.pi*iters/len(iters)))
    schedule=torch.cat((warmup_schedule,schedule))

    return schedule

@torch.no_grad()
def update_teacher(student,teacher,momentum):
    for params_s,params_t in zip(student.parameters(),teacher.parameters()):
         params_t.mul_(momentum).add_(params_s, alpha=1 - momentum)

def train(
        data_dir,
        output_dir="checkpoints",
        arch="vit_small",
        batch_size=256,
        epochs=100,
        lr=0.0005,
        warmup_epochs=10,
        weight_decay=0.04,
        momentum_teacher=0.996,
        out_dim=65536,
        device="cuda",
        save_every=5,
        resume=None,
        clip_grad=3.0,

):
    
    os.makedirs(output_dir,exist_ok=True)
    device=torch.device(device if torch.cuda.is_available() else "cpu")

    from dino_dataloader import get_dino_dataloader
    train_loader=get_dino_dataloader(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=4,
    )

    from model import build_dino_model
    student,teacher=build_dino_model(arch=arch,out_dim=out_dim)
    student=student.to(device)
    teacher=teacher.to(device)

    dino_loss=DinoLoss(out_dim=out_dim,warmup_teacher_temp_epochs=warmup_epochs).to(device)
    
    params_groups=[{'params': [p for n,p in student.named_parameters() if p.requires_grad]},]

    optimizer=torch.optim.AdamW(params_groups,lr=lr,weight_decay=weight_decay)
    
    niter_per_ep=len(train_loader)
    lr_schedule= cosine_scheduler(lr,1e-6,epochs,niter_per_ep,warmup_epochs=warmup_epochs)

    momentum_schedule=cosine_scheduler(momentum_teacher,1.0,epochs,niter_per_ep)

    scaler=GradScaler(enabled=(device.type == "cuda"))
    start_epoch=0

    if resume and os.path.exists(resume):
        print(f"Resuming from {resume}")
        checkpoint = torch.load(resume, map_location=device)
        student.load_state_dict(checkpoint['student'])
        teacher.load_state_dict(checkpoint['teacher'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")
    

    print(f"Start DINO training on {device}")
    print(f"Dataset: {len(train_loader.dataset)} images")
    print(f"Batch size: {batch_size}, Epochs: {epochs}")
    print(f"Model: {arch}, Output dim: {out_dim}")
    print(f"Learning rate: {lr}, Warmup epochs: {warmup_epochs}")
    print(f"Teacher momentum: {momentum_teacher}, Gradient clip: {clip_grad}")
    print(f"Using GLOBAL CROPS ONLY (2 views per image)")

    for epoch in range(start_epoch,epochs):
        student.train()
        teacher.eval()

        epoch_loss=0
        progress_bar=tqdm(train_loader,desc=f"Epoch{epoch+1}/{epochs}")

        for it,crops in enumerate(progress_bar):
            iteration=epoch*niter_per_ep+it
            for param_group in optimizer.param_groups:
                param_group['lr']=lr_schedule[iteration]
            crops=[crop.to(device) for crop in crops]

            with autocast():
                student_output=student(crops)
                with torch.no_grad():
                    teacher_output=teacher(crops)
                loss=dino_loss(student_output,teacher_output,epoch)
            
            # Check for NaN/Inf before backprop
            if not torch.isfinite(loss):
                print(f"WARNING: Non-finite loss detected at epoch {epoch+1}, iter {it}. Skipping batch.")
                continue
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            
            # Gradient clipping
            if clip_grad > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), clip_grad)
            
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                m=momentum_schedule[iteration]
                update_teacher(student,teacher,m)
            
            epoch_loss+=loss.item()
            progress_bar.set_postfix(
                {
                'loss': f'{loss.item():.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.6f}',
                'mom': f'{m:.4f}'
                }
            )

        
        avg_loss=epoch_loss/len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} - Avg Loss: {avg_loss:.4f}")
        
        # Early collapse detection
        if avg_loss < 0.5:
            print(f"WARNING: Loss dropped below 0.5 (={avg_loss:.4f}). Possible collapse!")
        
        # Save checkpoint
        if (epoch + 1) % save_every == 0 or epoch == epochs - 1:
            checkpoint = {
                'student': student.state_dict(),
                'teacher': teacher.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'arch': arch,
                'out_dim': out_dim,
            }
            save_path = os.path.join(output_dir, f'checkpoint_{epoch+1:04d}.pth')
            torch.save(checkpoint, save_path)
            print(f"Saved checkpoint: {save_path}")
    

    final_path=os.path.join(output_dir,'final_model.pth')
    torch.save(
        {
            'student':student.state_dict(),
            'teacher':teacher.state_dict(),
            'arch':arch,
            'out_dim':out_dim,
        },final_path
    )

    print(f"Training Complete! Final model saved to {final_path}")
    return student,teacher



if __name__ =="__main__":
    # Train DINO (global crops only) - CORRECTED VERSION
    student, teacher = train(
        data_dir="/home/senthilkumar.m/Dino_Pathology/data/NCT-CRC-HE-100K/dataset",
        output_dir="checkpoints_dino_lunit",
        arch="vit_small",
        batch_size=256,
        epochs=30,
        lr=0.0003,
        warmup_epochs=5,
        weight_decay=0.04,
        momentum_teacher=0.996,
        out_dim=16384,          # REDUCED: from 65536 for stability
        device="cuda",
        save_every=2,
        resume="/home/senthilkumar.m/Dino_Pathology/checkpoints_dino_lunit/checkpoint_0002.pth",
        clip_grad=3.0,
    )