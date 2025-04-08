import os
import sys
import time
import csv

import torch
from torch.utils.tensorboard import SummaryWriter
import torchmetrics
from torchvision.utils import make_grid
from tqdm import tqdm
from torchmetrics.functional import (
    structural_similarity_index_measure as ssim,
    peak_signal_noise_ratio as psnr
)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance

# Import DataParallel if you wish to wrap your models
import torch.nn as nn
import torch.nn.functional as F  # Import for functional operations

from utils.dataset import get_dataloader
from utils.cycle_gan_model import get_cycle_gan_model
from utils.image_processing import merge_patches, process_images_parallel
from utils.global_variables import PATCH_SIZE
from apply_model import apply_gan_model
import cv2
from datetime import datetime
from glob2 import glob
import pathlib

# Training parameters
EPOCHS = 50
BATCH_SIZE = 8
SAVE_EVERY_EPOCH = True
LAMBDA_CYCLE = 10.0 # Suggestion: Emphasize cycle consistency
LAMBDA_IDENTITY = 0.5 # Suggestion: Identity loss weight

# Explicit CUDA initialization
if torch.cuda.is_available():
    torch.cuda.init()
    torch.cuda.empty_cache()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    print(f"✅ Initialized CUDA on device: {device}")
    # if torch.cuda.device_count() > 1:
    #     for n_device in torch.cuda.device_count():
    #         t_device = torch.device(f"cuda:{n_device}")
    #         torch.cuda.set_device(t_device)
    #         print(f"✅ Initialized CUDA on device: {t_device}")
    
# Now you continue as usual:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)
fid_metric = FrechetInceptionDistance(feature=2048).to(device)

LOG_DIR = "logs"
CHECKPOINT_DIR = "checkpoints"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
writer = SummaryWriter(LOG_DIR)

def save_checkpoint(G_AB, G_BA, D_A, D_B, optimizer_G, optimizer_D_A, optimizer_D_B, epoch):
    """
    Saves a CycleGAN checkpoint.
    Note: If models are wrapped in DataParallel, the keys in state_dict
    will have 'module.' prefix. That is typically fine as long as you
    load them the same way. If you want to remove the 'module.' prefix,
    you can do so manually before saving or after loading.
    """
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"cycle_gan_epoch_{epoch}.pth")
    
    # If using DataParallel, you might prefer to save the underlying model.module state_dict:
    # e.g., G_AB.module.state_dict() if isinstance(G_AB, nn.DataParallel) else G_AB.state_dict()
    # For simplicity, we save directly:
    torch.save({
        "epoch": epoch,
        "G_AB": G_AB.state_dict(),
        "G_BA": G_BA.state_dict(),
        "D_A": D_A.state_dict(),
        "D_B": D_B.state_dict(),
        "optimizer_G": optimizer_G.state_dict(),
        "optimizer_D_A": optimizer_D_A.state_dict(),
        "optimizer_D_B": optimizer_D_B.state_dict(),
    }, checkpoint_path)
    print(f"✅ Checkpoint saved: {checkpoint_path}")

def log_epoch_to_csv(csv_path,
                     epoch,
                     G_loss,
                     D_loss,
                     mae,
                     mse,
                     ssim_val,
                     psnr_val,
                     lpips_val,
                     fid_val):
    """
    Appends a row of metrics to a CSV file. Creates the file with a header if it doesn't exist.
    """
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        # Write header if file did not exist
        if not file_exists:
            writer.writerow(["epoch", "G_loss", "D_loss", "MAE", "MSE", "SSIM", "PSNR", "LPIPS", "FID"])
        # Write the row of metrics
        writer.writerow([epoch, G_loss, D_loss, mae, mse, ssim_val, psnr_val, lpips_val, fid_val])


def train_epoch(G_AB, G_BA, D_A, D_B,
                optimizer_G, optimizer_D_A, optimizer_D_B,
                dataloader, epoch,
                adversarial_loss, cycle_loss, identity_loss):
    """
    One training epoch for CycleGAN.
    """
    G_AB.train()
    G_BA.train()
    D_A.train()
    D_B.train()

    epoch_G_loss = 0.0
    epoch_D_loss = 0.0
    epoch_mae = 0.0
    epoch_mse = 0.0
    epoch_ssim, epoch_psnr, epoch_lpips = 0.0, 0.0, 0.0
                    
    # Initialize metrics
    mse_metric = torchmetrics.MeanSquaredError().to(device)
    mae_metric = torch.nn.L1Loss()
    
    # Reset FID stats at the start of an epoch
    fid_metric.reset()
    
    num_batches = len(dataloader)
    
    for batch_idx, (real_A, real_B) in enumerate(tqdm(dataloader)):
        real_A = real_A.to(device)
        real_B = real_B.to(device)

        # -------------------------------
        #  Generate images
        # -------------------------------
        fake_B = G_AB(real_A)
        fake_A = G_BA(real_B)

        # Reconstructed images (cycle)
        rec_A = G_BA(fake_B)
        rec_B = G_AB(fake_A)

        # -------------------------------
        #  Update Generators
        # -------------------------------
        optimizer_G.zero_grad()

        # Adversarial Loss (G_AB, G_BA must fool the discriminators)
        loss_G_AB = adversarial_loss(D_B(fake_B), torch.ones_like(D_B(fake_B)))
        loss_G_BA = adversarial_loss(D_A(fake_A), torch.ones_like(D_A(fake_A)))
        
        # Cycle Consistency Loss
        loss_cycle_A = cycle_loss(rec_A, real_A) * LAMBDA_CYCLE
        loss_cycle_B = cycle_loss(rec_B, real_B) * LAMBDA_CYCLE
        
        # Identity Loss
        loss_identity_A = identity_loss(G_BA(real_A), real_A) * LAMBDA_IDENTITY
        loss_identity_B = identity_loss(G_AB(real_B), real_B) * LAMBDA_IDENTITY

        # Combined generator loss
        loss_G = (loss_G_AB + loss_G_BA
                  + loss_cycle_A + loss_cycle_B
                  + loss_identity_A + loss_identity_B)

        # Combined generator loss
        # loss_G = (loss_G_AB + loss_G_BA
        #           + 8 * (loss_cycle_A + loss_cycle_B)
        #           + 0.5 * (loss_identity_A + loss_identity_B))
        
        loss_G.backward()
        optimizer_G.step()

        # -------------------------------
        #  Update Discriminator A
        # -------------------------------
        optimizer_D_A.zero_grad()
        loss_D_A_real = adversarial_loss(D_A(real_A), torch.ones_like(D_A(real_A)))
        loss_D_A_fake = adversarial_loss(D_A(fake_A.detach()), torch.zeros_like(D_A(fake_A)))
        loss_D_A_total = (loss_D_A_real + loss_D_A_fake) / 2
        loss_D_A_total.backward()
        optimizer_D_A.step()

        # -------------------------------
        #  Update Discriminator B
        # -------------------------------
        optimizer_D_B.zero_grad()
        loss_D_B_real = adversarial_loss(D_B(real_B), torch.ones_like(D_B(real_B)))
        loss_D_B_fake = adversarial_loss(D_B(fake_B.detach()), torch.zeros_like(D_B(fake_B)))
        loss_D_B_total = (loss_D_B_real + loss_D_B_fake) / 2
        loss_D_B_total.backward()
        optimizer_D_B.step()

        epoch_G_loss += loss_G.item()
        epoch_D_loss += (loss_D_A_total.item() + loss_D_B_total.item())
        
        # ---------------------------------
        # Compute Additional Metrics
        # ---------------------------------
        
        from torchmetrics.functional import mean_absolute_error, mean_squared_error

        # Compute pixel-level MAE between fake_B and real_B
        mae_fakeB_realB = mean_absolute_error(fake_B, real_B)
        mse_fakeB_realB = mean_squared_error(fake_B, real_B)
        epoch_mae += mae_fakeB_realB.item()
        epoch_mse += mse_fakeB_realB.item()

        # ------------------
        # Reference-based metrics
        # ------------------
        
        # If your images are in [-1, 1], shift them to [0, 1] for metrics:
        fake_B_01 = 0.5 * (fake_B + 1.0)
        real_B_01 = 0.5 * (real_B + 1.0)

        # 1) SSIM
        ssim_val = ssim(fake_B_01, real_B_01)
        epoch_ssim += ssim_val.item()

        # 2) PSNR
        psnr_val = psnr(fake_B_01, real_B_01)
        epoch_psnr += psnr_val.item()

        # 3) LPIPS
        lpips_val = lpips_metric(fake_B_01, real_B_01)
        epoch_lpips += lpips_val.item()

        # 4) FID (distribution-based)
        # Accumulate features for real_B and fake_B

        # Convert [0,1] to [0,255] and cast to uint8
        fake_B_uint8 = (fake_B_01 * 255).clamp(0, 255).to(torch.uint8)
        real_B_uint8 = (real_B_01 * 255).clamp(0, 255).to(torch.uint8)
        
        fid_metric.update(fake_B_uint8, real=True)
        fid_metric.update(real_B_uint8, real=False)
        
        # Log metrics to TensorBoard per batch
        global_step = epoch * len(dataloader) + batch_idx
        
        writer.add_scalar("Metrics_Iter/MAE_FakeB_RealB", mae_fakeB_realB, global_step)
        writer.add_scalar("Metrics_Iter/MSE_FakeB_RealB", mse_fakeB_realB, global_step)
        writer.add_scalar("Metrics_Iter/SSIM", ssim_val, global_step)
        writer.add_scalar("Metrics_Iter/PSNR", psnr_val, global_step)
        writer.add_scalar("Metrics_Iter/LPIP", lpips_val, global_step)
        writer.add_scalar("Loss_Iter/G_iter", loss_G.item(), global_step)
        writer.add_scalar("Loss_Iter/D_iter", loss_D_A_total.item(), global_step)
        
    # ---------------------------------
    # Log epoch-level summaries
    # ---------------------------------
    
    epoch_mae /= num_batches
    epoch_mse /= num_batches
    # End of epoch: compute average SSIM, PSNR, LPIPS
    
    epoch_ssim /= num_batches
    epoch_psnr /= num_batches
    epoch_lpips /= num_batches

    # Compute FID for the entire epoch
    epoch_fid = fid_metric.compute().item()
    
    return (epoch_G_loss / len(dataloader), epoch_D_loss / len(dataloader),\
            epoch_ssim, epoch_psnr, \
            epoch_lpips, epoch_fid, \
            epoch_mae, epoch_mse)

def train_model(source_dir: str, target_dir: str,
                input_test_dir: str, output_test_dir: str,
                output_merged_dir: str,  csv_path:str = "./fix_logs.csv"):
    """
    Main training loop for CycleGAN.
    """
    train_dataloader = get_dataloader(source_dir, target_dir, BATCH_SIZE, (PATCH_SIZE, PATCH_SIZE))

    # Load CycleGAN components
    G_AB, G_BA, D_A, D_B, \
    optimizer_G, optimizer_D_A, optimizer_D_B, \
    adversarial_loss, cycle_loss, identity_loss = get_cycle_gan_model()

    # -------------------------------
    #  Wrap models in DataParallel
    #  if multiple GPUs are available
    # -------------------------------
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training!")
        G_AB = nn.DataParallel(G_AB)
        G_BA = nn.DataParallel(G_BA)
        D_A = nn.DataParallel(D_A)
        D_B = nn.DataParallel(D_B)

    # Move models to the chosen device (e.g., "cuda:0")
    G_AB.to(device)
    G_BA.to(device)
    D_A.to(device)
    D_B.to(device)

    best_G_loss = float("inf")
    epoch = 0
    train_G_loss = 1.0

    # Suggestion: Implement a learning rate scheduler
    scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=EPOCHS, eta_min=0)
    scheduler_D_A = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D_A, T_max=EPOCHS, eta_min=0)
    scheduler_D_B = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D_B, T_max=EPOCHS, eta_min=0)


    while train_G_loss > 0.2:
        train_G_loss, train_D_loss, \
        epoch_ssim, epoch_psnr, \
        epoch_lpips, epoch_fid, \
        epoch_mae, epoch_mse = train_epoch(
            G_AB, G_BA, D_A, D_B,
            optimizer_G, optimizer_D_A, optimizer_D_B,
            train_dataloader, epoch,
            adversarial_loss, cycle_loss, identity_loss
        )
        
        print(f"📉 Epoch {epoch}, G Loss: {train_G_loss:.4f}, D Loss: {train_D_loss:.4f}")
        
        writer.add_scalar("Loss_Epoch/G_train", train_G_loss, epoch)
        writer.add_scalar("Loss_Epoch/D_train", train_D_loss, epoch)
        writer.add_scalar("Metrics_Epoch/SSIM", epoch_ssim, epoch)
        writer.add_scalar("Metrics_Epoch/PSNR", epoch_psnr, epoch)
        writer.add_scalar("Metrics_Epoch/LPIP", epoch_lpips, epoch)
        writer.add_scalar("Metrics_Epoch/FID", epoch_fid, epoch)
        writer.add_scalar("Metrics_Epoch/MAE_FakeB_RealB_epoch", epoch_mae, epoch)
        writer.add_scalar("Metrics_Epoch/MSE_FakeB_RealB_epoch", epoch_mse, epoch)
        log_epoch_to_csv(csv_path,
                         epoch,
                         train_G_loss,
                         train_D_loss,
                         epoch_mae,
                         epoch_mse,
                         epoch_ssim,
                         epoch_psnr,
                         epoch_lpips,
                         epoch_fid)
        
        if train_G_loss < best_G_loss or SAVE_EVERY_EPOCH:
            best_G_loss = train_G_loss
            save_checkpoint(G_AB, G_BA, D_A, D_B,
                            optimizer_G, optimizer_D_A, optimizer_D_B,
                            epoch)

        # Apply the generator G_AB to the test patches and merge them for visualization
        apply_gan_model(
            f"./checkpoints/cycle_gan_epoch_{epoch}.pth",
            input_test_dir,
            output_test_dir,
            generator_type="G_AB"
        )
        image = cv2.imread('./data/Raw/raw/003_0009.jpg')
        original_size = image.shape[:2]
        merge_patches(output_test_dir, f'{output_merged_dir}', epoch, original_size)

        #new block: adds image to tensorboard log
        from torchvision.transforms import  ToTensor
        from PIL import Image

        # Path to merged patches
        merged_patches_path = os.path.join(output_merged_dir, f'003_0009_{epoch}.jpg')
       
        if os.path.exists(merged_patches_path):
            merged_image = Image.open(merged_patches_path).convert('RGB')
            transform = Compose([Resize((original_size/3, original_size/3)), ToTensor()])
            merged_tensor = transform(merged_image)
            writer.add_image('Merged_image', merged_tensor, epoch)
        else:
            print(f'Warning: Merged image not found at {merged_patches_path}!')
            
        # Step the schedulers
        scheduler_G.step()
        scheduler_D_A.step()
        scheduler_D_B.step()
        
        epoch += 1

    writer.close()
    print("🎉 Training completed!")

if __name__ == "__main__":
    os.makedirs('data/patches', exist_ok=True)
    os.makedirs('data/patches/raw', exist_ok=True)
    os.makedirs('data/patches/GT', exist_ok=True)
    os.makedirs('data/patches/test', exist_ok=True)
    now = datetime.now()

    output_patches_raw = './data/patches/raw'
    output_patches_gt = './data/patches/GT'
    output_patches_test = './data/patches/test'

    process_images_parallel('./data/Raw/raw/*', output_patches_raw)
    print(f'Done with train raw patches | time > {datetime.now() - now}')

    import time
    time.sleep(10)

    now = datetime.now()
    process_images_parallel('./data/Raw/GT/*', output_patches_gt)
    print(f'Done with raw GT patches | time > {datetime.now() - now}')
    now = datetime.now()

    process_images_parallel('./data/Raw/test/raw/*', output_patches_test)
    print(f'Done with raw test patches | time > {datetime.now() - now}')

    source = './data/patches/raw/'
    target = './data/patches/GT/'
    input_test_dir = './data/patches/test'
    output_test_dir = './data/patches/test_model'
    output_merged_dir = './data/test_model'

    os.makedirs(output_test_dir, exist_ok=True)
    os.makedirs(output_merged_dir, exist_ok=True)

    os.makedirs(os.path.join(pathlib.Path(__file__).parent.resolve(), 'csv_logs'), exist_ok=True)
    csv_path = os.path.join(pathlib.Path(__file__).parent.resolve(), 'csv_logs', f'training_{str(datetime.now()).split(" ")[0]}_logs.csv')
    
    train_model(source, target, input_test_dir, output_test_dir, output_merged_dir, csv_path)
