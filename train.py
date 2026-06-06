import logging
import sys
import time
import traceback

import torch
import torch.nn as nn
import torch.optim as optim

from model import UNet
from dataset import get_dataloaders


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("train.log"),
    ],
)
log = logging.getLogger(__name__)


class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)

        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=5)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

        intersection = (probs * targets_one_hot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


def train():
    log.info("=== Training started ===")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    log.info(f"Device: {device}")
    if device.type == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")
        log.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    log.info("Loading dataloaders...")
    try:
        train_loader, val_loader = get_dataloaders()
        log.info(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

        # Checking the first batch
        sample_imgs, sample_masks = next(iter(train_loader))
        log.info(f"Sample batch — images: {sample_imgs.shape} {sample_imgs.dtype}, "
                 f"masks: {sample_masks.shape} {sample_masks.dtype}")
        log.debug(f"Image value range: [{sample_imgs.min():.3f}, {sample_imgs.max():.3f}]")
        log.debug(f"Mask unique values: {sample_masks.unique().tolist()}")

        if sample_imgs.max() > 1.0:
            log.warning(f"Images may not be normalized! Max value: {sample_imgs.max():.1f} "
                        f"(expected ~1.0). Consider dividing by 255.")
    except Exception:
        log.exception("Failed to load dataloaders")
        raise

    log.info("Building model...")
    try:
        model = UNet(n_classes=5).to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log.info(f"UNet created — trainable params: {n_params:,}")
    except Exception:
        log.exception("Failed to build model")
        raise

    # Weights for the classes to prevent  класів class imbalance
    weights = torch.tensor([0.3, 12.0, 1.0, 2.0, 4.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    dice_loss = DiceLoss()

    def combined_loss(outputs, masks):
        return criterion(outputs, masks) + dice_loss(outputs, masks)

    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    log.info(f"Criterion: {combined_loss} | Optimizer: Adam lr=0.0003")

    num_epochs = 50
    best_val_loss = float("inf")
    best_epoch = -1
    log.info(f"Starting training for {num_epochs} epochs")

    for epoch in range(num_epochs):
        log.info(f"--- Epoch {epoch + 1}/{num_epochs} ---")

        # Training
        model.train()
        train_loss = 0
        t0 = time.time()

        for batch_idx, (images, masks) in enumerate(train_loader):
            try:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                log.debug(f"[train] batch {batch_idx} — "
                          f"outputs: {outputs.shape}, masks: {masks.shape}")

                loss = combined_loss(outputs, masks)

                if torch.isnan(loss) or torch.isinf(loss):
                    log.error(f"[train] batch {batch_idx} — loss is {loss.item():.4f}! "
                              f"outputs range [{outputs.min():.3f}, {outputs.max():.3f}]")

                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                grad_norm = sum(
                    p.grad.norm().item() ** 2
                    for p in model.parameters() if p.grad is not None
                ) ** 0.5
                log.debug(f"[train] batch {batch_idx} — loss: {loss.item():.4f}, "
                          f"grad norm: {grad_norm:.4f}")

                optimizer.step()
                train_loss += loss.item()

            except Exception:
                log.exception(f"[train] crash at epoch {epoch + 1}, batch {batch_idx}")
                raise

        train_elapsed = time.time() - t0
        avg_train = train_loss / len(train_loader)
        log.info(f"Train — avg loss: {avg_train:.4f} | time: {train_elapsed:.1f}s")

        # Validation
        model.eval()
        val_loss = 0
        t0 = time.time()

        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(val_loader):
                try:
                    images = images.to(device)
                    masks = masks.to(device)

                    outputs = model(images)
                    loss = combined_loss(outputs, masks)

                    if torch.isnan(loss) or torch.isinf(loss):
                        log.error(f"[val] batch {batch_idx} — loss is {loss.item():.4f}!")

                    log.debug(f"[val] batch {batch_idx} — loss: {loss.item():.4f}")
                    val_loss += loss.item()

                except Exception:
                    log.exception(f"[val] crash at epoch {epoch + 1}, batch {batch_idx}")
                    raise

        val_elapsed = time.time() - t0
        avg_val = val_loss / len(val_loader)
        log.info(f"Val   — avg loss: {avg_val:.4f} | time: {val_elapsed:.1f}s")

        current_lr = optimizer.param_groups[0]['lr']
        log.info(f"LR: {current_lr:.6f}")

        # Save the best model
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_epoch = epoch + 1
            try:
                torch.save(model.state_dict(), "unet_best.pth")
                log.info(f"New best model saved (val loss: {best_val_loss:.4f})")
            except Exception:
                log.exception("Failed to save best model")
                raise

    log.info(f"Best val loss: {best_val_loss:.4f} at epoch {best_epoch}")

    save_path = "unet_final.pth"
    log.info(f"Saving final model to {save_path}...")
    try:
        torch.save(model.state_dict(), save_path)
        log.info("Final model saved successfully")
    except Exception:
        log.exception("Failed to save final model")
        raise

    log.info("=== Training complete ===")


if __name__ == "__main__":
    try:
        train()
    except Exception:
        log.critical("Training crashed:\n" + traceback.format_exc())
        sys.exit(1)