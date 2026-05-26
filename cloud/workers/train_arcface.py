"""ArcFace fine-tuning of Wildlife-mega-L-384 for horse re-identification.

Usage:
    # Activate venv first
    source .venv/bin/activate

    # Run from project root (loads DATABASE_URL from cloud/.env)
    python cloud/workers/train_arcface.py

    # Custom Drive root:
    python cloud/workers/train_arcface.py --drive-root /path/to/drive/root

    # Upload to S3 after training:
    python cloud/workers/train_arcface.py --upload-s3
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import timm
import torch
import torchvision.transforms as T
from wildlife_tools.data import ImageDataset
from wildlife_tools.train import ArcFaceLoss, BasicTrainer, set_seed

# Add parent dir so we can import config
sys.path.insert(0, os.path.dirname(__file__))
import config
import db

MODEL_NAME = "hf-hub:BVRA/wildlife-mega-L-384"
IMAGE_SIZE = 384
EMBEDDING_DIM = 1536
OUTPUT_PATH = "finetuned-mega-L-384.pth"
CHECKPOINT_DIR = "checkpoints"
S3_BUCKET = "horse-id-models"
S3_KEY = "finetuned-mega-L-384.pth"

DEFAULT_DRIVE_ROOT = os.path.expanduser(
    "~/Google Drive/Shared drives/horse-id/cloud-data-root"
)


def load_training_data(drive_root: str) -> pd.DataFrame:
    """Query DB for training photos and map to local file paths."""
    conn = db.get_connection()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT p.filename, h.name AS horse_name, h.id AS horse_id, hd.name AS herd_name
            FROM photos p
            JOIN horses h ON h.id = p.horse_id
            JOIN herds hd ON hd.id = h.herd_id
            WHERE p.processing_status = 'ready'
              AND p.detection_result = 'SINGLE'
              AND p.excluded = false
        """)
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, row)) for row in cur.fetchall()]

    print(f"Found {len(rows)} eligible photos in DB")

    records = []
    skipped = 0
    for row in rows:
        path = os.path.join(drive_root, row["herd_name"], row["horse_name"], row["filename"])
        if os.path.exists(path):
            records.append({"path": path, "identity": row["horse_id"]})
        else:
            skipped += 1

    if skipped > 0:
        print(f"Skipped {skipped} photos (not found on local Drive)")

    df = pd.DataFrame(records)
    n_horses = df["identity"].nunique()
    print(f"Training set: {len(df)} photos, {n_horses} horses")
    return df


def build_transform():
    """Training augmentation — no cropping/masking (previously hurt performance)."""
    return T.Compose([
        T.Resize([IMAGE_SIZE, IMAGE_SIZE]),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        T.RandomErasing(p=0.3),
    ])


def freeze_early_layers(model):
    """Freeze everything except the last Swin stage + norm.

    Swin-L parameter breakdown:
      layers.0:     896K
      layers.1:   3,858K
      layers.2: 128,993K
      layers.3:  61,439K
      norm:           3K
      patch_embed:   10K
    Training only layers.3 + norm keeps ~61M trainable (vs 195M total).
    """
    for name, param in model.named_parameters():
        if not (name.startswith("layers.3") or name.startswith("norm")):
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Parameters: {trainable:,} trainable, {frozen:,} frozen")


def main():
    parser = argparse.ArgumentParser(description="ArcFace fine-tuning for horse re-ID")
    parser.add_argument("--drive-root", default=DEFAULT_DRIVE_ROOT, help="Path to Drive mount root")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--upload-s3", action="store_true", help="Upload weights to S3 after training")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    print(f"Device: {args.device}")
    set_seed(args.seed)

    # Init config (loads DATABASE_URL from cloud/.env)
    config.init()

    # Load training data
    df = load_training_data(args.drive_root)
    if len(df) == 0:
        print("No training data found. Check --drive-root path.")
        sys.exit(1)

    num_classes = df["identity"].nunique()

    # Build dataset with augmentation
    dataset = ImageDataset(df, transform=build_transform())

    # Load pretrained backbone
    print(f"Loading {MODEL_NAME}...")
    backbone = timm.create_model(MODEL_NAME, num_classes=0, pretrained=True)
    freeze_early_layers(backbone)

    # ArcFace objective
    objective = ArcFaceLoss(
        num_classes=num_classes,
        embedding_size=EMBEDDING_DIM,
        margin=0.5,
        scale=64,
    )

    # Optimizer — higher lr for ArcFace head (randomly initialized) vs backbone (pretrained)
    params = [
        {"params": [p for p in backbone.parameters() if p.requires_grad], "lr": args.lr},
        {"params": objective.parameters(), "lr": args.lr * 10},
    ]
    optimizer = torch.optim.SGD(params, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Trainer
    trainer = BasicTrainer(
        dataset=dataset,
        model=backbone,
        objective=objective,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=args.epochs,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=0,
    )

    if args.resume:
        print(f"Resuming from {args.resume}")
        trainer.load(args.resume)
        # BasicTrainer.train() always runs range(self.epochs), so adjust
        # to only run the remaining epochs
        remaining = args.epochs - trainer.epoch
        if remaining <= 0:
            print(f"Already at epoch {trainer.epoch}, nothing to do.")
            sys.exit(0)
        trainer.epochs = remaining
        print(f"Resuming from epoch {trainer.epoch}, {remaining} epochs remaining")

    # Save checkpoint after each epoch
    def epoch_callback(trainer, epoch_data):
        loss = epoch_data["train_loss_epoch_avg"]
        print(f"  Epoch {trainer.epoch}: loss={loss:.4f}")
        trainer.save(CHECKPOINT_DIR, file_name=f"checkpoint_epoch_{trainer.epoch}.pth")
        # Also save latest
        trainer.save(CHECKPOINT_DIR, file_name="checkpoint_latest.pth")

    trainer.epoch_callback = epoch_callback

    print(f"\nStarting training: {trainer.epochs} epochs, batch_size={args.batch_size}, lr={args.lr}")
    trainer.train()

    # Save final backbone weights (not ArcFace head)
    print(f"\nSaving backbone weights to {OUTPUT_PATH}")
    torch.save(backbone.state_dict(), OUTPUT_PATH)

    if args.upload_s3:
        import boto3
        print(f"Uploading to s3://{S3_BUCKET}/{S3_KEY}")
        s3 = boto3.client("s3", region_name="us-east-2")
        s3.upload_file(OUTPUT_PATH, S3_BUCKET, S3_KEY)
        print("Upload complete.")

    print("Done.")


if __name__ == "__main__":
    main()
