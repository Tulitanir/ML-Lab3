import argparse

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import (
    MNIST_ROOT, CKPT_DIR, DEVICE,
    IMG_SIZE, LATENT_DIM, NUM_STEPS, BETA,
    BATCH_SIZE, EPOCHS, LR, NOISE_FACTOR,
)
from model import HopfieldDenoiser


def _mnist_loader(train: bool) -> DataLoader:
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    ds = datasets.MNIST(root=str(MNIST_ROOT), train=train, download=True, transform=tf)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=train, num_workers=2, pin_memory=True)


def _add_noise(x: torch.Tensor, factor: float) -> torch.Tensor:
    return (x + factor * torch.randn_like(x)).clamp(0.0, 1.0)


def _run_epoch(
    model: HopfieldDenoiser,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, total_psnr, n = 0.0, 0.0, 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for clean, _ in loader:
            clean = clean.to(DEVICE)
            noisy = _add_noise(clean, NOISE_FACTOR)

            restored = model(noisy)
            loss = criterion(restored, clean)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            bs = clean.size(0)
            total_loss += loss.item() * bs
            mse = nn.functional.mse_loss(restored.detach(), clean).item()
            psnr = 10 * torch.log10(torch.tensor(1.0 / (mse + 1e-8))).item()
            total_psnr += psnr * bs
            n += bs

    return total_loss / n, total_psnr / n


def train() -> None:
    CKPT_DIR.mkdir(exist_ok=True)
    train_loader = _mnist_loader(train=True)
    val_loader = _mnist_loader(train=False)

    model = HopfieldDenoiser(
        img_size=IMG_SIZE,
        latent_dim=LATENT_DIM,
        num_steps=NUM_STEPS,
        beta=BETA,
    ).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print(
        f"HopfieldDenoiser | img {IMG_SIZE}x{IMG_SIZE} | latent {LATENT_DIM} | "
        f"steps {NUM_STEPS} | beta {BETA} | noise sigma={NOISE_FACTOR} | device {DEVICE}"
    )

    history: dict[str, list[float]] = {
        "train_loss": [], "val_loss": [], "train_psnr": [], "val_psnr": []
    }
    best_psnr = 0.0

    for epoch in range(1, EPOCHS + 1):
        t_loss, t_psnr = _run_epoch(model, train_loader, criterion, optimizer)
        v_loss, v_psnr = _run_epoch(model, val_loader, criterion, None)
        scheduler.step()

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["train_psnr"].append(t_psnr)
        history["val_psnr"].append(v_psnr)

        flag = ""
        if v_psnr > best_psnr:
            best_psnr = v_psnr
            torch.save(
                {"epoch": epoch, "model": model.state_dict(),
                 "latent_dim": LATENT_DIM, "num_steps": NUM_STEPS, "beta": BETA},
                CKPT_DIR / "best.pt",
            )
            flag = "  <- best"

        print(
            f"Epoch {epoch:3d}/{EPOCHS} | "
            f"train loss {t_loss:.5f} PSNR {t_psnr:.2f} | "
            f"val loss {v_loss:.5f} PSNR {v_psnr:.2f}{flag}"
        )

    torch.save(
        {"epoch": EPOCHS, "model": model.state_dict(),
         "latent_dim": LATENT_DIM, "num_steps": NUM_STEPS, "beta": BETA},
        CKPT_DIR / "last.pt",
    )
    _plot_history(history, CKPT_DIR / "training_curves.png")
    print(f"\nBest val PSNR: {best_psnr:.2f} dB")


def _plot_history(history: dict, save_path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, history["train_loss"], label="Train")
    ax1.plot(epochs, history["val_loss"], label="Val")
    ax1.set_title("MSE Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, history["train_psnr"], label="Train")
    ax2.plot(epochs, history["val_psnr"], label="Val")
    ax2.set_title("PSNR (dB)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("PSNR")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Plot saved to {save_path}")


def _load_model(checkpoint: str) -> HopfieldDenoiser:
    ckpt = torch.load(checkpoint, map_location=DEVICE, weights_only=False)
    model = HopfieldDenoiser(
        img_size=IMG_SIZE,
        latent_dim=ckpt.get("latent_dim", LATENT_DIM),
        num_steps=ckpt.get("num_steps", NUM_STEPS),
        beta=ckpt.get("beta", BETA),
    ).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


@torch.no_grad()
def restore(image_path: str, checkpoint: str, noise: float = 0.0) -> None:
    """Restore a noisy image and save the result next to the source file."""
    from PIL import Image

    model = _load_model(checkpoint)

    tf = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    img_tensor = tf(Image.open(image_path)).unsqueeze(0).to(DEVICE)

    if noise > 0:
        img_tensor = _add_noise(img_tensor, noise)

    restored = model(img_tensor)

    out_path = image_path.replace(".", "_restored.")
    transforms.ToPILImage()(restored.squeeze(0).cpu()).save(out_path)
    print(f"Restored image saved to {out_path}")

    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(img_tensor.squeeze().cpu(), cmap="gray")
    axes[0].set_title("Input (noisy)")
    axes[0].axis("off")
    axes[1].imshow(restored.squeeze().cpu(), cmap="gray")
    axes[1].set_title("Restored")
    axes[1].axis("off")
    plt.tight_layout()
    cmp_path = out_path.replace("_restored.", "_comparison.")
    plt.savefig(cmp_path, dpi=150)
    print(f"Comparison saved to {cmp_path}")
    plt.show()


@torch.no_grad()
def demo(checkpoint: str, n_samples: int = 10) -> None:
    """Visualise clean / noisy / restored triplets from the MNIST test set."""
    model = _load_model(checkpoint)
    loader = _mnist_loader(train=False)
    clean_batch, _ = next(iter(loader))
    clean_batch = clean_batch[:n_samples].to(DEVICE)
    noisy_batch = _add_noise(clean_batch, NOISE_FACTOR)
    restored_batch = model(noisy_batch)

    fig, axes = plt.subplots(3, n_samples, figsize=(n_samples * 1.5, 5))
    for row, (batch, title) in enumerate(
        zip([clean_batch, noisy_batch, restored_batch], ["Clean", "Noisy", "Restored"])
    ):
        for col in range(n_samples):
            ax = axes[row, col]
            ax.imshow(batch[col].squeeze().cpu(), cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(title, fontsize=10)

    plt.suptitle(f"Hopfield Denoiser  |  noise sigma={NOISE_FACTOR}  |  {IMG_SIZE}x{IMG_SIZE} MNIST")
    plt.tight_layout()
    out = CKPT_DIR / "demo.png"
    plt.savefig(out, dpi=150)
    print(f"Demo saved to {out}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hopfield network image denoiser")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    subparsers.add_parser("train", help="Train HopfieldDenoiser on MNIST")

    p_restore = subparsers.add_parser("restore", help="Restore a noisy image")
    p_restore.add_argument("--input", required=True, help="Path to input image")
    p_restore.add_argument("--checkpoint", default=str(CKPT_DIR / "best.pt"))
    p_restore.add_argument(
        "--noise", type=float, default=0.0,
        help="Additional Gaussian noise to add before restoring (0 = none)"
    )

    p_demo = subparsers.add_parser("demo", help="Visualise denoising on MNIST test samples")
    p_demo.add_argument("--checkpoint", default=str(CKPT_DIR / "best.pt"))
    p_demo.add_argument("--n", type=int, default=10, help="Number of samples to show")

    args = parser.parse_args()

    if args.mode == "train":
        train()
    elif args.mode == "restore":
        restore(args.input, args.checkpoint, noise=args.noise)
    elif args.mode == "demo":
        demo(args.checkpoint, n_samples=args.n)
