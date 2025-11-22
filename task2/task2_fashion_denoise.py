import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ======================
#  A. Dataset & Preprocess
# ======================
class PairCsvDataset(Dataset):
    """
    读取一对 noisy/clean CSV 文件，每一行是一张 28x28 = 784 像素的灰度图。
    noisy 作为输入 x，clean 作为标签 y。
    """
    def __init__(self, noisy_csv, clean_csv, normalize=True):
        self.noisy_df = pd.read_csv(noisy_csv)
        self.clean_df = pd.read_csv(clean_csv)
        assert len(self.noisy_df) == len(self.clean_df), "noisy / clean 数量不一致"
        self.normalize = normalize

    def __len__(self):
        return len(self.noisy_df)

    def __getitem__(self, idx):
        # 从 DataFrame 里取出一行数据（784 维）
        x = self.noisy_df.iloc[idx].values.astype(np.float32)
        y = self.clean_df.iloc[idx].values.astype(np.float32)

        # 归一化到 [0,1]
        if self.normalize:
            x = x / 255.0
            y = y / 255.0

        # 转成 PyTorch tensor，并 reshape 成 [C=1, H=28, W=28]
        x = torch.from_numpy(x).view(1, 28, 28)
        y = torch.from_numpy(y).view(1, 28, 28)
        return x, y


def build_dataloaders(root="/Users/tangtang/PycharmProjects/PythonProject1/Assign2/FASHION-MNIST", batch_size=128):
    root = Path(root)
    train_noisy_csv = root / "train_noisy.csv"
    train_clean_csv = root / "train_clean.csv"
    test_noisy_csv = root / "test_noisy.csv"
    test_clean_csv = root / "test_clean.csv"

    train_ds = PairCsvDataset(train_noisy_csv, train_clean_csv, normalize=True)
    test_ds = PairCsvDataset(test_noisy_csv, test_clean_csv, normalize=True)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )
    return train_loader, test_loader

# ======================
#  B. Neural Network Construction
# ======================

class DenoisingAutoencoder(nn.Module):
    """
    一个简单的卷积自编码器：
    Encoder: 1x28x28 -> 32 -> 64 通道
    Decoder: 64 -> 32 -> 1 通道，还原回干净图片。
    """
    def __init__(self):
        super().__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # [B,1,28,28] -> [B,32,28,28]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                           # [B,32,14,14]
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # [B,64,14,14]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)                            # [B,64,7,7]
        )

        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),  # [B,32,14,14]
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2),   # [B,1,28,28]
            nn.Sigmoid()  # 输出限制在 [0,1]
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out
# ======================
#  训练 & 评估
# ======================

def train_model(model, train_loader, test_loader,
                device, num_epochs=10, lr=1e-3):
    criterion = nn.MSELoss()                    # 指定的 MSE 损失
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # ---- 训练 ----
        model.train()
        running_train_loss = 0.0
        for noisy, clean in train_loader:
            noisy = noisy.to(device)
            clean = clean.to(device)

            optimizer.zero_grad()
            outputs = model(noisy)              # forward
            loss = criterion(outputs, clean)    # MSE
            loss.backward()                     # backward
            optimizer.step()

            running_train_loss += loss.item() * noisy.size(0)

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # ---- 验证 ----
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for noisy, clean in test_loader:
                noisy = noisy.to(device)
                clean = clean.to(device)
                outputs = model(noisy)
                loss = criterion(outputs, clean)
                running_val_loss += loss.item() * noisy.size(0)

        epoch_val_loss = running_val_loss / len(test_loader.dataset)
        val_losses.append(epoch_val_loss)

        print(
            f"Epoch {epoch+1}/{num_epochs} "
            f"- train MSE: {epoch_train_loss:.6f}, "
            f"val MSE: {epoch_val_loss:.6f}"
        )

    # 画学习曲线（loss vs epoch）
    plt.figure()
    epochs = range(1, num_epochs + 1)
    plt.plot(epochs, train_losses, label="Train MSE")
    plt.plot(epochs, val_losses, label="Val MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Task 2 - Denoising Autoencoder Learning Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("task2_learning_curve.png", dpi=200)
    plt.close()

    return train_losses, val_losses


def plot_denoising_results(noisy, denoised, clean, n=8,
                           fname="task2_denoising_examples.png"):
    """
    可视化去噪结果：上排 noisy，中排 denoised，下排 clean。
    """
    noisy = noisy[:n].cpu().numpy()
    denoised = denoised[:n].detach().cpu().numpy()
    clean = clean[:n].cpu().numpy()

    fig, axes = plt.subplots(3, n, figsize=(2 * n, 6))

    for i in range(n):
        # noisy
        axes[0, i].imshow(noisy[i, 0], cmap="gray")
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_ylabel("Noisy", fontsize=12)

        # denoised
        axes[1, i].imshow(denoised[i, 0], cmap="gray")
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_ylabel("Denoised", fontsize=12)

        # clean
        axes[2, i].imshow(clean[i, 0], cmap="gray")
        axes[2, i].axis("off")
        if i == 0:
            axes[2, i].set_ylabel("Clean", fontsize=12)

    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close(fig)

def main():
    # 1. DataLoader
    train_loader, test_loader = build_dataloaders(
        root="/Users/tangtang/PycharmProjects/PythonProject1/Assign2/FASHION-MNIST",
        batch_size=128
    )

    # 2. 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model = DenoisingAutoencoder().to(device)

    # 3. 训练
    train_losses, val_losses = train_model(
        model,
        train_loader,
        test_loader,
        device=device,
        num_epochs=10,
        lr=1e-3
    )

    # 4. 在测试集上可视化一些结果
    model.eval()
    noisy_batch, clean_batch = next(iter(test_loader))
    noisy_batch = noisy_batch.to(device)
    clean_batch = clean_batch.to(device)

    with torch.no_grad():
        denoised_batch = model(noisy_batch)

    plot_denoising_results(noisy_batch, denoised_batch, clean_batch, n=8)

if __name__ == "__main__":
    main()
