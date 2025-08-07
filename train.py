import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import pickle
import json
import math
from PIL import Image
import torchvision.transforms as T

class PixelProjectionLayer(nn.Module):
    def __init__(self, input_dim=3, output_dim=256):
        super().__init__()
        self.proj = nn.Conv2d(input_dim, output_dim, kernel_size=1)
    def forward(self, x):
        return self.proj(x)

class FourierPositionalEncoding(nn.Module):
    def __init__(self, hidden_dim: int, height: int, width: int):
        super().__init__()
        if hidden_dim % 4 != 0: raise ValueError(f"hidden_dim {hidden_dim} must be divisible by 4")
        num_bands = hidden_dim // 4
        bands = torch.pow(2.0, torch.arange(num_bands, dtype=torch.float32))
        y_pos, x_pos = torch.linspace(-1, 1, height, dtype=torch.float32), torch.linspace(-1, 1, width, dtype=torch.float32)
        y_args, x_args = y_pos[:, None] * bands[None, :], x_pos[:, None] * bands[None, :]
        y_emb, x_emb = torch.cat([torch.sin(y_args), torch.cos(y_args)], dim=-1), torch.cat([torch.sin(x_args), torch.cos(x_args)], dim=-1)
        y_emb_expanded, x_emb_expanded = y_emb.unsqueeze(1).expand(-1, width, -1), x_emb.unsqueeze(0).expand(height, -1, -1)
        pos_encoding = torch.cat([y_emb_expanded, x_emb_expanded], dim=-1)
        self.register_buffer('pos_encoding', pos_encoding.permute(2, 0, 1).unsqueeze(0), persistent=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos_encoding

class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim, n_heads):
        super().__init__()
        self.cross_attn, self.self_attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True), nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(embed_dim, embed_dim * 4), nn.GELU(), nn.Linear(embed_dim * 4, embed_dim))
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(embed_dim), nn.LayerNorm(embed_dim), nn.LayerNorm(embed_dim)
    def forward(self, query, mask_pixels):
        attn_output, _ = self.cross_attn(query=query, key=mask_pixels, value=mask_pixels); query = self.norm1(query + attn_output)
        attn_output, _ = self.self_attn(query=query, key=query, value=query); query = self.norm2(query + attn_output)
        ffn_output = self.ffn(query); query = self.norm3(query + ffn_output)
        return query

class MaskTokenizer(nn.Module):
    def __init__(self, embed_dim=256, n_heads=8, n_queries=4, n_layers=1):
        super().__init__()
        self.learnable_queries, self.decoder_layers, self.embed_dim = nn.Parameter(torch.randn(1, n_queries, embed_dim)), nn.ModuleList([TransformerDecoderLayer(embed_dim, n_heads) for _ in range(n_layers)]), embed_dim
    def forward(self, all_pixel_features, mask):
        mask_pixels = all_pixel_features[mask]
        if mask_pixels.shape[0] == 0: return torch.zeros(1, self.embed_dim, device=all_pixel_features.device)
        queries = self.learnable_queries.expand(1, -1, -1)
        for layer in self.decoder_layers: queries = layer(queries, mask_pixels.unsqueeze(0))
        return queries.mean(dim=1)

class Projector(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__(); self.fc1, self.activation, self.fc2, self.layer_norm = nn.Linear(input_dim, (input_dim + output_dim) // 2), nn.GELU(), nn.Linear((input_dim + output_dim) // 2, output_dim), nn.LayerNorm(output_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.layer_norm(self.fc2(self.activation(self.fc1(x))))

class AnomalyDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.variant_folders = sorted([os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    def __len__(self): return len(self.variant_folders)
    def __getitem__(self, idx):
        folder_path = self.variant_folders[idx]
        image = self.transform(Image.open(os.path.join(folder_path, "image.png")).convert("RGB"))
        with open(os.path.join(folder_path, "original_masks.pkl"), "rb") as f: masks = pickle.load(f)
        with open(os.path.join(folder_path, "labels.json"), "r") as f: labels = json.load(f)
        return image, masks, labels

def custom_collate_fn(batch):
    images = torch.stack([item[0] for item in batch], dim=0)
    masks = [item[1] for item in batch]
    labels = [item[2] for item in batch]
    return images, masks, labels

def train():
    TRAIN_CONFIG = {"data_root_dir": "/home/s2behappy4/data/gyuhyeong/MLLM_Anomaly/Demo_data/", "target_embedding_path": "target_embeddings_grid.pt", "image_size": 512, "embed_dim": 256, "llm_hidden_dim": 4096, "batch_size": 8, "learning_rate": 1e-4, "epochs": 20, "checkpoint_path": "/home/s2behappy4/data/gyuhyeong/MLLM_Anomaly/checkpoints/latest_checkpoint.pth"}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Starting Training on {device} ---")

    pixel_proj_layer = PixelProjectionLayer(output_dim=TRAIN_CONFIG["embed_dim"]).to(device)
    pos_encoding = FourierPositionalEncoding(TRAIN_CONFIG["embed_dim"], TRAIN_CONFIG["image_size"], TRAIN_CONFIG["image_size"]).to(device)
    mask_tokenizer = MaskTokenizer(embed_dim=TRAIN_CONFIG["embed_dim"]).to(device)
    projector = Projector(input_dim=TRAIN_CONFIG["embed_dim"], output_dim=TRAIN_CONFIG["llm_hidden_dim"]).to(device)

    params_to_train = list(pixel_proj_layer.parameters()) + list(mask_tokenizer.parameters()) + list(projector.parameters())
    optimizer = torch.optim.Adam(params_to_train, lr=TRAIN_CONFIG["learning_rate"])

    start_epoch = 0
    if os.path.exists(TRAIN_CONFIG["checkpoint_path"]):
        print(f"Loading checkpoint from {TRAIN_CONFIG['checkpoint_path']}")
        checkpoint = torch.load(TRAIN_CONFIG["checkpoint_path"])
        pixel_proj_layer.load_state_dict(checkpoint['pixel_proj_layer_state_dict'])
        mask_tokenizer.load_state_dict(checkpoint['mask_tokenizer_state_dict'])
        projector.load_state_dict(checkpoint['projector_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}")
    
    target_embeddings = torch.load(TRAIN_CONFIG["target_embedding_path"]); target_normal_emb, target_anomaly_emb = target_embeddings["normal"].to(device), target_embeddings["anomaly"].to(device)
    dataset = AnomalyDataset(TRAIN_CONFIG["data_root_dir"]); dataloader = DataLoader(dataset, batch_size=TRAIN_CONFIG["batch_size"], shuffle=True, num_workers=4, pin_memory=True, collate_fn=custom_collate_fn)
    loss_fn = nn.CosineEmbeddingLoss()

    for epoch in range(start_epoch, TRAIN_CONFIG["epochs"]):
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{TRAIN_CONFIG['epochs']}")
        for batch_images, batch_masks, batch_labels in loop:
            
            batch_images = batch_images.to(device); optimizer.zero_grad()
            pixel_features_base = pixel_proj_layer(batch_images); pixel_features_pos = pos_encoding(pixel_features_base).permute(0, 2, 3, 1)
            total_loss = 0
            for i in range(len(batch_images)):
                image_features, masks, labels = pixel_features_pos[i], batch_masks[i], batch_labels[i]
                for mask_idx_str, label in labels.items():
                    mask = torch.from_numpy(masks[int(mask_idx_str)]).to(device)
                    predicted_embedding = projector(mask_tokenizer(image_features, mask))
                    target_embedding = target_normal_emb if label == "normal" else target_anomaly_emb
                    loss_target = torch.ones(predicted_embedding.shape[0]).to(device)
                    total_loss += loss_fn(predicted_embedding, target_embedding.unsqueeze(0), loss_target)
            if isinstance(total_loss, torch.Tensor): total_loss.backward(); optimizer.step(); loop.set_postfix(loss=total_loss.item())

        print(f"Epoch {epoch+1} finished. Saving checkpoint...")
        os.makedirs(os.path.dirname(TRAIN_CONFIG["checkpoint_path"]), exist_ok=True)
        torch.save({'epoch': epoch, 'pixel_proj_layer_state_dict': pixel_proj_layer.state_dict(),'mask_tokenizer_state_dict': mask_tokenizer.state_dict(),'projector_state_dict': projector.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': total_loss}, TRAIN_CONFIG["checkpoint_path"])
        print("Checkpoint saved.")
    
    print("--- Training finished. ---")

if __name__ == "__main__":
    train()