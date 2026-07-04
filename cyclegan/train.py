import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import glob
from .model import Generator, Discriminator

class UnpairedDataset(Dataset):
    def __init__(self, root_A, root_B, transform=None):
        self.files_A = sorted(glob.glob(os.path.join(root_A, '*.*')))
        self.files_B = sorted(glob.glob(os.path.join(root_B, '*.*')))
        self.transform = transform

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]).convert('RGB'))
        item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]).convert('RGB'))
        return {'A': item_A, 'B': item_B}

def train_cyclegan(epochs=2):
    print(f"Training CycleGAN for {epochs} epochs...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    G_A2B = Generator().to(device)
    G_B2A = Generator().to(device)
    D_A = Discriminator().to(device)
    D_B = Discriminator().to(device)
    
    criterion_GAN = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()
    
    optimizer_G = optim.Adam(list(G_A2B.parameters()) + list(G_B2A.parameters()), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D_A = optim.Adam(D_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D_B = optim.Adam(D_B.parameters(), lr=0.0002, betas=(0.5, 0.999))

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = UnpairedDataset('data/source_india/images/train', 'data/target_sea/images/train', transform)
    if len(dataset) == 0:
        print("No data found for CycleGAN training.")
        return
        
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    for epoch in range(epochs):
        loss_G_val = 0
        loss_D_val = 0
        for i, batch in enumerate(dataloader):
            real_A = batch['A'].to(device)
            real_B = batch['B'].to(device)
            
            optimizer_G.zero_grad()
            
            loss_id_A = criterion_identity(G_B2A(real_A), real_A)
            loss_id_B = criterion_identity(G_A2B(real_B), real_B)
            loss_identity = (loss_id_A + loss_id_B) / 2
            
            fake_B = G_A2B(real_A)
            loss_GAN_A2B = criterion_GAN(D_B(fake_B), torch.ones_like(D_B(fake_B)))
            
            fake_A = G_B2A(real_B)
            loss_GAN_B2A = criterion_GAN(D_A(fake_A), torch.ones_like(D_A(fake_A)))
            loss_GAN = (loss_GAN_A2B + loss_GAN_B2A) / 2
            
            recov_A = G_B2A(fake_B)
            loss_cycle_A = criterion_cycle(recov_A, real_A)
            recov_B = G_A2B(fake_A)
            loss_cycle_B = criterion_cycle(recov_B, real_B)
            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2
            
            loss_G = loss_GAN + 10.0 * loss_cycle + 5.0 * loss_identity
            loss_G.backward()
            optimizer_G.step()
            
            optimizer_D_A.zero_grad()
            loss_real = criterion_GAN(D_A(real_A), torch.ones_like(D_A(real_A)))
            loss_fake = criterion_GAN(D_A(fake_A.detach()), torch.zeros_like(D_A(fake_A.detach())))
            loss_D_A = (loss_real + loss_fake) / 2
            loss_D_A.backward()
            optimizer_D_A.step()
            
            optimizer_D_B.zero_grad()
            loss_real = criterion_GAN(D_B(real_B), torch.ones_like(D_B(real_B)))
            loss_fake = criterion_GAN(D_B(fake_B.detach()), torch.zeros_like(D_B(fake_B.detach())))
            loss_D_B = (loss_real + loss_fake) / 2
            loss_D_B.backward()
            optimizer_D_B.step()
            
            loss_G_val = loss_G.item()
            loss_D_val = (loss_D_A + loss_D_B).item()
            
        print(f"[Epoch {epoch+1}/{epochs}] G_loss: {loss_G_val:.4f} D_loss: {loss_D_val:.4f}")
    
    os.makedirs('outputs', exist_ok=True)
    torch.save(G_A2B.state_dict(), 'outputs/G_A2B.pth')
    print("CycleGAN training complete.")

if __name__ == '__main__':
    train_cyclegan(epochs=2)
