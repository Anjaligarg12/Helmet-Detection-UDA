import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import glob
import shutil
from torchvision.utils import save_image
from .model import Generator

def adapt_dataset(source_dir, dest_dir, model_path):
    print(f"Adapting dataset from {source_dir} to {dest_dir}...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    G_A2B = Generator().to(device)
    if os.path.exists(model_path):
        G_A2B.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"Warning: Model {model_path} not found. Using untrained generator.")
    G_A2B.eval()
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    inv_normalize = transforms.Normalize(
        mean=[-1.0, -1.0, -1.0],
        std=[2.0, 2.0, 2.0]
    )

    splits = ['train', 'val']
    for split in splits:
        src_img_dir = os.path.join(source_dir, 'images', split)
        src_lbl_dir = os.path.join(source_dir, 'labels', split)
        dst_img_dir = os.path.join(dest_dir, 'images', split)
        dst_lbl_dir = os.path.join(dest_dir, 'labels', split)
        
        os.makedirs(dst_img_dir, exist_ok=True)
        os.makedirs(dst_lbl_dir, exist_ok=True)
        
        files = glob.glob(os.path.join(src_img_dir, '*.*'))
        for idx, f in enumerate(files):
            img = Image.open(f).convert('RGB')
            img_t = transform(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                fake_B = G_A2B(img_t)
            
            # Save adapted image
            fake_B = inv_normalize(fake_B.squeeze(0)).clamp(0, 1)
            filename = os.path.basename(f)
            save_image(fake_B, os.path.join(dst_img_dir, filename))
            
            # Copy label
            lbl_name = os.path.splitext(filename)[0] + '.txt'
            lbl_src = os.path.join(src_lbl_dir, lbl_name)
            if os.path.exists(lbl_src):
                shutil.copy(lbl_src, os.path.join(dst_lbl_dir, lbl_name))
                
            # Save a sample to outputs/samples
            if idx < 3 and split == 'val':
                os.makedirs("outputs/samples", exist_ok=True)
                save_image(fake_B, f"outputs/samples/adapted_{filename}")
                
    print("Domain adaptation complete.")

if __name__ == '__main__':
    adapt_dataset('data/source_india', 'data/adapted', 'outputs/G_A2B.pth')
