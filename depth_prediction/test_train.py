import argparse
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from models.depth_net import DepthNet

def run_single_image(img_path, ckpt_path="checkpoints/selection_trained.pth", out_path="pred_depth.png"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Same preprocessing as training
    img_tf = transforms.Compose([
        transforms.Resize((192, 640)),
        transforms.ToTensor(),
        transforms.Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225]),
    ])

    img = Image.open(img_path).convert("RGB")
    inp = img_tf(img).unsqueeze(0).to(device)  # [1,3,H,W]

    model = DepthNet().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    with torch.no_grad():
        pred = model(inp)[0, 0]  # [H,W]

    # Normalize to 0–255 for visualization
    pred_np = pred.cpu().numpy()
    pred_np = pred_np - pred_np.min()
    pred_np = pred_np / (pred_np.max() + 1e-8)
    pred_vis = (pred_np * 255).astype(np.uint8)
    Image.fromarray(pred_vis).save(out_path)
    print(f"Saved depth visualization to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Run depth prediction on a single image.")
    parser.add_argument("--img", required=True, help="Path to input RGB image (PNG).")
    parser.add_argument("--ckpt", default="checkpoints/selection_trained.pth", help="Path to model checkpoint.")
    parser.add_argument("--out", default="pred_depth.png", help="Path to save the predicted depth visualization PNG.")
    args = parser.parse_args()

    run_single_image(args.img, ckpt_path=args.ckpt, out_path=args.out)

if __name__ == "__main__":
    main()
