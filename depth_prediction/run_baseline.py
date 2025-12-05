import torch
from torch.utils.data import DataLoader, random_split
from models.baseline import RandomClassifierBaseline
from models.zero_model import ZeroBaseline
from data.kitti_selection import KITTIDepthSelectionDataset
from tqdm import tqdm
import os

def scale_invariant_loss(pred, target):
    valid = target > 0
    pred = torch.clamp(pred, min=1e-3)
    target = torch.clamp(target, min=1e-3)
    pred = pred[valid]
    target = target[valid]
    # scale is needed to 
    scale = torch.median(target) / torch.median(pred)
    pred = pred * scale

    d = torch.log(pred+1e-6) - torch.log(target+1e-6)
    return torch.mean(d*d) - (torch.mean(d)**2)

def compute_metrics(pred, target):
    valid = target > 0

    pred = torch.clamp(pred, min=1e-3)
    target = torch.clamp(target, min=1e-3)

    pred = pred[valid]
    target = target[valid]

    # Ignore actual depth values, just determine the distance of pixels relative to each other are correct for now. 
    # i.e we are doing mono depth perception not stereo depth perception. 
    scale = torch.median(target) / torch.median(pred)
    pred = pred * scale

    # SILog
    log_diff = torch.log(pred + 1e-6) - torch.log(target + 1e-6)
    silog = torch.sqrt(torch.mean(log_diff**2) - torch.mean(log_diff)**2)

    # sqErrorRel
    sq_rel = torch.mean(((pred - target)**2) / (target + 1e-6))

    # absErrorRel
    abs_rel = torch.mean(torch.abs(pred - target) / (target + 1e-6))

    # iRMSE
    irmse = torch.sqrt(torch.mean((1.0/(pred + 1e-6) - 1.0/(target + 1e-6))**2))

    return silog.item(), sq_rel.item(), abs_rel.item(), irmse.item()


def evaluate_val_metrics(model, val_loader, device):
    model.eval()
    val_loss = 0
    silog_list = []
    sq_rel_list = []
    abs_rel_list = []
    irmse_list = []

    with torch.no_grad():
        for img, depth in val_loader:
            img, depth = img.to(device), depth.to(device)
            pred = model(img)
            val_loss += scale_invariant_loss(pred, depth).item()

            silog, sq_rel, abs_rel, irmse = compute_metrics(pred, depth)
            silog_list.append(silog)
            sq_rel_list.append(sq_rel)
            abs_rel_list.append(abs_rel)
            irmse_list.append(irmse)

    val_loss /= len(val_loader)
    silog = sum(silog_list) / len(silog_list)
    sq_rel = sum(sq_rel_list) / len(sq_rel_list)
    abs_rel = sum(abs_rel_list) / len(abs_rel_list)
    irmse = sum(irmse_list) / len(irmse_list)

    print(f"Val Loss={val_loss:.4f}, "
        f"SILog={silog:.4f}, "
        f"sqRel={sq_rel:.4f}, "
        f"absRel={abs_rel:.4f}, "
        f"iRMSE={irmse:.4f}")


def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = KITTIDepthSelectionDataset("KITTI/selection")
    train_size = 0
    val_size = len(dataset) - train_size
    _, val_ds = random_split(dataset, [train_size, val_size])


    val_loader   = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)

    model = RandomClassifierBaseline().to(device)

    evaluate_val_metrics(model, val_loader, device)

if __name__ == "__main__":
    run()
