from torch.utils.data import DataLoader

from transformer.dense_prediction_transformer import DensePredictionTransformer
import torch
import torch.nn as nn
import torch.functional as F
from tqdm.auto import tqdm
import argparse
import numpy as np

from train_selection import scale_invariant_loss
from data.kitti_loader import KITTIDepthLoader



model = DensePredictionTransformer(transformer_features=256)

def get_loaders():
    pass



def train_transformer():

    BATCH_SIZE = 1
    LR = 1e-4
    EPOCHS = 100


    global model

    dataset_train = KITTIDepthLoader("D:\\KITTI\\train")
    ds_size = len(dataset_train)

    indices = list(range(ds_size))
    np.random.shuffle(indices)

    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices)

    train_loader = DataLoader(dataset=dataset_train, shuffle=False, batch_size=BATCH_SIZE, sampler=train_sampler,
                              num_workers=4, pin_memory=True)

    optimer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = scale_invariant_loss

    model.train()

    for epoch in tqdm(range(EPOCHS)):
        tqdm_dataloader = tqdm(train_loader)
        for batch_idx, (inputs, targets) in enumerate(tqdm_dataloader):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimer.zero_grad()
            loss.backward()
            optimer.step()

            tqdm_dataloader.write(f"Epoch {epoch} Batch {batch_idx} Loss {loss.item():.4f}")



    torch.save(model.state_dict(), "transformer_model.pth")

def train_transformer_acceleratred():
    from accelerate import Accelerator  # <--- The Magic Tool

    global model

    # This automatically detects GPUs and sets up BF16 precision
    accelerator = Accelerator(mixed_precision="bf16")


    BATCH_SIZE = 8  # Per GPU. Total Batch Size = 8 * 8 = 64
    LR = 1e-4
    EPOCHS = 20

    dataset_train = KITTIDepthLoader("D:\\KITTI\\train")
    ds_size = len(dataset_train)

    indices = list(range(ds_size))
    np.random.shuffle(indices)

    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices)

    train_loader = DataLoader(dataset=dataset_train, shuffle=False, batch_size=BATCH_SIZE, sampler=train_sampler,
                              num_workers=4, pin_memory=True)

    optimer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = scale_invariant_loss

    # D. Prepare everything with Accelerator
    # It wraps your model in DDP and handles device placement
    model, optimizer, dataloader = accelerator.prepare(model, optimer, train_loader)

    # E. Training Loop
    model.train()

    for epoch in tqdm(range(EPOCHS)):
        tqdm_dataloader = tqdm(train_loader)
        for step, (inputs, targets) in enumerate(tqdm_dataloader):
            inputs = inputs.to(accelerator.device)
            targets = targets.to(accelerator.device)

            preds = model(inputs)
            loss = criterion(preds, targets)

            optimizer.zero_grad()
            accelerator.backward(loss)  # <--- Replace loss.backward()
            optimizer.step()

            if step % 10 == 0 and accelerator.is_main_process:
                accelerator.print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")

    if accelerator.is_main_process:
        print("Training finished.")
        # Unwrap and save
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save(unwrapped_model.state_dict(), "dpt_kitti_l40s.pth")



if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument("--accelerate", action="store_true", help="Use accelerate to train the model")
    args = args.parse_args()

    if args.accelerate:
        train_transformer_acceleratred()
    else:
        train_transformer()

