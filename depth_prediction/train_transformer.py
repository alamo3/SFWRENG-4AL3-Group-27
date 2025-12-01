from torch.utils.data import DataLoader

from transformer.dense_prediction_transformer import DensePredictionTransformer
import torch
import torch.nn as nn
import torch.functional as F
from tqdm.auto import tqdm

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

if __name__ == "__main__":
    train_transformer()

