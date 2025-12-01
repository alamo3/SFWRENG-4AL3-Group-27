from transformer.dense_prediction_transformer import DensePredictionTransformer
import torch
import torch.nn as nn
import torch.functional as F
from tqdm import tqdm

from train_selection import scale_invariant_loss



model = DensePredictionTransformer(transformer_features=256)

def train_transformer():

    BATCH_SIZE = 1
    LR = 1e-4
    EPOCHS = 100


    global model

    dataloader = [(torch.randn(BATCH_SIZE, 3, 352, 1216), torch.rand(BATCH_SIZE, 1, 352, 1216)) for _ in range(100)]

    optimer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = scale_invariant_loss

    model.train()

    for epoch in tqdm(range(EPOCHS)):

        for batch_idx, (inputs, targets) in enumerate(tqdm(dataloader)):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimer.zero_grad()
            loss.backward()
            optimer.step()

        print(f"Epoch: {epoch}, Loss: {loss.item()}")


    torch.save(model.state_dict(), "transformer_model.pth")

if __name__ == "__main__":
    train_transformer()

