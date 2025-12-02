from torch.utils.data import DataLoader

from transformer.dense_prediction_transformer import DensePredictionTransformer
import torch
import torch.nn as nn
import torch.functional as F
from tqdm.auto import tqdm
import argparse
import numpy as np

from train_selection import scale_invariant_loss
from train_selection import compute_metrics
from data.kitti_loader import KITTIDepthLoader



model = DensePredictionTransformer(transformer_features=256)

def get_loaders():
    pass

silog_per_epoch = []
sq_rel_per_epoch = []
abs_rel_per_epoch = []
irmse_per_epoch = []
val_loss_per_epoch = []
train_loss_per_epoch = []

def evaluate_val_metrics(model_val, val_loader, device):
    model_val.eval()
    val_loss = 0
    silog_list = []
    sq_rel_list = []
    abs_rel_list = []
    irmse_list = []

    with torch.no_grad():
        tqdm_loader = tqdm(val_loader)
        for step, (inputs, targets) in enumerate(tqdm_loader):
            img, depth = inputs.to(device), targets.to(device)
            pred = model_val(img)

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

    return val_loss, silog, sq_rel, abs_rel, irmse


def train_transformer(train_dir, test_dir):

    BATCH_SIZE = 1
    LR_ENCODER = 5e-6 ## We want to fine tune the transformer encoder
    LR_DECODER = 1e-4 ## We want to train the decoder from scratch
    EPOCHS = 20

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    global model
    model.to(device)

    encoder_params = list(map(id, model.encoder.parameters()))
    decoder_params = filter(lambda p: id(p) not in encoder_params, model.parameters())

    dataset_train = KITTIDepthLoader(train_dir)
    dataset_val = KITTIDepthLoader(train_dir)
    ds_size = len(dataset_train)

    indices = list(range(ds_size))
    np.random.shuffle(indices)

    val_split_index = int(np.floor(0.2 * ds_size))
    train_idx, val_idx = indices[val_split_index:], indices[:val_split_index]

    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
    val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_idx)

    train_loader = DataLoader(dataset=dataset_train, shuffle=False, batch_size=BATCH_SIZE, sampler=train_sampler,
                              num_workers=4, pin_memory=True)

    val_loader = DataLoader(dataset=dataset_val, shuffle=False, batch_size=BATCH_SIZE,
                            sampler=val_sampler,num_workers=4, pin_memory=True)

    optimizer = torch.optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': LR_ENCODER},
        {'params': decoder_params, 'lr': LR_DECODER}
    ], weight_decay=1e-2)

    criterion = scale_invariant_loss

    # Learning rate scheduler (Cosine annealing w/ One Cycle policy)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[LR_ENCODER, LR_DECODER],
        steps_per_epoch=len(train_loader),
        epochs=EPOCHS,
        anneal_strategy="cos"
    )


    lowest_val_loss = float("inf")
    for epoch in tqdm(range(EPOCHS)):
        model.train()
        tqdm_dataloader = tqdm(train_loader)
        train_loss_this_epoch = 0

        for batch_idx, (inputs, targets) in enumerate(tqdm_dataloader):

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss_this_epoch += loss.item()

            tqdm_dataloader.write(f"Epoch {epoch} Batch {batch_idx} Loss {loss.item():.4f}")

        # Now we calculate validation loss
        val_loss, silog, sq_rel, abs_rel, irmse = evaluate_val_metrics(model, val_loader, device)
        tqdm_dataloader.write(f"Epoch {epoch} Validation Loss {val_loss:.4f} SILog {silog:.4f} SqRel {sq_rel:.4f} AbsRel {abs_rel:.4f} IRMSE {irmse:.4f}")

        train_loss_this_epoch = train_loss_this_epoch / len(train_loader)

        train_loss_per_epoch.append(train_loss_this_epoch)
        val_loss_per_epoch.append(val_loss)
        silog_per_epoch.append(silog)
        sq_rel_per_epoch.append(sq_rel)
        abs_rel_per_epoch.append(abs_rel)
        irmse_per_epoch.append(irmse)

        #Save the current lists to file
        np.save("train_loss_per_epoch.npy", np.array(train_loss_per_epoch))
        np.save("val_loss_per_epoch.npy", np.array(val_loss_per_epoch))
        np.save("silog_per_epoch.npy", np.array(silog_per_epoch))
        np.save("sq_rel_per_epoch.npy", np.array(sq_rel_per_epoch))
        np.save("abs_rel_per_epoch.npy", np.array(abs_rel_per_epoch))
        np.save("irmse_per_epoch.npy", np.array(irmse_per_epoch))

        torch.save(model.state_dict(), f"checkpoints/transformer_model_epoch_{epoch}.pth")

        if val_loss < lowest_val_loss:
            lowest_val_loss = val_loss
            torch.save(model.state_dict(), "checkpoints/transformer_model_best.pth")






def train_transformer_accelerated(train_dir, test_dir):
    from accelerate import Accelerator  # We are training in parallel baby

    global model

    # This automatically detects GPUs and sets up BF16 precision
    accelerator = Accelerator(mixed_precision="bf16")

    encoder_params = list(map(id, model.encoder.parameters()))
    decoder_params = filter(lambda p: id(p) not in encoder_params, model.parameters())

    BATCH_SIZE = 1
    LR_ENCODER = 5e-6  ## We want to fine tune the transformer encoder
    LR_DECODER = 1e-4  ## We want to train the decoder from scratch
    EPOCHS = 20

    dataset_train = KITTIDepthLoader(train_dir)

    ds_size = len(dataset_train)
    val_size = int(np.floor(0.2 * ds_size))

    train_size = ds_size - val_size

    ## We cannot do the index splitting technique when using accelerate
    generator = torch.Generator().manual_seed(23)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset_train, [train_size, val_size], generator=generator)

    train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=BATCH_SIZE,
                              num_workers=4, pin_memory=True)

    val_loader = DataLoader(dataset=val_dataset, shuffle=False, batch_size=BATCH_SIZE,
                             num_workers=4, pin_memory=True)

    optimizer = torch.optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': LR_ENCODER},
        {'params': decoder_params, 'lr': LR_DECODER}
    ], weight_decay=1e-2)

    criterion = scale_invariant_loss

    # Learning rate scheduler (Cosine annealing w/ One Cycle policy)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[LR_ENCODER, LR_DECODER],
        steps_per_epoch=len(train_loader),
        epochs=EPOCHS,
        anneal_strategy="cos"
    )

    # D. Prepare everything with Accelerator

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )
    lowest_val_loss = float("inf")
    for epoch in tqdm(range(EPOCHS), disable=not accelerator.is_local_main_process):
        model.train()
        tqdm_dataloader = tqdm(train_loader, disable=not accelerator.is_local_main_process)
        train_loss_this_epoch = 0
        for step, (inputs, targets) in enumerate(tqdm_dataloader):

            preds = model(inputs)
            loss = criterion(preds, targets)

            optimizer.zero_grad()
            accelerator.backward(loss)  # <--- Replace loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss_this_epoch += loss.item()

            if step % 10 == 0 and accelerator.is_main_process:
                accelerator.print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")

        accelerator.wait_for_everyone()


        val_loss, silog, sq_rel, abs_rel, irmse = evaluate_val_metrics(model, val_loader, accelerator.device)
        metrics = torch.tensor([val_loss, silog, sq_rel, abs_rel, irmse], device=accelerator.device)

        metrics = metrics.unsqueeze(0)
        gathered_metrics = accelerator.gather(metrics)

        avg_metrics = gathered_metrics.mean(dim=0)

        val_loss, silog, sq_rel, abs_rel, irmse = avg_metrics.tolist()


        if accelerator.is_main_process:
            train_loss_this_epoch = train_loss_this_epoch / len(train_loader)
            tqdm_dataloader.write(
                f"Epoch {epoch} Validation Loss {val_loss:.4f} SILog {silog:.4f} SqRel {sq_rel:.4f} AbsRel {abs_rel:.4f} IRMSE {irmse:.4f}")


            train_loss_per_epoch.append(train_loss_this_epoch)
            val_loss_per_epoch.append(val_loss)
            silog_per_epoch.append(silog)
            sq_rel_per_epoch.append(sq_rel)
            abs_rel_per_epoch.append(abs_rel)
            irmse_per_epoch.append(irmse)

            # Save the current lists to file
            np.save("train_loss_per_epoch.npy", np.array(train_loss_per_epoch))
            np.save("val_loss_per_epoch.npy", np.array(val_loss_per_epoch))
            np.save("silog_per_epoch.npy", np.array(silog_per_epoch))
            np.save("sq_rel_per_epoch.npy", np.array(sq_rel_per_epoch))
            np.save("abs_rel_per_epoch.npy", np.array(abs_rel_per_epoch))
            np.save("irmse_per_epoch.npy", np.array(irmse_per_epoch))

            unwrapped_model = accelerator.unwrap_model(model)
            torch.save(unwrapped_model.state_dict(), f"checkpoints/transformer_model_epoch_{epoch}.pth")

            if val_loss < lowest_val_loss:
                lowest_val_loss = val_loss
                torch.save(unwrapped_model.state_dict(), "checkpoints/transformer_model_best.pth")

        accelerator.wait_for_everyone()







if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("train_dir", help="Directory for training data")
    args.add_argument("test_dir", help="Directory for test data")
    args.add_argument("--accelerate", action="store_true", help="Use accelerate to train the model")
    args = args.parse_args()

    if args.accelerate:
        train_transformer_accelerated(args.train_dir, args.test_dir)
    else:
        train_transformer(args.train_dir, args.test_dir)

