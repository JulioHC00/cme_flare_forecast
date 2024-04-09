from .early_stopping import EarlyStopping
from ..schedulers.mock_scheduler import MockScheduler
from .save_checkpoint import save_checkpoint
from ..utils.merge_dicts import merge_dicts
from ..utils.move_to_device import move_to_device
from .move_df_tensors_to_cpu import (
    convert_df_tensors_to_cpu,
    convert_dict_tensors_to_cpu,
)
from tqdm import tqdm
import torch
import os
import pandas as pd
import time


def train(model_parts, config, checkpoint_dir=None):
    device = config["shared"]["device"]
    model = model_parts["model"]
    optimizer = model_parts["optimizer"]
    scheduler = model_parts["scheduler"]
    logger = model_parts["logger"]
    criterion = model_parts["loss"]

    scheduler_call_on = config["scheduler"]["call_on"]

    logger.log_model_params(model)

    if checkpoint_dir is None:
        checkpoint_dir = "checkpoints/"

    # Check paths exist and create if not
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)

    # Number of epochs
    n_epochs = config["train"]["n_epochs"]

    # Save every n epochs
    checkpoint_every_n_epochs = config["train"]["checkpoint_every_n_epochs"]

    # Load the dataloaders
    dataloaders = model_parts["dataloaders"]
    train_dataloader = dataloaders["train"]
    val_dataloader = dataloaders["val"]
    test_dataloader = dataloaders["test"]

    # Setup early stopping
    do_early_stopping = config["train"]["early_stop"]["enabled"]

    if do_early_stopping:
        early_stopping_args = config["train"]["early_stop"]["args"]

        early_stopping = EarlyStopping(**early_stopping_args)
    else:
        early_stopping = None

    best_epoch = None

    baseline_metrics = val_dataloader.dataset.get_baseline_metrics()
    print("BASELINE METRICS")
    for key, value in baseline_metrics.items():
        print(f"{key}: {value}")

    for epoch in tqdm(range(n_epochs), desc="Epochs", position=0, leave=True):
        # Make sure model is in train mode. Good.
        model.train()

        # Reset ever epoch. Good.
        cum_train_loss = 0
        cum_val_loss = 0

        # Reset validation labels and preds. Good.
        # Must be shape (len(dataloader), 1)
        val_labels = torch.tensor([]).reshape(0).to(device)
        # Must be size (len(dataloader), len(preds))
        val_preds = torch.tensor([]).reshape(0).to(device)

        # Reset the metadata dict. Good.
        metadata_dict = {}

        for batch in tqdm(
            train_dataloader, desc="Training Batches", position=1, leave=True
        ):
            x, y, metadata = batch
            x, y, metadata = (
                x.to(device),
                y.to(device),
                move_to_device(metadata, device),
            )

            # Zero the gradients. Good.
            optimizer.zero_grad()

            model_preds = model(x, metadata)

            loss = criterion(model_preds, y, metadata)
            # Backprop. Good.
            loss.backward()

            # Step the optimizer. Good.
            optimizer.step()

            if scheduler_call_on == "batch":
                scheduler.step()

            cum_train_loss += loss.item()

        # Make sure model is in eval mode. Good.
        model.eval()

        for batch in tqdm(
            val_dataloader, desc="Validation Batches", position=1, leave=True
        ):
            # No need to backprop here. Good.
            with torch.no_grad():
                x, y, metadata = batch
                x, y, metadata = (
                    x.to(device),
                    y.to(device),
                    move_to_device(metadata, device),
                )

                preds = model(x, metadata)
                loss = criterion(preds, y, metadata)

                cum_val_loss += loss.item()

                val_labels = torch.cat((val_labels, y))
                val_preds = torch.cat((val_preds, preds))
                metadata_dict = merge_dicts(metadata_dict, metadata)

        start = time.time()

        val_labels = val_labels.squeeze()
        val_preds = val_preds.squeeze()
        # Add the labels and preds to the metadata dict
        # metadata_dict["labels"] = val_labels.cpu().detach().numpy()
        # metadata_dict["preds"] = val_preds.cpu().detach().numpy()
        metadata_dict["labels"] = val_labels.cpu().detach()
        metadata_dict["preds"] = val_preds.cpu().detach()

        # Convert to pandas dataframe
        # metadata_df = pd.DataFrame.from_dict(metadata_dict, orient="index").transpose()
        # metadata_df = convert_df_tensors_to_cpu(metadata_df)
        metadata_df = metadata_dict
        metadata_df = convert_dict_tensors_to_cpu(metadata_df)

        val_processing_time = time.time()
        val_proessing_took = val_processing_time - start

        print(f"Validation processing took {val_proessing_took:.2f} seconds")

        # Step scheduler
        if scheduler_call_on == "epoch":
            scheduler.step()

        if not isinstance(scheduler, MockScheduler):
            logger.log_learning_rate(scheduler.get_last_lr()[0])

        train_loss = cum_train_loss / len(train_dataloader)
        val_loss = cum_val_loss / len(val_dataloader)

        # Log the epoch
        logger.log_loss(train_loss, "train")
        logger.log_loss(val_loss, "val")

        logging_start = time.time()

        # Log metrics
        logger.log_metrics(val_labels, val_preds, "val")

        # Log plots
        logger.log_plots(metadata_df, "val")

        # Reset the metrics
        logger.reset_metrics()

        logging_end = time.time()

        logging_took = logging_end - logging_start

        print(f"Logging took {logging_took:.2f} seconds")

        if logger.is_currently_best:
            best_epoch = epoch

        # Save checkpoint every n epochs
        if (
            (epoch % checkpoint_every_n_epochs == 0)
            or (epoch == n_epochs - 1)
            or logger.is_currently_best
        ):
            save_checkpoint(
                checkpoint_dir,
                epoch,
                train_loss,
                val_loss,
                model,
                optimizer,
                scheduler,
            )

        # Check for early stopping
        if early_stopping:
            early_stopping(val_loss)
            if early_stopping.stop:
                print("Early stopping")

                save_checkpoint(
                    checkpoint_dir,
                    epoch,
                    train_loss,
                    val_loss,
                    model,
                    optimizer,
                    scheduler,
                )

                break

    # Finish the logger
    logger.finish()

    logger_metrics = logger.get_summary()

    # Combine with baseline metrics
    logger_metrics = merge_dicts(logger_metrics, baseline_metrics)

    return logger_metrics, best_epoch
