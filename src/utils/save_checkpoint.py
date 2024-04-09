import torch


def save_checkpoint(
    checkpoint_dir, epoch, train_loss, val_loss, model, optimizer, scheduler
):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "other_metadata": "any_additional_information",
    }

    save_path = checkpoint_dir + f"checkpoint_{epoch}.pth"

    torch.save(checkpoint, save_path)
