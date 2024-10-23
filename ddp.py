import os
import random
import numpy as np
import logging
import torch
import torch.distributed as dist
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import Net  # Ensure this imports your actual model
from torch.cuda.amp import autocast, GradScaler
import torch.nn.utils

# Set up logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Set random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Function to save the model
def save_model(output_dir, model, epoch):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(output_dir, f"checkpoint_epoch_{epoch}.pth")
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info(f"Model checkpoint saved at {model_checkpoint}")

def load_checkpoint(model, checkpoint_path):
    if os.path.isfile(checkpoint_path):
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint)
        logger.info("Checkpoint loaded successfully.")
    else:
        logger.info(f"No checkpoint found at: {checkpoint_path}")
    return model

class AverageMeter:
    """
    Computes and stores the average and current value.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0  # Average
        self.sum = 0  # Sum of values
        self.count = 0  # Number of updates

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the top-k predictions for specified values of k.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # Get top-k predictions
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()  # Shape: [maxk, batch_size]
        correct = pred.eq(target.view(1, -1).expand_as(pred))  # Shape: [maxk, batch_size]

        res = []
        for k in topk:
            # For each k, calculate the number of correct predictions
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# Validation function
def validate(model, val_loader, device):
    criterion = nn.CrossEntropyLoss()
    eval_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    logger.info("***** Running Validation *****")
    model.eval()
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validating"):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            # Add autocast for mixed precision
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            eval_losses.update(loss.item(), inputs.size(0))

            # Measure accuracy
            prec1, prec5 = accuracy(outputs, labels, topk=(1, 5))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

    logger.info(f"Validation Accuracy: {top1.avg:.4f}%, Validation Loss: {eval_losses.avg:.4f}, Top-5 Accuracy: {top5.avg:.4f}%")
    return top1.avg, top5.avg

# Training function
def train_ddp(rank, world_size):
    # Initialize the process group
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    set_seed(42)  # Set a fixed seed for reproducibility

    # Set hyperparameters
    batch_size = 64
    num_epochs = 350
    learning_rate = 1e-3
    output_dir = './output'

    # Ensure the output directory exists
    if rank == 0 and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Model setup
    model = Net(num_classes=1000).to(device)  # Replace with your actual model
    model = load_checkpoint(model, "./output/checkpoint_epoch_1.pth")
    model = DDP(model, device_ids=[rank], output_device=rank)

    # Data augmentation and normalization for ImageNet
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


    # Datasets
    train_dataset = datasets.ImageNet(root='/data/jacob/ImageNet/', split='train', transform=transform)
    val_dataset = datasets.ImageNet(root='/data/jacob/ImageNet/', split='val', transform=transform_test)

    # Sampler and DataLoader
    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=8, pin_memory=True)

    # Loss, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # Mixed Precision Scaler
    scaler = GradScaler()

    # TensorBoard writer and log file (only for rank 0 process)
    if rank == 0:
        writer = SummaryWriter(log_dir=os.path.join(output_dir, 'logs'))
        file = open(os.path.join(output_dir, "log.txt"), "w")

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_sampler.set_epoch(epoch)

        # Initialize running loss and correct counts
        running_loss = 0.0
        total_correct = 0
        total_samples = 0

        # Create tqdm iterator for rank 0
        if rank == 0:
            tqdm_loader = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", ncols=100)
        else:
            tqdm_loader = train_loader

        for inputs, labels in tqdm_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            # Forward pass with autocast
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # Update running loss and counts
            with torch.no_grad():
                running_loss += loss.item() * inputs.size(0)
                _, preds = outputs.topk(1, 1, True, True)
                preds = preds.t()
                correct = preds.eq(labels.view(1, -1).expand_as(preds))
                total_correct += correct[:1].reshape(-1).float().sum(0).item()
                total_samples += labels.size(0)

            # Update progress bar
            if rank == 0:
                current_loss = running_loss / total_samples
                current_acc = 100.0 * total_correct / total_samples
                tqdm_loader.set_postfix({'Loss': f'{current_loss:.4f}', 'Acc': f'{current_acc:.2f}%'})

        # After epoch, aggregate metrics
        running_loss_tensor = torch.tensor(running_loss, device=device)
        total_correct_tensor = torch.tensor(total_correct, device=device)
        total_samples_tensor = torch.tensor(total_samples, device=device)

        dist.reduce(running_loss_tensor, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(total_correct_tensor, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(total_samples_tensor, dst=0, op=dist.ReduceOp.SUM)

        if rank == 0:
            average_loss = running_loss_tensor.item() / total_samples_tensor.item()
            average_acc = (total_correct_tensor.item() / total_samples_tensor.item()) * 100.0
            logger.info(f"Epoch {epoch+1}, Loss: {average_loss:.4f}, Training Accuracy: {average_acc:.2f}%")

        # Validation
        accuracy1, accuracy5 = validate(model, val_loader, device)

        # TensorBoard writer and log file
        if rank == 0:
            writer.add_scalar('Loss/train', average_loss, epoch+1)
            writer.add_scalar('Accuracy/train_top1', average_acc, epoch+1)
            writer.add_scalar('Accuracy/val_top1', accuracy1, epoch+1)
            writer.add_scalar('Accuracy/val_top5', accuracy5, epoch+1)
            logger.info(f"Epoch {epoch+1}, Loss: {average_loss:.4f}, "
                        f"Training Accuracy: {average_acc:.2f}%, Validation Top-1 Accuracy: {accuracy1:.2f}%, "
                        f"Top-5 Accuracy: {accuracy5:.2f}%")
            file.write(f"Epoch {epoch+1}, Loss: {average_loss:.4f}, "
                       f"Training Accuracy: {average_acc:.2f}%, Validation Top-1 Accuracy: {accuracy1:.2f}%, "
                       f"Top-5 Accuracy: {accuracy5:.2f}%\n")
            file.flush()

            # Update tqdm description
            tqdm_loader.set_description(f"Epoch {epoch+1}/{num_epochs} [Val Acc@1: {accuracy1:.2f}%]")

        # Save checkpoint every 2 epochs
        if rank == 0 and (epoch + 1) % 2 == 0:
            save_model(output_dir, model, epoch + 1)

        scheduler.step()

    if rank == 0:
        writer.close()
        file.close()

    dist.destroy_process_group()

# Main function
def main():
    world_size = torch.cuda.device_count()
    rank = int(os.environ['RANK'])  # Rank should be set by distributed launcher
    train_ddp(rank, world_size)

if __name__ == "__main__":
    main()
