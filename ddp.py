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
from model import Net
from torch.cuda.amp import GradScaler
import torch.nn.utils
import matplotlib.pyplot as plt  # 1. Import Matplotlib

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

# Function to calculate accuracy
def simple_accuracy(preds, labels):
    return (preds == labels).mean() * 100

def top_5_accuracy(output, target):
    """
    Compute the top-5 accuracy for classification tasks using model outputs.
    """
    # [Function implementation remains unchanged]
    if isinstance(output, torch.Tensor):
        logits = output.detach()
    else:
        logits = torch.tensor(output)

    if isinstance(target, torch.Tensor):
        targets = target.detach()
    else:
        targets = torch.tensor(target)

    if not logits.is_floating_point():
        logits = logits.float()

    if targets.dtype != torch.long:
        targets = targets.long()

    with torch.no_grad():
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
            targets = targets.unsqueeze(0)

        if logits.dim() != 2:
            raise ValueError(f"Expected logits to be 2D, but got {logits.dim()}D")

        top5_preds = torch.topk(logits, k=5, dim=1).indices
        targets_expanded = targets.view(-1, 1)
        correct = top5_preds.eq(targets_expanded)
        correct_any = correct.any(dim=1).float()
        top5_accuracy = correct_any.mean().item() * 100.0

        return top5_accuracy

class AverageMeter:
    """
    Computes and stores the average and current value.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# Validation function
def validate(model, val_loader, device):
    criterion = nn.CrossEntropyLoss()
    eval_losses = AverageMeter()
    logger.info("***** Running Validation *****")
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            eval_losses.update(loss.item(), inputs.size(0))

            preds = torch.argmax(outputs, dim=-1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    accuracy = simple_accuracy(all_preds, all_labels)
    top5 = top_5_accuracy(all_preds, all_labels)
    logger.info(f"Validation Accuracy: {accuracy:.4f}%, Validation Loss: {eval_losses.avg:.4f}, Top-5 Accuracy: {top5:.4f}%")
    return accuracy, top5

# 2. Define the plotting function
def plot_validation_accuracy(accuracies, output_dir):
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(accuracies) + 1)
    plt.plot(epochs, accuracies, 'b-', marker='o', label='Validation Accuracy')
    plt.title('Validation Accuracy vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'validation_accuracy.png')
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Validation accuracy plot saved at {plot_path}")

# Training function
def train_ddp(rank, world_size):
    # Initialize the process group
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    print(f"Rank {rank} initialized")
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    set_seed(42)  # Set a fixed seed for reproducibility

    # Set hyperparameters
    batch_size = 64
    num_epochs = 350       
    learning_rate = 2e-3 / world_size 
    output_dir = './output'

    # Create output directory if it doesn't exist
    if rank == 0 and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Model setup
    model = Net(num_classes=1000).to(device)  # ImageNet has 1000 classes
    model = DDP(model, device_ids=[rank])

    # Data augmentation and normalization for ImageNet
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_cifar = transforms.Compose([
        transforms.ToTensor()
    ])
    print("Loading data")
    
    train_dataset = datasets.ImageNet(root='/home/ubuntu/ImageNet/', split='train', transform=transform)
    val_dataset = datasets.ImageNet(root='/home/ubuntu/ImageNet/', split='val', transform=transform_test)

    print("Data loaded")
    # train_dataset = datasets.CIFAR10(root='/data/jacob/cifar10/', train=True, download=True, transform=transform_cifar)
    # val_dataset = datasets.CIFAR10(root='/data/jacob/cifar10/', train=False, download=True, transform=transform_cifar)
    
    # Sampler and DataLoader
    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=8, pin_memory=True)

    # Loss, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.1)

    # Mixed Precision Scaler
    scaler = GradScaler()

    # Gradient clipping value
    clip_value = 1.0  # Clip gradients to this value

    # Regularization weight for geodesic loss
    lambda_reg = 0.01  # Weight for the geodesic regularization

    # TensorBoard writer (only for rank 0 process)
    if rank == 0:
        writer = SummaryWriter(log_dir=os.path.join(output_dir, 'logs'))
    
    # Open a log file (only on rank 0)
    if rank == 0:
        file = open(os.path.join(output_dir, "log.txt"), "w")
        val_accuracies = []  # 3. Initialize a list to store validation accuracies

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss = loss
            # Backward pass with gradient scaling for mixed precision
            scaler.scale(total_loss).backward()
            # Update weights
            scaler.step(optimizer)
            scaler.update()

            running_loss += total_loss.item()

        logger.info(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")

        # Validation
        accuracy, accuracy5 = validate(model, val_loader, device)

        # TensorBoard writer and logging (only on rank 0)
        if rank == 0:
            writer.add_scalar('Loss/train', running_loss / len(train_loader), epoch+1)
            writer.add_scalar('Accuracy/val', accuracy, epoch+1)
            logger.info(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}, Accuracy: {accuracy:.4f}, Top5: {accuracy5:.4f}")
            file.write(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}, Accuracy: {accuracy:.4f}, Top5: {accuracy5:.4f}\n")
            file.flush()
            val_accuracies.append(accuracy)  # 4. Collect validation accuracy
            
        # Save checkpoint every 3 epochs
        if (epoch) % 3 == 0:
            save_model(output_dir, model, epoch+1)

        scheduler.step()

    if rank == 0: 
        writer.close()
        file.close()
        # 5. Plot validation accuracy after training
        plot_validation_accuracy(val_accuracies, output_dir)

    dist.destroy_process_group()

# Main function
def main():
    world_size = torch.cuda.device_count()
    rank = int(os.environ['RANK'])  # rank should be set by distributed launcher
    print(f"Rank: {rank}, World Size: {world_size}")
    train_ddp(rank, world_size)

if __name__ == "__main__":
    main()
