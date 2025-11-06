import os
import csv
from tqdm import tqdm
import torch
import argparse
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset


class MiniPlaces(Dataset):
    def __init__(self, root_dir, split, transform=None, label_dict=None):
        """
        Initialize the MiniPlaces dataset with the root directory for the images,
        the split (train/val/test), an optional data transformation,
        and an optional label dictionary.

        Args:
            root_dir (str): Root directory for the MiniPlaces images.
            split (str): Split to use ('train', 'val', or 'test').
            transform (callable, optional): Optional data transformation to apply to the images.
            label_dict (dict, optional): Optional dictionary mapping integer labels to class names.
        """
        assert split in ['train', 'val', 'test']
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.filenames = []
        self.labels = []

        self.label_dict = label_dict if label_dict is not None else {}

        with open(os.path.join(self.root_dir, self.split + '.txt')) as r:
            lines = r.readlines()
            for line in lines:
                line = line.split()
                self.filenames.append(line[0])
                if split == 'test':
                    label = line[0]
                else:
                    label = int(line[1])
                self.labels.append(label)
                if split == 'train':
                    text_label = line[0].split(os.sep)[2]
                    self.label_dict[label] = text_label
                

    def __len__(self):
        """
        Return the number of images in the dataset.

        Returns:
            int: Number of images in the dataset.
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Return a single image and its corresponding label when given an index.

        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            tuple: Tuple containing the image and its label.
        """
        if self.transform is not None:
            image = self.transform(
                Image.open(os.path.join(self.root_dir, "images", self.filenames[idx])))
        else:
                image = Image.open(os.path.join(self.root_dir, "images", self.filenames[idx]))
        label = self.labels[idx]
        return image, label    

class MyConv(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        
        # Initial convolution with larger kernel for better feature extraction
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # More residual blocks with increased capacity
        self.conv2 = nn.Sequential(
            self._make_residual_block(64, 128, stride=1),
            self._make_residual_block(128, 128, stride=1),
            self._make_residual_block(128, 128, stride=1)  # Added extra block
        )
        
        self.conv3 = nn.Sequential(
            self._make_residual_block(128, 256, stride=2),
            self._make_residual_block(256, 256, stride=1),
            self._make_residual_block(256, 256, stride=1),  # Added extra block
            self._make_residual_block(256, 256, stride=1)   # Added extra block
        )
        
        self.conv4 = nn.Sequential(
            self._make_residual_block(256, 512, stride=2),
            self._make_residual_block(512, 512, stride=1),
            self._make_residual_block(512, 512, stride=1)   # Added extra block
        )
        
        # Additional deeper layer for more complex features
        self.conv5 = nn.Sequential(
            self._make_residual_block(512, 1024, stride=2),
            self._make_residual_block(1024, 1024, stride=1)
        )
        
        # Combined channel and spatial attention mechanism
        self.attention = CBAM(1024)
        
        # Global average pooling instead of flatten for spatial invariance
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # More balanced classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),  # Reduced dropout
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),  # Reduced dropout
            nn.Linear(512, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_residual_block(self, in_channels, out_channels, stride=1):
        """Create a residual block with skip connection"""
        layers = []
        
        # Main path
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        
        main_path = nn.Sequential(*layers)
        
        # Skip connection
        if stride != 1 or in_channels != out_channels:
            skip_connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            skip_connection = nn.Identity()
        
        return ResidualBlock(main_path, skip_connection)
    
    def _initialize_weights(self):
        """Initialize network weights using He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x, return_intermediate=False):
        # Store intermediate features if requested
        intermediates = []
        
        # Feature extraction
        x = self.conv1(x)
        if return_intermediate: intermediates.append(x)
        
        x = self.conv2(x)
        if return_intermediate: intermediates.append(x)
        
        x = self.conv3(x)
        if return_intermediate: intermediates.append(x)
        
        x = self.conv4(x)
        if return_intermediate: intermediates.append(x)
        
        x = self.conv5(x)
        if return_intermediate: intermediates.append(x)
        
        # Apply combined attention
        x = self.attention(x)
        
        # Global pooling and classification
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        if return_intermediate:
            return x, intermediates
        return x


class ChannelAttention(nn.Module):
    """Channel attention mechanism to focus on important feature channels"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)


class SpatialAttention(nn.Module):
    """Spatial attention mechanism to focus on important spatial locations"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(attention)
        return x * self.sigmoid(attention)


class CBAM(nn.Module):
    """Convolutional Block Attention Module combining channel and spatial attention"""
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class ResidualBlock(nn.Module):
    """Residual block with skip connection"""
    def __init__(self, main_path, skip_connection):
        super().__init__()
        self.main_path = main_path
        self.skip_connection = skip_connection
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = self.skip_connection(x)
        out = self.main_path(x)
        out += identity
        return self.relu(out)

    
def mixup_data(x, y, alpha=1.0, device='cpu'):
    """Apply mixup augmentation to a batch of data"""
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample()
        if isinstance(device, str):
            device = torch.device(device)
        lam = lam.to(device)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute mixup loss"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def evaluate_with_tta(model, test_loader, criterion, device, num_tta=5):
    """
    Evaluate with Test Time Augmentation for better accuracy
    """
    model.eval()
    
    # Define TTA transforms
    tta_transforms = [
        transforms.Compose([transforms.ToPILImage(), transforms.Resize((128, 128)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        transforms.Compose([transforms.ToPILImage(), transforms.Resize((128, 128)), transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        transforms.Compose([transforms.ToPILImage(), transforms.Resize((140, 140)), transforms.CenterCrop(128), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        transforms.Compose([transforms.ToPILImage(), transforms.Resize((120, 120)), transforms.Pad(4), transforms.CenterCrop(128), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        transforms.Compose([transforms.ToPILImage(), transforms.Resize((128, 128)), transforms.RandomRotation(degrees=5), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    ]

    with torch.no_grad():
        total_loss = 0.0
        num_correct = 0
        num_samples = 0

        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Apply TTA
            all_logits = []
            for i in range(min(num_tta, len(tta_transforms))):
                # Apply TTA transform
                tta_inputs = []
                for inp in inputs:
                    # Convert to PIL and apply transform
                    tta_inp = tta_transforms[i](inp.cpu())
                    tta_inputs.append(tta_inp)
                tta_inputs = torch.stack(tta_inputs).to(device)
                
                logits = model(tta_inputs)
                all_logits.append(logits)
            
            # Average predictions
            avg_logits = torch.mean(torch.stack(all_logits), dim=0)
            loss = criterion(avg_logits, labels)
            total_loss += loss.item()

            # Compute accuracy
            _, predictions = torch.max(avg_logits, dim=1)
            num_correct += (predictions == labels).sum().item()
            num_samples += len(inputs)

    avg_loss = total_loss / len(test_loader)
    accuracy = num_correct / num_samples
    
    return avg_loss, accuracy


def evaluate(model, test_loader, criterion, device):
    """
    Evaluate the CNN classifier on the validation set.

    Args:
        model (CNN): CNN classifier to evaluate.
        test_loader (torch.utils.data.DataLoader): Data loader for the test set.
        criterion (callable): Loss function to use for evaluation.
        device (torch.device): Device to use for evaluation.

    Returns:
        float: Average loss on the test set.
        float: Accuracy on the test set.
    """
    model.eval() # Set model to evaluation mode

    with torch.no_grad():
        total_loss = 0.0
        num_correct = 0
        num_samples = 0

        for inputs, labels in test_loader:
            # Move inputs and labels to device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Compute the logits and loss
            logits = model(inputs)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            # Compute the accuracy
            _, predictions = torch.max(logits, dim=1)
            num_correct += (predictions == labels).sum().item()
            num_samples += len(inputs)
            

    # Evaluate the model on the validation set
    avg_loss = total_loss / len(test_loader)
    accuracy = num_correct / num_samples
    
    return avg_loss, accuracy

def train_with_config(model, train_loader, val_loader, optimizer, criterion, scheduler, 
                     device, config, scaler=None, use_mixup=True, mixup_alpha=0.4):
    """
    Advanced training function with hardware-adaptive configuration
    """
    model = model.to(device)
    best_accuracy = 0.0
    num_epochs = config['num_epochs']
    use_mixed_precision = config['use_mixed_precision']
    
    for epoch in range(num_epochs):
        model.train()

        with tqdm(total=len(train_loader),
                  desc=f'Epoch {epoch +1}/{num_epochs}',
                  position=0,
                  leave=True) as pbar:
            for inputs, labels in train_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad()

                # Mixed precision training
                if use_mixed_precision and scaler is not None:
                    with torch.cuda.amp.autocast():
                        # Apply mixup augmentation
                        if use_mixup and torch.rand(1) < 0.5:
                            inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, mixup_alpha, device)
                            logits = model(inputs)
                            loss = mixup_criterion(criterion, logits, targets_a, targets_b, lam)
                        else:
                            logits = model(inputs)
                            loss = criterion(logits, labels)
                    
                    # Backward pass with gradient scaling
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Standard precision training
                    if use_mixup and torch.rand(1) < 0.5:
                        inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, mixup_alpha, device)
                        logits = model(inputs)
                        loss = mixup_criterion(criterion, logits, targets_a, targets_b, lam)
                    else:
                        logits = model(inputs)
                        loss = criterion(logits, labels)
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                # Update progress bar
                pbar.update(1)
                pbar.set_postfix(loss=loss.item())
                
                # Step OneCycleLR scheduler every batch
                if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    scheduler.step()

            # Evaluation
            avg_loss, accuracy = evaluate(model, val_loader, criterion, device)
            print(f'Validation set: Average loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}')
            
            # Save best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                save_dict = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': accuracy,
                    'epoch': epoch,
                    'config': config
                }
                torch.save(save_dict, 'best_model.ckpt')
                print(f'🎯 New best accuracy: {accuracy:.4f}')
            
        # Step other schedulers once per epoch
        if not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            scheduler.step()
    
    print(f"🏆 Training completed! Best accuracy: {best_accuracy:.4f}")


def train_with_config(model, train_loader, val_loader, optimizer, criterion, scheduler, 
                     device, config, use_mixup=True, mixup_alpha=0.4):
    """
    Enhanced training function with mixed precision support and adaptive configuration
    """
    # Place the model on device
    model = model.to(device)
    best_accuracy = 0.0
    
    # Setup mixed precision training for CUDA
    scaler = None
    if config['use_mixed_precision'] and device.type == 'cuda':
        scaler = torch.amp.GradScaler('cuda')
        print("🚀 Using mixed precision training (FP16)")
    
    # Setup DataParallel for multi-GPU (only if not already wrapped)
    if config['use_ddp'] and torch.cuda.device_count() > 1 and not isinstance(model, nn.DataParallel):
        model = nn.DataParallel(model)
        print(f"🔥 Using DataParallel with {torch.cuda.device_count()} GPUs")
    
    for epoch in range(config['num_epochs']):
        model.train()

        with tqdm(total=len(train_loader),
                  desc=f'Epoch {epoch + 1}/{config["num_epochs"]}',
                  position=0,
                  leave=True) as pbar:
            for inputs, labels in train_loader:
                inputs = inputs.to(device, non_blocking=config.get('pin_memory', False))
                labels = labels.to(device, non_blocking=config.get('pin_memory', False))

                optimizer.zero_grad()

                # Apply mixup augmentation
                if use_mixup and torch.rand(1) < 0.5:
                    inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, mixup_alpha, device)
                    
                    # Forward pass with mixed precision
                    if scaler is not None:
                        with torch.amp.autocast('cuda'):
                            logits = model(inputs)
                            loss = mixup_criterion(criterion, logits, targets_a, targets_b, lam)
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        logits = model(inputs)
                        loss = mixup_criterion(criterion, logits, targets_a, targets_b, lam)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                else:
                    # Normal training with mixed precision
                    if scaler is not None:
                        with torch.amp.autocast('cuda'):
                            logits = model(inputs)
                            loss = criterion(logits, labels)
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        logits = model(inputs)
                        loss = criterion(logits, labels)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()

                pbar.update(1)
                pbar.set_postfix(loss=loss.item())
                
                # Step OneCycleLR scheduler every batch
                if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    scheduler.step()

            # Evaluation after each epoch
            avg_loss, accuracy = evaluate(model, val_loader, criterion, device)
            print(f'Validation set: Average loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}')
            
            # Save best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                model_to_save = model.module if hasattr(model, 'module') else model
                torch.save({'model_state_dict': model_to_save.state_dict(),
                           'optimizer_state_dict': optimizer.state_dict(),
                           'accuracy': accuracy,
                           'config': config}, 'best_model.ckpt')
                print(f'New best accuracy: {accuracy:.4f}')
            
        # Step other schedulers once per epoch (if not OneCycleLR)
        if not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            scheduler.step()
    
    print(f"Training completed! Best accuracy: {best_accuracy:.4f}")
    return best_accuracy


def train(model, train_loader, val_loader, optimizer, criterion, scheduler, device,
          num_epochs, use_mixup=True, mixup_alpha=0.4):
    """
    Train the CNN classifer on the training set and evaluate it on the validation set every epoch.

    Args:
    model (CNN): CNN classifier to train.
    train_loader (torch.utils.data.DataLoader): Data loader for the training set.
    val_loader (torch.utils.data.DataLoader): Data loader for the validation set.
    optimizer (torch.optim.Optimizer): Optimizer to use for training.
    criterion (callable): Loss function to use for training.
    device (torch.device): Device to use for training.
    num_epochs (int): Number of epochs to train the model.
    use_mixup (bool): Whether to use mixup augmentation.
    mixup_alpha (float): Alpha parameter for mixup.
    """

    # Place the model on device
    model = model.to(device)
    best_accuracy = 0.0
    
    for epoch in range(num_epochs):
        model.train() # Set model to training mode

        with tqdm(total=len(train_loader),
                  desc=f'Epoch {epoch +1}/{num_epochs}',
                  position=0,
                  leave=True) as pbar:
            for inputs, labels in train_loader:
                #Move inputs and labels to device
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Apply mixup augmentation
                if use_mixup and torch.rand(1) < 0.5:  # Apply mixup 50% of the time
                    inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, mixup_alpha, device)
                    
                    # Compute the logits and loss with mixup
                    logits = model(inputs)
                    loss = mixup_criterion(criterion, logits, targets_a, targets_b, lam)
                else:
                    # Normal training
                    logits = model(inputs)
                    loss = criterion(logits, labels)
                
                # Backward pass and optimization with gradient clipping
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
                optimizer.step()

                # Update the progress bar
                pbar.update(1)
                pbar.set_postfix(loss=loss.item())
                
                # Step OneCycleLR scheduler every batch
                if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    scheduler.step()

            avg_loss, accuracy = evaluate(model, val_loader, criterion, device)
            print(
                f'Validation set: Average loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}'
                )
            
            # Save best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save({'model_state_dict': model.state_dict(),
                           'optimizer_state_dict': optimizer.state_dict(),
                           'accuracy': accuracy}, 'best_model.ckpt')
                print(f'New best accuracy: {accuracy:.4f}')
            
        # Step other schedulers once per epoch (if not OneCycleLR)
        if not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            scheduler.step()

def test(model, test_loader, device):
    """
    Get predictions for the test set.

    Args:
        model (CNN): classifier to evaluate.
        test_loader (torch.utils.data.DataLoader): Data loader for the test set.
        device (torch.device): Device to use for evaluation.

    Returns:
        list: List of (filename, prediction) tuples for the test set.
    """
    model = model.to(device)
    model.eval() # Set model to evaluation mode

    with torch.no_grad():
        all_preds = []

        for inputs, labels in test_loader:
            # Move inputs and labels to device
            inputs = inputs.to(device)

            logits = model(inputs)

            _, predictions = torch.max(logits, dim=1)
            preds = list(zip(labels, predictions.tolist()))
            all_preds.extend(preds)
    return all_preds

def main(args):
    # Simple device detection
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    # Basic configuration
    batch_size = 64
    num_workers = 4
    num_epochs = 10
    learning_rate = 0.001
    weight_decay = 0.01
    
    image_net_mean = torch.Tensor([0.485, 0.456, 0.406])
    image_net_std = torch.Tensor([0.229, 0.224, 0.225])
    
    ## Define data transformation with advanced augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((160, 160)),  # Larger resize for better crop diversity
        transforms.RandomResizedCrop((128, 128), scale=(0.8, 1.0), ratio=(0.9, 1.1)),  # Better crop augmentation
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20, interpolation=transforms.InterpolationMode.BILINEAR),  # Increased rotation with better interpolation
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),  # Perspective transformation
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),  # Affine transforms
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1),  # Stronger color augmentation
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.15),  # Blur augmentation
        transforms.ToTensor(),
        transforms.Normalize(image_net_mean, image_net_std),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.25), ratio=(0.3, 3.3)),  # Enhanced cutout
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(image_net_mean, image_net_std),
    ])

    data_root = 'data'
    
    # Create MiniPlaces dataset object
    miniplaces_train = MiniPlaces(data_root,
                                  split='train',
                                  transform=train_transform)
    miniplaces_val = MiniPlaces(data_root,
                                split='val',
                                transform=val_transform,
                                label_dict=miniplaces_train.label_dict)

    # Create DataLoaders with simple configuration
    train_loader = DataLoader(miniplaces_train,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              shuffle=True,
                              pin_memory=False)
    val_loader = DataLoader(miniplaces_val,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=False,
                            pin_memory=False)

    # Initialize model
    model = MyConv(num_classes=len(miniplaces_train.label_dict))
    
    # Simple optimizer configuration
    optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr=learning_rate,
                                  weight_decay=weight_decay,
                                  betas=(0.9, 0.999))

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Simple learning rate scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=learning_rate, 
        epochs=num_epochs, 
        steps_per_epoch=len(train_loader),
        pct_start=0.1,  # 10% warmup
        div_factor=10,  # Initial LR = max_lr/div_factor
        final_div_factor=100  # Final LR = initial_lr/final_div_factor
    )
    
    # No mixed precision in simplified version
    scaler = None

    if not args.test:
        # Start training with simple configuration
        train(model, train_loader, val_loader, optimizer, criterion, scheduler,
                         device, num_epochs)

        # Save model
        model_to_save = model.module if hasattr(model, 'module') else model
        save_dict = {'model_state_dict': model_to_save.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'learning_rate': learning_rate,
                    'num_epochs': num_epochs,
                    'batch_size': batch_size}
        torch.save(save_dict, 'model.ckpt')

    else:
        miniplaces_test = MiniPlaces(data_root,
                                     split='test',
                                     transform=val_transform)
        test_loader = DataLoader(miniplaces_test,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                shuffle=False,
                                pin_memory=False)        
        checkpoint = torch.load(args.checkpoint, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        preds = test(model, test_loader, device)
        write_predictions(preds, 'predictions.csv')

def write_predictions(preds, filename):
    with open(filename, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        for im, pred in preds:
            writer.writerow((im, pred))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--checkpoint', default='model.ckpt')
    args = parser.parse_args()
    main(args)
