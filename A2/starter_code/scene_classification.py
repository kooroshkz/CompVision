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


    def forward(self, x, return_intermediate=False):
        return torch.tensor(0)

    
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

def train(model, train_loader, val_loader, optimizer, criterion, device,
          num_epochs):
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
    """

    # Place the model on device
    model = model.to(device)
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

                # Compute the logits and loss
                logits = model(inputs)
                loss = torch.tensor(0)

                # Update the progress bar
                pbar.update(1)
                pbar.set_postfix(loss=loss.item())

            avg_loss, accuracy = evaluate(model, val_loader, criterion, device)
            print(
                f'Validation set: Average loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}'
                )

def test(model, test_loader, device):
    """
    Get predictions for the test set.

    Args:
        model (CNN): classifier to evaluate.
        test_loader (torch.utils.data.DataLoader): Data loader for the test set.
        device (torch.device): Device to use for evaluation.

    Returns:
        float: Average loss on the test set.
        float: Accuracy on the test set.
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
            
            

    # Evaluate the model on the validation set
    avg_loss = total_loss / len(test_loader)
    accuracy = num_correct / num_samples
    
    return avg_loss, accuracy

def main(args):
    image_net_mean = torch.Tensor([0.485, 0.456, 0.406])
    image_net_std = torch.Tensor([0.229, 0.224, 0.225])
    
    ## Define data transformation
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((128,128)),
        transforms.Normalize(image_net_mean, image_net_std),
    ])

    data_root = 'data'
    
    # Create MiniPlaces dataset object
    miniplaces_train = MiniPlaces(data_root,
                                  split='train',
                                  transform=data_transform)
    miniplaces_val = MiniPlaces(data_root,
                                split='val',
                                transform=data_transform,
                                label_dict=miniplaces_train.label_dict)

    # Create the dataloaders
    
    # Define the batch size and number of workers
    batch_size = 64
    num_workers = 2

    # Create DataLoader for training and validation sets
    train_loader = DataLoader(miniplaces_train,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              shuffle=True)
    val_loader = DataLoader(miniplaces_val,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MyConv(num_classes=len(miniplaces_train.label_dict))
                   

    optimizer = None 

    criterion = None

    if not args.test:

        train(model, train_loader, val_loader, optimizer, criterion,
              device, num_epochs=5)

        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict()}, 'model.ckpt')

    else:
        miniplaces_test = MiniPlaces(data_root,
                                     split='test',
                                     transform=data_transform)
        test_loader = DataLoader(miniplaces_test,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                shuffle=False)        
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
