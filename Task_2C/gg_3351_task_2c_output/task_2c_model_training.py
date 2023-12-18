import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
from pathlib import Path
import multiprocessing


device = "cuda" if torch.cuda.is_available() else "cpu"

train_dir = "dataset/train"
test_dir = "dataset/test"

threads = multiprocessing.cpu_count()

BATCH_SIZE = 32

EPOCHS = 100

IMG_SIZE = (275, 183)

data_transform = transforms.Compose([
    
    transforms.Resize(size=IMG_SIZE),

    transforms.RandomHorizontalFlip(p=0.5),
    
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
   
    transforms.ToTensor()
])



def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device = device):
    train_loss, train_acc = 0, 0
    model.to(device)
    for batch, (X, y) in enumerate(data_loader):
        # Send data to GPU
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(y_true=y,
                                 y_pred=y_pred.argmax(dim=1)) 

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

    # Calculate loss and accuracy per epoch and print out what's happening
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")
    

def test_step(data_loader: torch.utils.data.DataLoader,
              model: torch.nn.Module,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device = device):
    test_loss, test_acc = 0, 0
    model.to(device)
    model.eval() # put model in eval mode
    # Turn on inference context manager
    with torch.inference_mode(): 
        for X, y in data_loader:
            # Send data to GPU
            X, y = X.to(device), y.to(device)
            
            # 1. Forward pass
            test_pred = model(X)
            
            # 2. Calculate loss and accuracy
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y_true=y,
                y_pred=test_pred.argmax(dim=1)
            )
        
        # Adjust metrics and print out
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")

def save(model,model_name):
    MODEL_PATH = Path("models")
    MODEL_PATH.mkdir(parents=True, # create parent directories if needed
                    exist_ok=True # if models directory already exists, don't error
    )

    # Create model save path
    MODEL_NAME = model_name + ".pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

    # Save the model
    print(f"Saving model to: {MODEL_SAVE_PATH}")
    torch.save(obj=model,
            f=MODEL_SAVE_PATH)



def main():
    train_data = datasets.ImageFolder(root=train_dir,
                                transform=data_transform,
                                target_transform=None)

    test_data = datasets.ImageFolder(root=test_dir, 
                                    transform=data_transform)

    class_names = train_data.classes

    train_dataloader = DataLoader(dataset=train_data, 
                                batch_size=BATCH_SIZE, 
                                num_workers=threads,
                                shuffle=True)

    test_dataloader = DataLoader(dataset=test_data, 
                                batch_size=BATCH_SIZE, 
                                num_workers=threads, 
                                shuffle=False)

    weights = torchvision.models.EfficientNet_V2_S_Weights.DEFAULT 
    model = torchvision.models.efficientnet_v2_s(weights=weights).to(device)

    for param in model.features.parameters():
        param.requires_grad = False

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Get the length of class_names (one output unit for each class)
    output_shape = len(class_names)

    # Recreate the classifier layer and seed it to the target device
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True), 
        torch.nn.Linear(in_features=1280, 
                        out_features=output_shape, # same number of output units as our number of classes
                        bias=True))

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(EPOCHS):
        print(f"Epoch: {epoch}\n---------")
        train_step(data_loader=train_dataloader, 
            model=model, 
            loss_fn=loss_fn,
            optimizer=optimizer,
            accuracy_fn=accuracy_fn
        )
        test_step(data_loader=test_dataloader,
            model=model,
            loss_fn=loss_fn,
            accuracy_fn=accuracy_fn
        )

    save(model,"model_1")


if __name__ == '__main__':
    
    main()