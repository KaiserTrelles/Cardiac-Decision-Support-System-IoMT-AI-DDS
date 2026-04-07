import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import os
import time

if __name__ == '__main__':

    # --- 1. DEFINE DATA LOCATION ---
    data_dir = r'C:\Users\Usuario\Desktop\A Real-Time Cardiac Decision Support System for Internet of Medical Things Using Edge AI and DDS\Datasets\Children Dataset'

    # --- 2. DEFINE TRANSFORMS WITH AUGMENTATION ---
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    print("Loading data...")
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'val']}
    
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16, shuffle=True, num_workers=4)
                   for x in ['train', 'val']}

    class_names = image_datasets['train'].classes
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} (Your GPU!)")

    # --- 3. LOAD MODEL AND ADD MORE DROPOUT ---
    print("Loading MobileNetV3-Small model...")
    
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    num_ftrs = model.classifier[3].in_features
    
    model.classifier[2] = nn.Dropout(p=0.5, inplace=True)
    model.classifier[3] = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)

    # --- 4. DEFINE LOSS AND NEW OPTIMIZER ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-4)

    # --- 5. TRAIN THE MODEL ---
    num_epochs = 100 
    print(f"Starting training for up to {num_epochs} epochs...")
    print("Using Early Stopping with patience=10")
    since = time.time()

    # --- 6. CHECKPOINTING & HISTORY VARIABLES ---
    best_val_acc = 0.0
    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 10

    # ADDED: Dictionary to store history
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])

            # ADDED: Store metrics in the history dictionary
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val':
                if epoch_acc > best_val_acc:
                    best_val_acc = epoch_acc
                    best_val_loss = epoch_loss
                    torch.save(model.state_dict(), 'ecg_mobilenet_BEST_children.pth')
                    print(f"*** New best model saved (Acc: {best_val_acc:.4f}, Loss: {best_val_loss:.4f})! ***")
                    epochs_no_improve = 0 
                
                elif (epoch_acc == best_val_acc) and (epoch_loss < best_val_loss):
                    best_val_loss = epoch_loss
                    torch.save(model.state_dict(), 'ecg_mobilenet_BEST_children.pth')
                    print(f"*** Best model updated (Acc: {best_val_acc:.4f}, Loss: {best_val_loss:.4f})! ***")
                    epochs_no_improve = 0
                
                else:
                    epochs_no_improve += 1

        # ADDED: Save history after every epoch so you don't lose it if it crashes
        torch.save(history, 'training_history_children.pth')

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping! No improvement in {patience} epochs.")
            break 

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Validation Acc: {best_val_acc:.4f}')
    print(f'Best Validation Loss: {best_val_loss:.4f}')
    print("History saved to training_history_children.pth")
