import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm
import numpy as np
import torch.optim as optim
import os
import json


import time

def clean_preprocess(preprocess):
    """
    Removes Resize and CenterCrop from a transform sequence to preserve details
    of already scaled images.
    The method is a bit hardcoded, but works well for AlexNet, ResNet, etc.
    """
    # if hasattr(preprocess, 'transforms'):
    #     new_transforms = [t for t in preprocess.transforms if "Resize" not in str(t) and "CenterCrop" not in str(t)]
    #     if new_transforms:
    #         return nn.Sequential(*new_transforms)
    # return preprocess
    return v2.Normalize(mean=preprocess.mean, std=preprocess.std)

def getImageLoader(train_path='../image_scaled_train', test_path='../image_scaled_test', save_mapping_to=None):
    # Define transformations (optional, but recommended)
    transform = v2.Compose([
        v2.ToImage(), # Convert images to tensors
        v2.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0), antialias=True), # Conservative crop
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.1), # Add vertical flip
        v2.RandomAffine(degrees=20, translate=(0.15, 0.15), scale=(0.9, 1.1)), # More aggressive affine
        v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.01), # More aggressive color jitter
        v2.ToDtype(torch.float32, scale=True), # Convert to float and scale to [0, 1]
        v2.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0), # Obscure part of the image
    ])

    transform_eval = v2.Compose([
        v2.ToImage(), # Convert images to tensors
        v2.ToDtype(torch.float32, scale=True), # Match training dtype and scaling
    ])

    # Load data from folder
    train_dataset = datasets.ImageFolder(root=train_path, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_path, transform=transform_eval)

    # Save class mapping for later inference
    if save_mapping_to:
        with open(save_mapping_to, 'w') as f:
            json.dump(train_dataset.classes, f)
        print(f"Class mapping saved to {save_mapping_to}")
    else:
        with open('classes.json', 'w') as f:
            json.dump(train_dataset.classes, f)
        print(f"Class mapping saved to classes.json")

    # Create a DataLoader to iterate over the data
    dataloader_train = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
    dataloader_test = DataLoader(test_dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)

    return dataloader_train, dataloader_test

def trainTestModel(model, epochs, dataloader_train, dataloader_test, preprocess, optimizer, scheduler,
                   log_file=None,
                   device='cuda', criterion=None,
                   patience=7,
                   cm_every=10):
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    write_log = None
    if(log_file is not None):
        write_log = open(log_file, 'w')
        write_log.write(f"epoch;train_loss;val_loss;train_accuracy;val_accuracy;train_f1_score;val_f1_score;train_f1_macro;val_f1_macro;lr;time_sec;best_val_loss;patience_count\n")

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_path = None # Track the actual file on disk

    for epoch in range(epochs):
        start_time = time.time()
        # In this mode some layers of the model act differently
        model.train()
        loss_total = 0
        
        # Progress bar for training
        progress_bar = tqdm(dataloader_train, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        
        train_predictions, train_true_vals = [], []

        for i, batch in enumerate(progress_bar):
            # Data batch
            X_batch, y_batch = batch

            # Device with non_blocking=True for performance
            X_batch = preprocess(X_batch.to(device, non_blocking=True))
            y_batch = y_batch.to(device, non_blocking=True)

            # Gradient set to 0
            model.zero_grad()

            # Prediction calculation
            y_pred = model(X_batch)

            train_predictions.extend(y_pred.detach().cpu().numpy())
            # Save the real values to use them later
            train_true_vals.extend(y_batch.cpu().numpy())

            # Calculation of the loss function
            loss = criterion(y_pred, y_batch)
            # Backpropagation: calculate the gradient of the loss according to each layer
            loss.backward()

            # Gradient descent: updating parameters
            optimizer.step()

            loss_total += loss.item()
            
            # Update progress bar
            current_lr = optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({"loss": f"{loss_total/(i+1):.3f}", "lr": f"{current_lr:.2e}"})

        avg_loss = loss_total / len(dataloader_train)
        # Set of predictions from the dataset
        train_predictions = np.array(train_predictions)
        # Id prediction
        train_predictions = np.argmax(train_predictions, axis=-1)
        # Set of true values of the dataset
        train_true_vals = np.array(train_true_vals)

        # Evaluate on validation set
        model.eval()
        val_loss = 0.0

        test_predictions, test_true_vals = [], []
        
        # Progress bar for validation
        val_progress_bar = tqdm(dataloader_test, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False)
        
        with torch.no_grad():
            for X_val, y_val in val_progress_bar:
                X_val = preprocess(X_val.to(device, non_blocking=True))
                y_val = y_val.to(device, non_blocking=True)
                
                outputs = model(X_val)
                loss = criterion(outputs, y_val)
                val_loss += loss.item()

                test_predictions.extend(outputs.detach().cpu().numpy())
                # Save the real values to use them later
                test_true_vals.extend(y_val.cpu().numpy())
                
                val_progress_bar.set_postfix({"val_loss": f"{val_loss/len(dataloader_test):.3f}"})

        val_loss /= len(dataloader_test)

        # Set of predictions from the dataset
        test_predictions = np.array(test_predictions)
        # Id prediction
        test_predictions = np.argmax(test_predictions, axis=-1)
        # Set of true values of the dataset
        test_true_vals = np.array(test_true_vals)
        
        train_acc = accuracy_score(train_true_vals, train_predictions)
        train_f1 = f1_score(train_true_vals, train_predictions, average='weighted')

        val_acc = accuracy_score(test_true_vals, test_predictions)
        val_f1 = f1_score(test_true_vals, test_predictions, average='weighted')

        macro_train_f1 = f1_score(train_true_vals, train_predictions, average='macro')
        macro_val_f1 = f1_score(test_true_vals, test_predictions, average='macro')

        lr_cls = optimizer.param_groups[0]['lr']
        lr_str = f"LR: {lr_cls:.2e}"
        if len(optimizer.param_groups) > 1:
            lr_back = optimizer.param_groups[1]['lr']
            lr_str = f"LR Back: {lr_back:.2e}, LR Cls: {lr_cls:.2e}"
        
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_loss:.3f}, Val Loss: {val_loss:.3f}, Val Acc: {val_acc:.3f}, Val F1: {val_f1:.3f}, {lr_str}")

        # Check for cycle end (for CosineAnnealingWarmRestarts)
        # This is the "special save" at the bottom of the valley
        if hasattr(scheduler, 'T_cur') and hasattr(scheduler, 'T_i'):
            # If T_cur + 1 == T_i, the NEXT step() will restart the cycle
            if scheduler.T_cur + 1 >= scheduler.T_i:
                cycle_num = getattr(scheduler, 'cycle', 0)
                cycle_path = log_file + f".cycle_{cycle_num}.valley.model" if log_file else f"cycle_{cycle_num}.valley.model"
                torch.save(model.state_dict(), cycle_path)
                msg = f"\t--> End of Cycle {cycle_num} reached. 'Valley' model saved to {cycle_path}. Scheduler will RESTART in next epoch."
                print(msg)
                if write_log:
                    write_log.write(f"# {msg}\n")
                
                # Increment an internal cycle counter if it doesn't exist
                if not hasattr(scheduler, 'cycle'):
                    scheduler.cycle = 1
                else:
                    scheduler.cycle += 1

        # Intermediate predictions saving logic
        if cm_every is not None and (epoch + 1) % cm_every == 0:
            # Use base filename without extension for artifacts
            base_name = os.path.splitext(log_file)[0] if log_file else "model"
            preds_path = f"{base_name}_epoch_{epoch+1:02d}.preds.npz"
            np.savez(preds_path, true=test_true_vals, pred=test_predictions)
            print(f"\t--> Intermediate predictions saved to {preds_path}")

        if(write_log is not None):
            epoch_time = time.time() - start_time
            current_lr_str = f"{current_lr}"
            if len(optimizer.param_groups) > 1:
                # Store all LRs separated by comma if multiple
                current_lr_str = ",".join([str(pg['lr']) for pg in optimizer.param_groups])
            write_log.write(f"{epoch+1};{avg_loss:.3f};{val_loss:3f};{train_acc:.3f};{val_acc:.3f};{train_f1:.3f};{val_f1:.3f};{macro_train_f1:.3f};{macro_val_f1:.3f};{current_lr_str};{epoch_time:.1f};{best_val_loss:.3f};{epochs_no_improve}")
            write_log.write("\n")
            write_log.flush()

        # Best model saving logic
        if ((val_loss < best_val_loss) or ((epoch+1) % cm_every == 0)):
            # Remove previous best model if it exists to avoid cluttering with too many files
            if best_model_path is not None and os.path.exists(best_model_path):
                if((epoch+1) % cm_every > 1):
                    try:
                        os.remove(best_model_path)
                    except OSError:
                        pass
            # Update best_model_path with simplified naming
            base_name = os.path.splitext(log_file)[0] if log_file else "best_model"
            best_model_path = f"{base_name}_epoch_{epoch+1:02d}.model"

            torch.save(model.state_dict(), best_model_path)
            print(f"\t--> Best model saved to {best_model_path}")

        if (val_loss < best_val_loss):
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement.")
                break

        # Scheduler update
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

    if (write_log is not None):
        write_log.close()
