import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import v2
from torchvision import models
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


def prepare_model(model_name, num_classes, fine_tune_type='none', checkpoint_path=None, dropout=0.0):
    """
    Prepares the model for training.
    fine_tune_type: 'none' (model frozen - inference) or 'classifier' (freeze backbone) or 'full' (unfreeze all) or 'resnet_selective' (specific blocks)
    checkpoint_path: if present, represents the path from which to load the model weights from
    """
    if model_name == 'alexnet':
        weights = models.AlexNet_Weights.DEFAULT
        model = models.alexnet(weights=weights)
        preprocess = weights.transforms()
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)

    elif model_name == 'vgg16':
        weights = models.VGG16_Weights.DEFAULT
        model = models.vgg16(weights=weights)
        preprocess = weights.transforms()
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)

    elif model_name == 'resnet50':
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        preprocess = weights.transforms()
        old_fc = model.fc
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(model.fc.in_features, 1024),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(1024, num_classes)
        )
        # else:
        #     model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'vit_b_16':
        weights = models.ViT_B_16_Weights.DEFAULT
        model = models.vit_b_16(weights=weights)
        preprocess = weights.transforms()
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    else:
        raise ValueError(f"Model {model_name} not supported")

    # Clean up preprocess: Remove Resize and CenterCrop as images are already 224x224
    # This preserves edge details that would otherwise be cropped out.
    preprocess = clean_preprocess(preprocess)
    print("Preprocess updated: Removed Resize and CenterCrop (images are already scaled).")

    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            state_dict = torch.load(checkpoint_path)
            model.load_state_dict(state_dict)
            print(f"Successfully loaded checkpoint from {checkpoint_path}")
        except Exception as e:
            print(f"Error loading checkpoint {checkpoint_path}: {e}")
            print("Proceeding with pre-trained ImageNet weights.")

    for param in model.parameters():
        param.requires_grad = False

    if fine_tune_type == 'none':
        #inference
        pass
    else:
        # Set requires_grad based on fine_tune_type
        # if fine_tune_type == 'classifier':
        # Always training the classifier
        if model_name in ['alexnet', 'vgg16']:
            for param in model.classifier.parameters():
                param.requires_grad = True
        elif model_name == 'resnet50':
            for param in model.fc.parameters():
                param.requires_grad = True
        elif model_name == 'vit_b_16':
            for param in model.heads.parameters():
                param.requires_grad = True

        if fine_tune_type == 'full':
            if model_name == 'resnet50':
                for param in model.layer4.parameters():
                    param.requires_grad = True
                for param in model.layer3.parameters():
                    param.requires_grad = True
            else:
                for param in model.parameters():
                    param.requires_grad = True
        elif fine_tune_type == 'resnet_selective' and model_name == 'resnet50':
            for param in model.layer4.parameters():
                param.requires_grad = True

            # fc_module = model.fc if isinstance(model.fc, nn.Linear) else model.fc[1]
            for param in model.fc.parameters():
                param.requires_grad = True

    return model, preprocess


def get_optimizer(model, model_name, fine_tune_type, lr_classifier=1e-2, lr_backbone=1e-3):
    if fine_tune_type == 'full':
        if model_name == 'alexnet':
            optimizer = optim.Adam([
                {'params': model.classifier.parameters(), 'lr': lr_classifier},
                {'params': model.features.parameters(), 'lr': lr_backbone}
            ])  # , momentum=0.9)
        elif model_name == 'vgg16':
            optimizer = optim.Adam([
                {'params': model.classifier.parameters(), 'lr': lr_classifier},
                {'params': model.features.parameters(), 'lr': lr_backbone}
            ])  # , momentum=0.9)
        elif model_name == 'resnet50':
            # For ResNet, 'features' are everything except 'fc'
            # backbone_params = [model.layer4.parameters(), model.layer3.parameters()] #[p for n, p in model.named_parameters() if 'fc' not in n]
            # fc_params = model.fc.parameters() #if isinstance(model.fc, nn.Linear) else model.fc[1].parameters()
            optimizer = optim.Adam([
                {'params': model.fc.parameters(), 'lr': lr_classifier},
                {'params': model.layer4.parameters(), 'lr': lr_backbone},
                {'params': model.layer3.parameters(), 'lr': lr_backbone}
            ])  # , momentum=0.9)
        elif model_name == 'vit_b_16':
            backbone_params = [p for n, p in model.named_parameters() if 'heads' not in n]
            optimizer = optim.Adam([
                {'params': model.heads.parameters(), 'lr': lr_classifier},
                {'params': backbone_params, 'lr': lr_backbone}
            ])  # , momentum=0.9)
    elif fine_tune_type == 'resnet_selective' and model_name == 'resnet50':
        # Ensure layer4 and fc are unfrozen
        for param in model.parameters():
            param.requires_grad = False

        for param in model.layer4.parameters():
            param.requires_grad = True
        for param in model.fc.parameters():
            param.requires_grad = True

        optimizer = optim.Adam([
            {'params': model.fc.parameters(), 'lr': lr_classifier},
            {'params': model.layer4.parameters(), 'lr': lr_backbone}
        ])  # , momentum=0.9)
    else:
        # Default for classifier-only (only optimized parameters that require grad)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=lr_classifier)  # , momentum=0.9)

    return optimizer

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
