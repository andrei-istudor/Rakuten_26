import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from Train_Test import getImageLoader, trainTestModel, clean_preprocess
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import argparse
import os
import sys
import datetime
import re

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # This flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        self.terminal.flush()
        self.log.flush()

def get_session_info(model_name, resume_path=None):
    """
    Determines the session ID and name based on existing folders and resume path.
    Returns: session_id (int), session_name (str), base_folder (str)
    """
    sessions_dir = "experiments"
    if not os.path.exists(sessions_dir):
        os.makedirs(sessions_dir)
        
    # Find next session ID for this model
    pattern = re.compile(rf"{model_name}_(\d+)")
    existing_ids = []
    for d in os.listdir(sessions_dir):
        match = pattern.match(d)
        if match:
            existing_ids.append(int(match.group(1)))
    
    session_id = max(existing_ids) + 1 if existing_ids else 1
    session_id_str = f"{session_id:02d}"
    
    session_name = f"{model_name}_{session_id_str}"
    
    if resume_path:
        # Try to extract parent info from resume path
        # Expected format: experiments/alexnet_01/.../alexnet_01_epoch_08.model
        parent_match = re.search(rf"({model_name}_\d+_epoch_\d+)", resume_path)
        if parent_match:
            session_name += f"_from_{parent_match.group(1)}"
        else:
            # Fallback if filename doesn't match exactly
            parent_folder = os.path.basename(os.path.dirname(resume_path))
            if parent_folder:
                 session_name += f"_from_{parent_folder}"

    session_folder = os.path.join(sessions_dir, session_name)
    os.makedirs(session_folder, exist_ok=True)
    
    return session_id, session_name, session_folder

def prepare_model(model_name, num_classes, fine_tune_type='classifier', checkpoint_path=None, dropout=0.0):
    """
    Prepares the model for training.
    fine_tune_type: 'classifier' (freeze backbone) or 'full' (unfreeze all) or 'resnet_selective' (specific blocks)
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
            ]) #, momentum=0.9)
        elif model_name == 'vgg16':
             optimizer = optim.Adam([
                {'params': model.classifier.parameters(), 'lr': lr_classifier},
                {'params': model.features.parameters(), 'lr': lr_backbone}
            ])# , momentum=0.9)
        elif model_name == 'resnet50':
             # For ResNet, 'features' are everything except 'fc'
             # backbone_params = [model.layer4.parameters(), model.layer3.parameters()] #[p for n, p in model.named_parameters() if 'fc' not in n]
             # fc_params = model.fc.parameters() #if isinstance(model.fc, nn.Linear) else model.fc[1].parameters()
             optimizer = optim.Adam([
                {'params': model.fc.parameters(), 'lr': lr_classifier},
                {'params': model.layer4.parameters(), 'lr': lr_backbone},
                {'params': model.layer3.parameters(), 'lr': lr_backbone}
             ])# , momentum=0.9)
        elif model_name == 'vit_b_16':
             backbone_params = [p for n, p in model.named_parameters() if 'heads' not in n]
             optimizer = optim.Adam([
                {'params': model.heads.parameters(), 'lr': lr_classifier},
                {'params': backbone_params, 'lr': lr_backbone}
            ])# , momentum=0.9)
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
        ])# , momentum=0.9)
    else:
        # Default for classifier-only (only optimized parameters that require grad)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_classifier) #, momentum=0.9)
    
    return optimizer

def main():
    parser = argparse.ArgumentParser(description='Train CNN models for Rakuten dataset (V2)')
    parser.add_argument('--train_data', type=str, default='../../project_rakuten/image_train_15', help='Path to the train data folder')
    parser.add_argument('--test_data', type=str, default='../../project_rakuten/image_test_15', help='Path to the test data folder')
    parser.add_argument('--model', type=str, default='alexnet', choices=['alexnet', 'vgg16', 'resnet50'], help='Model architecture')
    parser.add_argument('--mode', type=str, default='classifier', choices=['classifier', 'full', 'resnet_selective'], help='Fine-tuning mode')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--log_prefix', type=str, default='2601', help='Prefix for log files')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device (cuda/cpu)')
    parser.add_argument('--lr_cls', type=float, default=1e-2, help='Learning rate for classifier')
    parser.add_argument('--lr_back', type=float, default=1e-3, help='Learning rate for backbone (full/selective modes)')
    parser.add_argument('--scheduler', type=str, default='steplr', choices=['cosine', 'steplr', 'plateau'], help='LR scheduler type')
    parser.add_argument('--step_size', type=int, default=10, help='Step size for StepLR')
    parser.add_argument('--gamma', type=float, default=0.1, help='Gamma for StepLR or Plateau')
    parser.add_argument('--T_0', type=int, default=30, help='T_0 for CosineAnnealingWarmRestarts')
    parser.add_argument('--T_mult', type=float, default=2.0, help='T_mult for CosineAnnealingWarmRestarts')
    parser.add_argument('--cm_every', type=int, default=5, help='Save predictions every N epochs')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate before FC layer (ResNet50)')
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    
    args = parser.parse_args()
    
    # 1. Setup Session and Logging
    session_id, session_name, session_folder = get_session_info(args.model, args.resume)
    log_file_path = os.path.join(session_folder, f"session.log")
    sys.stdout = Logger(log_file_path)
    
    print(f"--- Rakuten CNN Training V2 ---")
    print(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Session Name: {session_name}")
    print(f"Output Folder: {session_folder}")
    print(f"Train data: {args.train_data}")
    print(f"Test data: {args.test_data}")
    print(f"Model: {args.model}")
    print(f"Mode: {args.mode}")
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}")
    print(f"LR Classifier: {args.lr_cls}")
    print(f"LR Backbone: {args.lr_back}")
    print(f"Scheduler: {args.scheduler}")
    if args.scheduler == 'steplr':
        print(f"Step Size: {args.step_size}, Gamma: {args.gamma}")
    elif args.scheduler == 'plateau':
        print(f"Factor: {args.gamma}, Patience: 5")
    else:
        print(f"T_0: {args.T_0}, T_mult: {args.T_mult}")
    print(f"Dropout: {args.dropout}")
    print(f"Label Smoothing: {args.label_smoothing}")
    if args.resume:
        print(f"Resume from: {args.resume}")
    print(f"-------------------------------")

    # 2. Load Data
    train_path = args.train_data #'../../project_rakuten/image_train_15'
    test_path = args.test_data #'../../project_rakuten/image_test_15'
    print(f"Loading data from: {train_path} and {test_path}")
    
    try:
        mapping_save_path = os.path.join(session_folder, 'classes.json')
        dataloader_train, dataloader_test = getImageLoader(train_path=train_path, test_path=test_path, save_mapping_to=mapping_save_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Class weights calculation
    train_classes = dataloader_train.dataset.targets
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_classes), y=train_classes)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(args.device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Prepare Model
    model, preprocess = prepare_model(args.model, num_classes=27, fine_tune_type=args.mode, checkpoint_path=args.resume, dropout=args.dropout)
    model.to(args.device)

    # Optimizer & Scheduler
    optimizer = get_optimizer(model, args.model, args.mode, lr_classifier=args.lr_cls, lr_backbone=args.lr_back)
    
    # Use LabelSmoothing if requested
    if args.label_smoothing > 0:
        print(f"Using Label Smoothing: {args.label_smoothing}")
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)

    if args.scheduler == 'steplr':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.gamma, patience=5)
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.T_0, T_mult=int(args.T_mult), eta_min=1e-5)
    
    # Define CSV log path
    csv_log = os.path.join(session_folder, f"{session_name}.csv")
    
    # Start Training
    trainTestModel(
        model=model,
        epochs=args.epochs,
        dataloader_train=dataloader_train,
        dataloader_test=dataloader_test,
        preprocess=preprocess,
        optimizer=optimizer,
        scheduler=scheduler,
        log_file=csv_log,
        device=args.device,
        criterion=criterion,
        cm_every=args.cm_every
    )
    
    # Final save
    final_path = os.path.join(session_folder, f"{session_name}_final.model")
    torch.save(model.state_dict(), final_path)
    print(f"Final model saved to {final_path}")

if __name__ == '__main__':
    main()
