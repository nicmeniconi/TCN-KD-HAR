import os
import sys
import argparse
import json
from functools import partial
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
sys.path.append(os.path.dirname(os.getcwd())) # Add project dir to path
import utils.modeling as models
from utils.loadutils9 import *

import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

# Define the argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--dpath", type=str, required=True)
parser.add_argument("--model_out_path", type=str, required=True)
parser.add_argument("--activities", type=str, required=True)
parser.add_argument("--secs", type=float, required=True)
parser.add_argument("--split", type=float, required=True)
parser.add_argument("--batch_size", type=int, required=True)
parser.add_argument("--depth", type=int, required=True)
parser.add_argument("--epochs", type=int, required=True)
parser.add_argument("--lr", type=float, required=True)
parser.add_argument("--factor", type=float, required=True)
parser.add_argument("--patience", type=int, required=True)
parser.add_argument("--cv_folds", type=int, required=True)
parser.add_argument("--early_stop_patience", type=int, required=True) 
parser.add_argument("--modelname", type=str, required=True)
args = parser.parse_args()

# Define main function with cross-validation
def main():
    activities = json.loads(args.activities)
    num_classes = len(activities)
    set_random_seed(args.seed)
    
    # Load dataset once for all folds
    dataset = VIDIMU(args.dpath, args.secs, activities, ins='VIDIMU', out='act')
    kf = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)

    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        print(f"Starting fold {fold + 1}/{args.cv_folds}")
        
        set_random_seed(args.seed + fold)
        
        train_subset = Subset(dataset, train_idx)
        test_subset = Subset(dataset, test_idx)

        # Compute min-max scaling parameters on train set
        imu_min, imu_max, video_min, video_max = compute_scaling_params(dataset, train_idx, ins='VIDIMU')
        collate = partial(collate_fn, ins='VIDIMU', out='act', imu_min=imu_min, imu_max=imu_max, video_min=video_min, video_max=video_max)
        
        # DataLoaders with min-max scaling
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=0)
        test_loader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=0)

        # Model initialization
        for vid_batch, imu_batch, _ in train_loader:
            vid_channels = vid_batch.shape[1]
            imu_channels = imu_batch.shape[1]
            break

        model = models.TemporalResNetMultiInput(imu_channels, vid_channels, num_classes, num_blocks=args.depth)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=args.factor, patience=args.patience)

        # Train model
        train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, args.modelname, args.seed, fold, num_epochs=args.epochs)
        print(f"Completed fold {fold + 1}/{args.cv_folds}\n")

def test_model(model, test_loader):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for vid_batch, imu_batch, y_batch in test_loader:
            outputs = model(imu_batch.float(), vid_batch.float())
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=1)    
    return accuracy, precision, recall, f1

def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, modelname, seed, fold, num_epochs=20):
    model.train()
    best_accuracy = 0.0
    best_model_filename = None
    no_improvement_epochs = 0  # Counter for early stopping

    for epoch in range(num_epochs):
        total_loss, correct_train, total_train = 0.0, 0, 0

        for vid_batch, imu_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(imu_batch.float(), vid_batch.float())
            loss = criterion(outputs, y_batch.long())
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == y_batch).sum().item()
            total_train += y_batch.size(0)

        train_accuracy = correct_train / total_train
        test_accuracy, precision, recall, f1 = test_model(model, test_loader)

        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Loss: {total_loss / len(train_loader):.4f}, "
              f"Train Accuracy: {train_accuracy:.4f}, "
              f"Test Accuracy: {test_accuracy:.4f}, "
              f"Precision: {precision:.4f}, "
              f"Recall: {recall:.4f}, "
              f"F1-Score: {f1:.4f}")

        scheduler.step(test_accuracy)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr:.1e}")

        # Check if there is an improvement
        if test_accuracy > best_accuracy:
            if best_model_filename and os.path.exists(best_model_filename):
                os.remove(best_model_filename)

            best_accuracy = test_accuracy
            best_model_filename = f'{args.model_out_path}/best_{modelname}_fold{fold+1}_{args.depth}_{args.secs}_{seed}_acc_{best_accuracy:.4f}.pth'
            torch.save(model.state_dict(), best_model_filename)
            print(f"New best model for fold {fold+1} saved as {best_model_filename}")
            no_improvement_epochs = 0  # Reset counter on improvement
        else:
            no_improvement_epochs += 1  # Increment if no improvement

        # Early stopping check
        if no_improvement_epochs >= args.early_stop_patience:
            print(f"Early stopping triggered after {no_improvement_epochs} epochs without improvement.")
            break

if __name__ == "__main__":
    main()