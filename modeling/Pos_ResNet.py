import os
import sys
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

sys.path.append(os.path.dirname(os.getcwd()))
import utils.modeling as models
from utils.loadutils import *

import multiprocessing
multiprocessing.set_start_method('spawn', force=True)


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

# Parse JSON-formatted activities list
activities = json.loads(args.activities)
num_classes = len(activities)

# Load data
secs = args.secs
imu_window_size = int(secs * 50)
video_window_size = int(secs * 30)
print('IMU Data window size:', imu_window_size)
print('Video Data window size:', video_window_size)

# Define main function
def main():
    activities = json.loads(args.activities)
    num_classes = len(activities)
    set_random_seed(args.seed)
    
    # Initialize KFold here
    kf = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)

    ins='VID'
    out='act'
    dataset = VIDIMU(args.dpath, 
                     args.secs, 
                     activities=activities, 
                     ins=ins, 
                     out=out)
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        train_loader, test_loader = loaders_cv(dataset, 
                                               fold, 
                                               train_idx, 
                                               test_idx, 
                                               ins, out, 
                                               args.batch_size, 
                                               args.cv_folds, 
                                               args.seed)

        for in_batch, _ in train_loader:
            input_channels = in_batch.shape[1]
            break

        model = models.TemporalResNet(input_channels, num_classes, num_blocks=args.depth)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=args.factor, patience=args.patience)

        train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, args.modelname, args.seed, fold, num_epochs=args.epochs)
        print(f"Completed fold {fold + 1}/{args.cv_folds}\n")

def test_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.float()
            outputs = model(X_batch)
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
    no_improvement_epochs = 0 

    for epoch in range(num_epochs):
        total_loss = 0.0
        correct_train = 0
        total_train = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.float(), y_batch.long()
            optimizer.zero_grad()
            outputs = model(X_batch)

            loss = criterion(outputs, y_batch)
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

        if test_accuracy > best_accuracy:
            if best_model_filename is not None and os.path.exists(best_model_filename):
                os.remove(best_model_filename)

            best_accuracy = test_accuracy
            best_model_filename = f'{args.model_out_path}/best_{modelname}_fold{fold+1}_{args.depth}_{args.secs}_{seed}_acc_{best_accuracy:.4f}.pth'
            torch.save(model.state_dict(), best_model_filename)
            print(f"New best model for fold {fold+1} saved as {best_model_filename}")
            no_improvement_epochs = 0 
        else:
            no_improvement_epochs += 1 

        if no_improvement_epochs >= args.early_stop_patience:
            print(f"Early stopping triggered after {no_improvement_epochs} epochs without improvement.")
            break

if __name__ == "__main__":
    main()