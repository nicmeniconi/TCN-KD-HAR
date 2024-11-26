import torch
import torch.optim as optim
import torch.nn as nn
from pytorch_tcn import TCN
import os
import matplotlib.pyplot as plt
import random
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset

# Set random seed for reproducibility
def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Initialize seed
set_random_seed(42)

# class TCN(nn.Module):
#     def __init__(self, input_channels, num_classes, hidden_dim=64, kernel_size=5, num_layers=4):
#         super(TCN, self).__init__()
#         layers = []
#         for i in range(num_layers):
#             dilation_size = 2 ** i
#             in_channels = input_channels if i == 0 else hidden_dim
#             layers.append(
#                 nn.Conv1d(in_channels, hidden_dim, kernel_size, dilation=dilation_size, padding=(kernel_size - 1) * dilation_size)
#             )
#             layers.append(nn.ReLU())
#             layers.append(nn.BatchNorm1d(hidden_dim))
#             layers.append(nn.Dropout(0.2))
#         self.network = nn.Sequential(*layers)
#         self.fc = nn.Linear(hidden_dim, num_classes)

#     def forward(self, x):
#         x = self.network(x)
#         x = torch.mean(x, dim=2)  # Global Average Pooling
#         x = self.fc(x)
#         return x

# class MultimodalTCN(nn.Module):
#     def __init__(self, imu_input_channels, video_input_channels, num_classes, hidden_dim=64, kernel_size=5, num_layers=4):
#         super(MultimodalTCN, self).__init__()
        
#         # TCN for IMU data
#         self.imu_tcn = self._build_tcn(imu_input_channels, hidden_dim, kernel_size, num_layers)
        
#         # TCN for Video data
#         self.video_tcn = self._build_tcn(video_input_channels, hidden_dim, kernel_size, num_layers)
        
#         # Final fully connected layer
#         self.fc = nn.Linear(hidden_dim * 2, num_classes)  # Combine both outputs

#     def _build_tcn(self, input_channels, hidden_dim, kernel_size, num_layers):
#         """Helper function to construct the TCN layers."""
#         layers = []
#         for i in range(num_layers):
#             dilation_size = 2 ** i
#             in_channels = input_channels if i == 0 else hidden_dim
#             layers.append(
#                 nn.Conv1d(in_channels, hidden_dim, kernel_size, dilation=dilation_size, padding=(kernel_size - 1) * dilation_size)
#             )
#             layers.append(nn.ReLU())
#             # layers.append(nn.LeakyReLU(negative_slope=0.01))
#             layers.append(nn.BatchNorm1d(hidden_dim))
#             layers.append(nn.Dropout(0.2))
#         return nn.Sequential(*layers)

#     def forward(self, imu_input, video_input):
#         # Pass through the IMU TCN
#         imu_out = self.imu_tcn(imu_input)  # Shape: (batch_size, hidden_dim, sequence_length)
#         imu_out = torch.mean(imu_out, dim=2)  # Global Average Pooling: (batch_size, hidden_dim)
        
#         # Pass through the Video TCN
#         video_out = self.video_tcn(video_input)  # Shape: (batch_size, hidden_dim, sequence_length)
#         video_out = torch.mean(video_out, dim=2)  # Global Average Pooling: (batch_size, hidden_dim)
        
#         # Concatenate both TCN outputs
#         combined_out = torch.cat((imu_out, video_out), dim=1)  # Shape: (batch_size, hidden_dim * 2)

#         # Final fully connected layer
#         out = self.fc(combined_out)  # Shape: (batch_size, num_classes)
#         return out



class SingleModalityTCN(torch.nn.Module):
    def __init__(self, input_channels, num_classes, hidden_channels=[64, 64, 64], kernel_size=5, dropout=0.2, causal=True):
        super(SingleModalityTCN, self).__init__()
        
        # Use the TCN class from pytorch-tcn with correct parameters
        self.tcn = TCN(
            num_inputs=input_channels,  # Number of input features
            num_channels=hidden_channels,  # List of hidden channels
            kernel_size=kernel_size,
            dropout=dropout,
            causal=causal
        )
        
        # Fully connected layer for classification
        self.fc = torch.nn.Linear(hidden_channels[-1], num_classes)

    def forward(self, x):
        # Input shape: (batch_size, input_channels, sequence_length)
        x = self.tcn(x)  # Pass through the TCN

        # Apply Global Average Pooling along the sequence length dimension
        x = torch.mean(x, dim=2)  # Shape: (batch_size, num_channels)

        # Pass through the final fully connected layer
        x = self.fc(x)  # Shape: (batch_size, num_classes)

        return x

class MultimodalTCN(torch.nn.Module):
    def __init__(self, imu_input_channels, video_input_channels, num_classes, hidden_channels=[64, 64, 64], kernel_size=5, dropout=0.2):
        super(MultimodalTCN, self).__init__()
        # TCN for IMU data
        self.imu_tcn = TCN(
            num_inputs=imu_input_channels,
            num_channels=hidden_channels,
            kernel_size=kernel_size,
            dropout=dropout
        )
        
        # TCN for Video data
        self.video_tcn = TCN(
            num_inputs=video_input_channels,
            num_channels=hidden_channels,
            kernel_size=kernel_size,
            dropout=dropout
        )
        
        # Final fully connected layer
        self.fc = torch.nn.Linear(hidden_channels[-1] * 2, num_classes)  # Combine both outputs

    def forward(self, imu_input, video_input):
        # Pass through the IMU TCN
        imu_out = self.imu_tcn(imu_input)  # Shape: (batch_size, hidden_dim, sequence_length)
        imu_out = torch.mean(imu_out, dim=2)  # Global Average Pooling: (batch_size, hidden_dim)
        
        # Pass through the Video TCN
        video_out = self.video_tcn(video_input)  # Shape: (batch_size, hidden_dim, sequence_length)
        video_out = torch.mean(video_out, dim=2)  # Global Average Pooling: (batch_size, hidden_dim)
        
        # Concatenate both TCN outputs
        combined_out = torch.cat((imu_out, video_out), dim=1)  # Shape: (batch_size, hidden_dim * 2)

        # Final fully connected layer
        out = self.fc(combined_out)  # Shape: (batch_size, num_classes)
        return out







'''
same_testing_acc
'''

# from pytorch_tcn import TCN

# class MultimodalTCN(torch.nn.Module):
#     def __init__(self, imu_input_channels, video_input_channels, num_classes, 
#                  hidden_channels=[64, 64, 64], kernel_size=5, dropout=0.2, track_running_stats=True):
#         super(MultimodalTCN, self).__init__()

#         # TCN for IMU data
#         self.imu_tcn = self._build_tcn(imu_input_channels, hidden_channels, kernel_size, dropout, track_running_stats)

#         # TCN for Video data
#         self.video_tcn = self._build_tcn(video_input_channels, hidden_channels, kernel_size, dropout, track_running_stats)

#         # Final fully connected layer
#         self.fc = torch.nn.Linear(hidden_channels[-1] * 2, num_classes)  # Combine both outputs

#     def _build_tcn(self, num_inputs, hidden_channels, kernel_size, dropout, track_running_stats):
#         # Build the TCN with BatchNorm layers that have track_running_stats set
#         return TCN(
#             num_inputs=num_inputs,
#             num_channels=hidden_channels,
#             kernel_size=kernel_size,
#             dropout=dropout,
#             use_norm=lambda num_features: torch.nn.BatchNorm1d(
#                 num_features, track_running_stats=track_running_stats
#             )  # Override BatchNorm behavior
#         )

#     def forward(self, imu_input, video_input):
#         # Pass through the IMU TCN
#         imu_out = self.imu_tcn(imu_input)  # Shape: (batch_size, hidden_dim, sequence_length)
#         imu_out = torch.mean(imu_out, dim=2)  # Global Average Pooling: (batch_size, hidden_dim)

#         # Pass through the Video TCN
#         video_out = self.video_tcn(video_input)  # Shape: (batch_size, hidden_dim, sequence_length)
#         video_out = torch.mean(video_out, dim=2)  # Global Average Pooling: (batch_size, hidden_dim)

#         # Concatenate both TCN outputs
#         combined_out = torch.cat((imu_out, video_out), dim=1)  # Shape: (batch_size, hidden_dim * 2)

#         # Final fully connected layer
#         out = self.fc(combined_out)  # Shape: (batch_size, num_classes)
#         return out









'''
k-fold cross val
'''

# def cross_validate_model(model_class, train_dataset, test_loader, input_channels, num_classes,
#                          criterion, optimizer_class, num_epochs=20, batch_size=32, k_folds=5):
#     """
#     Perform k-fold cross-validation on the training dataset and evaluate the final model 
#     on the separate test dataset.
#     """
#     kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
#     fold_accuracies = []

#     for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset)):
#         print(f"Fold {fold + 1}/{k_folds}")

#         # Create Subsets for the current fold
#         train_subset = Subset(train_dataset, train_idx)
#         val_subset = Subset(train_dataset, val_idx)

#         # Create DataLoaders for the fold
#         train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
#         val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

#         # Initialize a new model and optimizer for each fold
#         model = model_class(input_channels=input_channels, num_classes=num_classes)
#         optimizer = optimizer_class(model.parameters())

#         # Train the model and get the best accuracy for the fold
#         best_accuracy = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs)
#         fold_accuracies.append(best_accuracy)

#     # Calculate and print the average accuracy across folds
#     avg_accuracy = np.mean(fold_accuracies)
#     print(f"\nAverage Cross-Validation Accuracy: {avg_accuracy:.4f}")

#     # Final evaluation on the separate test dataset
#     final_model = model_class(input_channels=input_channels, num_classes=num_classes)
#     final_optimizer = optimizer_class(final_model.parameters())
#     train_model(final_model, DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
#                 test_loader, criterion, final_optimizer, num_epochs)

#     # Evaluate on the test dataset and plot confusion matrix
#     test_accuracy, (preds, labels) = evaluate_model(final_model, test_loader)
#     print(f"\nFinal Test Accuracy: {test_accuracy:.4f}")
#     plot_confusion_matrix(preds, labels, class_names=[str(i) for i in range(num_classes)])


# def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20):
#     """Train the model and return the best validation accuracy."""
#     model.train()
#     best_val_accuracy = 0.0
#     train_accuracies, val_accuracies = [], []

#     for epoch in range(num_epochs):
#         total_loss, correct_train, total_train = 0.0, 0, 0

#         for X_batch, y_batch in train_loader:
#             X_batch, y_batch = X_batch.float(), y_batch.long()
#             optimizer.zero_grad()
#             outputs = model(X_batch)
#             loss = criterion(outputs, y_batch)
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()
#             _, predicted = torch.max(outputs, 1)
#             correct_train += (predicted == y_batch).sum().item()
#             total_train += y_batch.size(0)

#         train_accuracy = correct_train / total_train
#         train_accuracies.append(train_accuracy)

#         # Calculate validation accuracy
#         val_accuracy, _ = evaluate_model(model, val_loader)
#         val_accuracies.append(val_accuracy)

#         print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}, "
#               f"Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}")

#         if val_accuracy > best_val_accuracy:
#             best_val_accuracy = val_accuracy

#     # Plot training and validation accuracy
#     plot_training_progress(train_accuracies, val_accuracies, num_epochs)

#     return best_val_accuracy

# def evaluate_model(model, loader):
#     """Evaluate the model and return accuracy and predictions."""
#     model.eval()
#     all_preds, all_labels = [], []

#     with torch.no_grad():
#         for X_batch, y_batch in loader:
#             X_batch = X_batch.float()
#             outputs = model(X_batch)
#             _, predicted = torch.max(outputs, 1)
#             all_preds.extend(predicted.cpu().numpy())
#             all_labels.extend(y_batch.cpu().numpy())

#     accuracy = accuracy_score(all_labels, all_preds)
#     return accuracy, (all_preds, all_labels)

# def plot_training_progress(train_accuracies, val_accuracies, num_epochs):
#     """Plot the training and validation accuracy across epochs."""
#     plt.plot(range(1, num_epochs + 1), train_accuracies, label="Train Accuracy")
#     plt.plot(range(1, num_epochs + 1), val_accuracies, label="Validation Accuracy")
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy')
#     plt.legend()
#     plt.show()

# def plot_confusion_matrix(predictions, labels, class_names):
#     """Plot a confusion matrix for the given predictions and labels."""
#     cm = confusion_matrix(labels, predictions)
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
#     plt.xlabel('Predicted')
#     plt.ylabel('True')
#     plt.title('Confusion Matrix')
#     plt.show()

#     # Calculate precision, recall, and F1-score
#     precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
#     print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")


