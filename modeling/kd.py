import glob
import os
import sys
import argparse
import json
import torch 
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

sys.path.append(os.path.dirname(os.getcwd()))
import utils.modeling as models
from utils.loadutils import *


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
parser.add_argument("--early_stop_patience", type=int, required=True) 
args = parser.parse_args()


def distillation_loss(student_logits, teacher_logits, true_labels):
    alpha = 0.7
    temperature = 5
    hard_loss = nn.CrossEntropyLoss()(student_logits, true_labels)

    soft_teacher_probs = nn.functional.softmax(teacher_logits / temperature, dim=1)
    soft_student_probs = nn.functional.log_softmax(student_logits / temperature, dim=1)
    distill_loss = nn.functional.kl_div(soft_student_probs, soft_teacher_probs, reduction='batchmean') * (temperature ** 2)

    return alpha * distill_loss + (1 - alpha) * hard_loss

def test_model(student_model, test_loader, return_accuracy=False):
    student_model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for vid_batch, _, y_batch in test_loader:
            vid_batch = vid_batch.float()
            outputs = student_model(vid_batch)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=1)
    return accuracy, precision, recall, f1

def train_distillation(student_model, teacher_model, train_loader, test_loader, distillation_loss, optimizer, scheduler, modelname, num_epochs=20, model_path=None):

    if model_path and os.path.exists(model_path):
        student_model.load_state_dict(torch.load(model_path, weights_only=True))
        print(f"Loaded student model from {model_path}")

    student_model.train()
    best_accuracy = 0.0
    best_model_filename = None
    no_improvement_epochs = 0
    train_accuracies = []
    test_accuracies = []

    for epoch in range(num_epochs):
        total_loss = 0.0
        correct_train = 0
        total_train = 0

        for vid_batch, imu_batch, y_batch in train_loader:
            vid_batch = vid_batch.float()
            imu_batch = imu_batch.float()
            y_batch = y_batch.long()
            optimizer.zero_grad()

            with torch.no_grad():
                teacher_logits = teacher_model(imu_batch, vid_batch)

            student_logits = student_model(vid_batch)

            loss = distillation_loss(student_logits, teacher_logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(student_logits, 1)
            correct_train += (predicted == y_batch).sum().item()
            total_train += y_batch.size(0)

        train_accuracy = correct_train / total_train
        train_accuracies.append(train_accuracy)

        test_accuracy, precision, recall, f1 = test_model(student_model, test_loader)
        test_accuracies.append(test_accuracy)

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
            best_model_filename = f'{args.model_out_path}/best_{modelname}_{args.depth}_{args.secs}_{args.seed}_acc_{best_accuracy:.4f}.pth'
            torch.save(student_model.state_dict(), best_model_filename)
            print(f"New best model saved as {best_model_filename}")
            no_improvement_epochs = 0  
        else:
            no_improvement_epochs += 1 

        if no_improvement_epochs >= args.early_stop_patience:
            print(f"Early stopping triggered after {no_improvement_epochs} epochs without improvement.")
            break


def main():
    secs = args.secs
    imu_window_size = int(secs * 50)
    video_window_size = int(secs * 30)
    print('IMU Data window size:', imu_window_size)
    print('Video Data window size:', video_window_size)
    activities = json.loads(args.activities)
    num_classes = len(activities)
    fold = 1
    set_random_seed(args.seed)

    train_loader, test_loader = standardize_vidimu(args.dpath, 
                                                   time_in_seconds=secs, 
                                                   split=0.8, 
                                                   batch_size=32,
                                                   activities = activities,
                                                   ins='VIDIMU', 
                                                   out='act')
    
    for vid_batch, imu_batch, _ in train_loader:
        vid_channels = vid_batch.shape[1]
        imu_channels = imu_batch.shape[1]
        break

    student_model = models.TemporalResNet(input_channels=vid_channels, num_classes=num_classes, num_blocks=args.depth)
    teacher_model = models.TemporalResNetMultiInput(imu_channels, vid_channels, num_classes, num_blocks=args.depth)

    teacher_name = f'/Volumes/Data_Drive/vidimu_gridsearch_out/gridsearch_11_4/best_AngPos_ResNet_fold{fold}_{args.depth}_{args.secs}_{args.seed}_acc_'
    teacher_path = glob.glob(teacher_name+'*')[0]
    # print('TEACHER PATH ------> ', teacher_path)
    student_name = f'/Volumes/Data_Drive/vidimu_gridsearch_out/gridsearch_11_4/best_Pos_ResNet_fold{fold}_{args.depth}_{args.secs}_{args.seed}_acc_'
    student_path = glob.glob(student_name+'*')[0]
    # print('STUDENT PATH ------> ', student_path)

    teacher_model.load_state_dict(torch.load(teacher_path, weights_only=True))
    teacher_model.eval()  

    epochs = args.epochs
    optimizer = optim.Adam(student_model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=args.factor, patience=args.patience)

    train_distillation(student_model, teacher_model, train_loader, test_loader, distillation_loss, optimizer, scheduler, modelname='Pos_ResNetKD', num_epochs=epochs, model_path=student_path)

if __name__ == "__main__":
    main()